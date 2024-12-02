"""
NICE++ 
THINGS-EEG1
"""
import os
import argparse
import math
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable


from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/home/NICEpp/results/eeg1/' 
model_idx = 'test_eeg1'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
# parser.add_argument('data', metavar='DIR', nargs='?', default='/home/songyonghao/Documents/Data/IE/stimuli',
#                     help='path to dataset (default: imagenet)')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--model_text', default='minigpt_text', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=50, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (64, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class FlattenHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead(emb_size, n_classes)
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

# ! Important
class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/home/Data/Things-EEG1/Preprocessed_data/'
        self.img_data_path = '/home/Data/Things-EEG1/DNN_feature_maps/' + args.dnn + '/'
        self.txt_data_path = '/home/Data/Things-EEG1/DNN_feature_maps/' + args.model_text + '/'
        self.test_center_path = '/home/Data/Things-EEG1/DNN_feature_maps/' + args.dnn + '/'

        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print('initial define done.')


    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(self.eeg_data_path + '/eeg_sub-' + format(self.nSub, '02') + '_train.pkl', allow_pickle=True)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + '/eeg_sub-' + format(self.nSub, '02') + '_test.pkl', allow_pickle=True)
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        test_category = np.load('./Things-MEG1/test_category.npy', allow_pickle=True)
        test_idx = []
        for tc_idx in range(200):
            test_idx.append(12*(test_category[tc_idx]-1) + np.arange(12))
        test_idx = np.concatenate(test_idx)

        return train_data, train_label, test_data, test_label, test_idx

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feats_train.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feats_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature

    
    def get_text_data(self):
        train_txt_feature = np.load(self.txt_data_path + self.args.model_text + '_feats_train.npy', allow_pickle=True)
        test_txt_feature = np.load(self.txt_data_path + self.args.model_text + '_feats_test.npy', allow_pickle=True)

        train_txt_feature = np.squeeze(train_txt_feature)
        test_txt_feature = np.squeeze(test_txt_feature)

        return train_txt_feature, test_txt_feature

    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label, test_idx = self.get_eeg_data()
        train_img_feature, _ = self.get_image_data() 
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)
        train_txt_feature, _ = self.get_text_data()

        test_idx = np.int16(test_idx)
        train_eeg = np.delete(train_eeg, test_idx, axis=0)
        train_img_feature = np.delete(train_img_feature, test_idx, axis=0)
        train_txt_feature = np.delete(train_txt_feature, test_idx, axis=0)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]
        train_txt_feature = train_txt_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:748])
        val_image = torch.from_numpy(train_img_feature[:748])
        val_text = torch.from_numpy(train_txt_feature[:748])

        train_eeg = torch.from_numpy(train_eeg[748:])
        train_image = torch.from_numpy(train_img_feature[748:])
        train_text = torch.from_numpy(train_txt_feature[748:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image, train_text)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image, val_text)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        self.optimizer_eeg = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_img = torch.optim.Adam(self.Proj_img.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            for i, (eeg, img, txt) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                txt_features = Variable(txt.cuda().type(self.Tensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # --- update the image projector ---
                self.optimizer_img.zero_grad()
                img_features = self.Proj_img(img_features)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)
                txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)

                # contrative loss - image and text
                logit_scale = self.logit_scale.exp()
                logits_per_img_txt = logit_scale * img_features @ txt_features.t()
                logits_per_txt = logits_per_img_txt.t()

                loss_cos_it = (self.criterion_cls(logits_per_img_txt, labels) + self.criterion_cls(logits_per_txt, labels)) / 2
                loss_cos_it.backward()
                self.optimizer_img.step()

                # --- update the EEG encoder and projector ---
                self.optimizer_eeg.zero_grad()
                img_features.detach_()

                eeg_features = self.Enc_eeg(eeg)
                eeg_features = self.Proj_eeg(eeg_features)
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                # contrative loss - EEG and image
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_cos_ie = (self.criterion_cls(logits_per_eeg, labels) + self.criterion_cls(logits_per_img, labels)) / 2
                loss_cos_ie.backward()
                self.optimizer_eeg.step()

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg, vtxt) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        # TODO - how to validate
                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            torch.save(self.Enc_eeg.module.state_dict(), './model/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), './model/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), './model/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos img_txt: %.4f' % loss_cos_it.detach().cpu().numpy(),
                      '  Cos img_eeg: %.4f' % loss_cos_ie.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
 
                      )
                self.log_write.write('Epoch %d: Cos img_txt: %.4f, Cos img_eeg: %.4f, loss val: %.4f\n'
                    %(e, loss_cos_it.detach().cpu().numpy(), loss_cos_ie.detach().cpu().numpy(), vloss.detach().cpu().numpy()))


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model/' + model_idx + 'Proj_img_cls.pth'), strict=False)
        

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))            

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                all_center = self.Proj_img(all_center)
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                all_center = all_center / all_center.norm(dim=1, keepdim=True)

                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        
        return top1_acc, top3_acc, top5_acc

def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
    for i in range(num_sub):
        if i in [0, 5, 17, 22]:
            continue

        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        # i = 3
        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
