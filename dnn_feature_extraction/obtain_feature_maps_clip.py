"""
Obtain CLIP features of training and test images in Things-EEG.
"""

import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image
import requests

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import AutoImageProcessor, ViTForImageClassification, ViTFeatureExtractor, ViTModel
from transformers import CLIPProcessor, CLIPModel
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/Data/Things-EEG2/', type=str)
args = parser.parse_args()

print('Extract feature maps CLIP <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

# image preprocessing

# centre_crop = trn.Compose([
# 	trn.Resize((224, 224)),
# 	trn.ToTensor(),
# 	# trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 	trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Image directories
img_set_dir = os.path.join(args.project_dir, 'Image_set/image_set')
img_partitions = os.listdir(img_set_dir)
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
		'full_feature_maps', 'clip', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	# * better to use a dataloader
	for i, image in enumerate(image_list):
		img = Image.open(image).convert('RGB')
		inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=img, return_tensors="pt", padding=True)
		inputs.data['pixel_values'].cuda()
		x = model(**inputs).image_embeds
		feats = x.detach().cpu().numpy()
		# for f, feat in enumerate(x):
		# 	feats[model.feat_list[f]] = feat.data.cpu().numpy()
		file_name = p + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), feats)
