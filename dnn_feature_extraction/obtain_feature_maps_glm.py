"""
Obtain visualGLM description and CLIP_text features of training and test images in Things-EEG.

using pretrained visualGLM model - the description is not good enough 
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
import re

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModel

import requests

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/Documents/Data/Things-EEG2/', type=str)
args = parser.parse_args()

print('Extract feature maps VisualGLM - CLIP_text <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# * CLIP_text
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# * VisualGLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
glm_model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()

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
	save_text_dir = os.path.join(args.project_dir, 'Description',
		'visualGLM1', p)
	if os.path.isdir(save_text_dir) == False:
		os.makedirs(save_text_dir)
	save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
		'full_feature_maps', 'glm_text1', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	# * better to use a dataloader
	for i, image in enumerate(image_list):
		# TODO: image description in to txt
		label_text = image.split('/')[-2][6:]
		response, history = glm_model.chat(tokenizer, image, "Q: Describe the appearance of the " + label_text + " in one sentence. A:", history=[])
		if i == 0:
			reponse, history = glm_model.chat(tokenizer, image, "Q: Describe the appearance of the " + label_text + " in one sentence. A:", history=[])

		# detect how many . in the sentence of response
		if response.count('.') >= 1:
			response = response.split('.')[0] + '. ' + response.split('.')[1] + '.' # save the front two sentences but damage the words like U.S.

		response = re.sub('[\u4e00-\u9fa5]', '', response) # discard Chinese characters
		# response = re.sub('[＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。°]', '', response) # discard special characters
		response = re.sub("[^A-Za-z0-9,. ]", "", response) # just leave the English words and numbers
		# just save the front 70 words in the paragraph response
		response = response[:210] # save front 77 words
		np.savetxt(os.path.join(save_text_dir, p + '_' + format(i+1, '07') + '.txt'), [response], fmt='%s')

		img = Image.open(image).convert('RGB')
		label_text = image.split('/')[-2][6:]
		inputs = processor(text=[response], images=img, return_tensors="pt", padding=True)
		inputs.data['pixel_values'].cuda()
		x = model(**inputs).text_embeds
		feats = x.detach().cpu().numpy()
		# for f, feat in enumerate(x):
		# 	feats[model.feat_list[f]] = feat.data.cpu().numpy()
		file_name = p + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), feats)
