"""
Package CLIP features for center images
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

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/Data/Things-EEG2/Image_set', type=str)
args = parser.parse_args()

print('Extract feature maps CLIP of images for center <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)

# =============================================================================
# Select the layers of interest and import the model
# =============================================================================
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

img_set_dir = os.path.join(args.project_dir, 'image_set/center_images/')
condition_list = os.listdir(img_set_dir)
condition_list.sort()

all_centers = []

for cond in condition_list:
    one_cond_dir = os.path.join(args.project_dir, 'image_set/center_images/', cond)
    cond_img_list = os.listdir(one_cond_dir)
    cond_img_list.sort()
    cond_center = []
    for img in cond_img_list:
        img_path = os.path.join(one_cond_dir, img)
        img = Image.open(img_path).convert('RGB')
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=img, return_tensors="pt", padding=True)
        inputs.data['pixel_values'].cuda()
        with torch.no_grad():
            outputs = model(**inputs).image_embeds
    # * for mean center
        cond_center.append(np.squeeze(outputs.detach().cpu().numpy()))
    all_centers.append(np.array(cond_center))

np.save(os.path.join(args.project_dir, 'center_all_image_clip.npy'), all_centers)
