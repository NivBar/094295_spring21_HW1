#!/usr/bin/env python
# coding: utf-8

import json
import os
from PIL import Image
import glob
import torch
import torchvision



image_list = []
for filename in glob.glob('test/*.jpg'):
    orig, filename_ = filename.split('/')
    id_, bbox_, label_ = filename_.split('__')
    bbox_ = json.loads(bbox_)
    if any(i <= 0 for i in bbox_):
        print(bbox_)
        im=Image.open(filename)
        image_list.append(im)
        print(filename)
#         display(im)
        os.replace(filename, f"test_problematic/{filename_}") 


# In[ ]:


image_list = []
for filename in glob.glob('train/*.jpg'):
    orig, filename_ = filename.split('/')
    id_, bbox_, label_ = filename_.split('__')
    bbox_ = json.loads(bbox_)
    if any(i <= 0 for i in bbox_):
        print(bbox_)
        im=Image.open(filename)
        image_list.append(im)
        print(filename)
#         display(im)
        os.replace(filename, f"train_problematic/{filename_}") 





