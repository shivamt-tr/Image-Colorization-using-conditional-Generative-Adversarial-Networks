# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:21:36 2022

@author: tripa
"""

import os
from PIL import Image

files = os.listdir(os.path.join(os.getcwd(), 'data', 'test'))
res_dir = os.path.join(os.getcwd(), 'resized')
os.makedirs(res_dir, exist_ok=True)

for f in files:
    image = Image.open(os.path.join(os.getcwd(), 'data', 'test', f)).convert("RGB")
    image = image.resize((512, 512))
    image.save(os.path.join(res_dir, f))
    