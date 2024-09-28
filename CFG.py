# import requests
import random
import os
import json
from PIL import Image
import cv2
import torch
import albumentations as A
import numpy as np
import pdb


class CFG:
    seed          = 42 
    debug         = False 
    exp_name      = 'open-ended-medvqa'
    comment       = 'Parameter-efficient finetuning of multimodal VQA models for dermatology'
    output_dir    = './'
    model_name    = 'CLIP-GPT'
    train_bs      = 1
    valid_bs      = 1
    img_size      = [224, 224]#[128,128]# [512, 512]
    epochs        = 3 ## increase to 15
    n_accumulate  = max(1, 64//train_bs)
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(2279/(train_bs*n_accumulate)*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_fold        = 5
    num_classes   = 1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rescale       = np.array([img_size[0]//2,img_size[0]//2,
                              img_size[0]//2,img_size[0]//2])

    train_json    = 'mediqa-m3g-startingkit-v2/mediqa-m3-clinicalnlp2024/train.json'
    valid_json    = 'mediqa-m3g-startingkit-v2/mediqa-m3-clinicalnlp2024/valid_inputonly.json'
    train_groups = 'mediqa-m3g-startingkit-v2/images_final/images_train'
    valid_groups = 'mediqa-m3g-startingkit-v2/images_final/images_valid'
    test_json    = 'm3g-test-allinputs-v2/input.json'
    test_groups = 'm3g-test-allinputs-v2/images_test'
    
    loss_func     = "BLEU"

    data_transforms = {
        "train": A.Compose([
        A.Resize(*img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),  # Randomly adjust brightness and contrast
        A.RandomRotate90(p=0.5),  # Randomly rotate the image by 90 degrees
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),  # Randomly shift, scale, and rotate
        A.RandomResizedCrop(*img_size,ratio=(0.95, 1.05), p=0.5),  # Randomly crop and resize
        A.GaussNoise(var_limit=(0.0, 0.01), p=0.1),  # Add Gaussian noise
        A.Blur(blur_limit=(0, 1), p=0.2),  # Apply blur
        # A.HueSaturationValue(p=0.3),  # Adjust hue, saturation, and value
    ], p=1.0), 
        "valid": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }