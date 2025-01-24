import os
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from numpy import cov, trace, iscomplexobj


model = inception_v3(pretrained=True, transform_input=False)
model.fc = nn.Identity()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_fid(mu1, sigma1, mu2, sigma2):

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_activation(image_paths):

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img).unsqueeze(0)
        images.append(img)

    images = torch.cat(images).to(device)
    
    with torch.no_grad():
        activations = model(images).cpu().numpy()
    
    return activations

def get_image_paths(base_dir):

    all_image_paths = glob(os.path.join(base_dir, '**/*.jpg'), recursive=True) + \
                      glob(os.path.join(base_dir, '**/*.png'), recursive=True)
    return sorted(all_image_paths)

def calculate_fid_for_expansion(train_dir, expansion_dir):

    train_image_paths = get_image_paths(train_dir)
    
    fids = []
    
    for train_img in train_image_paths:

        class_name = os.path.basename(os.path.dirname(train_img))
        img_name = os.path.basename(train_img).replace('.jpg', '').replace('.png', '')
        matched_expansion_imgs = [os.path.join(expansion_dir, class_name, img_name + f'_expand_{i}.png') for i in range(5)]
        real_act = calculate_activation([train_img])
        fake_act = calculate_activation(matched_expansion_imgs)
        mu_real, sigma_real = real_act.mean(axis=0), cov(real_act, rowvar=False)
        mu_fake, sigma_fake = fake_act.mean(axis=0), cov(fake_act, rowvar=False)
        fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        fids.append(fid)
    
    return np.mean(fids)

train_dir = 'data/m_rock/train'
expansion_dir = 'data/m_rock_expansion/save/distdiff_batch_5x(03)'

fid_score = calculate_fid_for_expansion(train_dir, expansion_dir)
print(f'FID Score: {fid_score}')
