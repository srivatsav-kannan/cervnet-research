import os
import shutil

img_dir = '/Users/srivatsavkannan/Datasets/CervicalNew/All'
img_dir_cur = '/Users/srivatsavkannan/Datasets/CervicalNew/json'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON'
for img in os.listdir(img_dir):
    if img.endswith('.png'):
        new_d = json_dir+'/'+(img[:-4])+'.json'
        print(new_d)
        shutil.copyfile(new_d, img_dir_cur+'/'+(img[:-4])+'.json')