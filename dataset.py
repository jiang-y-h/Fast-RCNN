import json
import os
import random

import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import math


class COCOdataset(Dataset):
    def __init__(self, dir='/data/COCO2017/', mode='val',
                 transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, -.406], [0.229, 0.224, 0.225])])):
        assert mode in ['train', 'val'], 'mode must be \'train\' or \'val\''
        self.dir = dir
        self.mode = mode
        self.transform = transform
        with open(os.path.join(self.dir, '%s.json' % self.mode), 'r', encoding='utf-8') as f:
            self.ss_regions = json.load(f)
        self.img_dir = os.path.join(self.dir, '%s2017' % self.mode)

    def __len__(self):
        return len(self.ss_regions)

    def __getitem__(self, i, max_num_pos=8, max_num_neg=16):
        img = PIL.Image.open(os.path.join(self.img_dir, '%012d.jpg' %
                                     self.ss_regions[i]['id']))
        img = img.convert('RGB')
        img = img.resize([224, 224])
        pos_regions = self.ss_regions[i]['pos_regions']
        neg_regions = self.ss_regions[i]['neg_regions']
        if self.transform != None:
            img = self.transform(img)
        if len(pos_regions) > max_num_pos:
            pos_regions = random.sample(pos_regions, max_num_pos)
        if len(neg_regions) > max_num_neg:
            neg_regions = random.sample(neg_regions, max_num_neg)
        regions = pos_regions + neg_regions
        random.shuffle(regions)
        rects = []
        rela_locs = []
        cats = []
        for region in regions:
            rects.append(region['rect'])
            p_rect = region['rect']
            g_rect = region['gt_rect']
            t_x = (g_rect[0] + g_rect[2] - p_rect[0] - p_rect[2]) / 2 / (p_rect[2] - p_rect[0])
            t_y = (g_rect[1] + g_rect[3] - p_rect[1] - p_rect[3]) / 2 / (p_rect[3] - p_rect[1])
            t_w = math.log((g_rect[2] - g_rect[0]) / (p_rect[2] - p_rect[0]))
            t_h = math.log((g_rect[3] - g_rect[1]) / (p_rect[3] - p_rect[1]))
            rela_locs.append([t_x, t_y, t_w, t_h])
            cats.append(region['category'])
        roi_idx_len = len(regions)
        return img, rects, roi_idx_len, rela_locs, cats

# dataset = COCOdataset()
# print(dataset[1][0].shape)
# print(dataset[1][1])
# from torch.utils.data import DataLoader
# dataloader = DataLoader(dataset, batch_size=2)
# print(next(iter(dataloader))[1])

if __name__ == '__main__':
    dataset = COCOdataset()
    print(dataset.__len__())
    img, rects, roi_idx_len, rela_locs, cats = dataset.__getitem__(10)
    print(img, rects, roi_idx_len, rela_locs, cats)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1)
    for i, temp in enumerate(loader):
        print(i,type(temp))

