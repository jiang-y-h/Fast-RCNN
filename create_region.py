import argparse
import json
import os
import random
import sys
import time

from progressbar import *
from pycocotools.coco import COCO
from selectivesearch import selective_search
from skimage import io, util, color


def cal_iou(a, b):
    a_min_x, a_min_y, a_delta_x, a_delta_y = a
    b_min_x, b_min_y, b_delta_x, b_delta_y = b
    a_max_x = a_min_x + a_delta_x
    a_max_y = a_min_y + a_delta_y
    b_max_x = b_min_x + b_delta_x
    b_max_y = b_min_y + b_delta_y
    if min(a_max_y, b_max_y) < max(a_min_y, b_min_y) or min(a_max_x, b_max_x) < max(a_min_x, b_min_x):
        return 0
    else:
        intersect_area = (min(a_max_y, b_max_y) - max(a_min_y, b_min_y) + 1) * \
            (min(a_max_x, b_max_x) - max(a_min_x, b_min_x) + 1)
        union_area = (a_delta_x + 1) * (a_delta_y + 1) + \
            (b_delta_x + 1) * (b_delta_y + 1) - intersect_area
        return intersect_area / union_area


def ss_img(img_id, coco, cat_dict, args):
    img_path = os.path.join(args.data_dir, args.mode +
                            '2017', '%012d.jpg' % img_id)
    coco_dict = {cat['id']: cat['name']
                 for cat in coco.loadCats(coco.getCatIds())}
    img = io.imread(img_path)
    if img.ndim == 2:    # Python 中灰度图的 img.ndim = 2
        img = color.gray2rgb(img)
    _, ss_regions = selective_search(
        img, args.scale, args.sigma, args.min_size)         # 'rect': (left, top, width, height)
    anns = coco.loadAnns(coco.getAnnIds(
        imgIds=[img_id], catIds=coco.getCatIds(catNms=args.cats)))
    pos_regions = []
    neg_regions = []
    h = img.shape[0]
    w = img.shape[1]
    for region in ss_regions:
        for ann in anns:
            iou = cal_iou(region['rect'], ann['bbox'])
            if iou >= 0.1:
                rect = list(region['rect'])
                rect[0] = rect[0] / w
                rect[1] = rect[1] / h
                rect[2] = rect[0] + rect[2] / w
                rect[3] = rect[1] + rect[3] / h
                gt_rect = list(ann['bbox'])
                gt_rect[0] = gt_rect[0] / w
                gt_rect[1] = gt_rect[1] / h
                gt_rect[2] = gt_rect[0] + gt_rect[2] / w
                gt_rect[3] = gt_rect[1] + gt_rect[3] / h
                if iou >= 0.5:
                    pos_regions.append({'rect': rect, 
                                        'gt_rect': gt_rect,
                                        'category': cat_dict[coco_dict[ann['category_id']]]})
                else:
                    neg_regions.append({'rect': rect, 
                                        'gt_rect': gt_rect,
                                        'category': 0})
    return pos_regions, neg_regions


def main():
    parser = argparse.ArgumentParser('parser to create regions')
    parser.add_argument('--data_dir', type=str, default='/devdata/project/ai_learn/COCO2017/')
    parser.add_argument('--mode', type=str, default='val')   # train/val
    parser.add_argument('--save_dir', type=str, default='/devdata/project/ai_learn/COCO2017/')
    parser.add_argument('--cats', type=str, nargs='*', default=[
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])
    parser.add_argument('--scale', type=float, default=30.0)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--min_size', type=int, default=50)
    args = parser.parse_args()
    coco = COCO(os.path.join(args.data_dir, 'annotations',
                             'instances_%s2017.json' % args.mode))
    cat_dict = {args.cats[i]: i+1 for i in range(len(args.cats))}
    cat_dict['background'] = 0

    # get relavant image ids
    if args.mode == 'train':
        num_cat = 400
    if args.mode == 'val':
        num_cat = 100
    img_ids = []
    cat_ids = coco.getCatIds(catNms=args.cats)
    for cat_id in cat_ids:
        cat_img_ids = coco.getImgIds(catIds=[cat_id])
        if len(cat_img_ids) > num_cat:
            cat_img_ids = random.sample(cat_img_ids, num_cat)
        img_ids += cat_img_ids
    img_ids = list(set(img_ids))

    # selective_search each image
    # [{'id': 1, 'pos_regions':[...], 'neg_regions':[...]}, ...]

    num_imgs = len(img_ids)
    ss_regions = []
    p = ProgressBar(widgets=['Progress: ', Percentage(),
                             ' ', Bar('#'), ' ', Timer(), ' ', ETA()])
    for i in p(range(num_imgs)):
        img_id = img_ids[i]
        pos_regions, neg_regions = ss_img(img_id, coco, cat_dict, args)
        ss_regions.append({'id': img_id,
                           'pos_regions': pos_regions,
                           'neg_regions': neg_regions})

    # save
    with open(os.path.join(args.save_dir, '%s.json' % args.mode), 'w', encoding='utf-8') as f:
        json.dump(ss_regions, f)


if __name__ == '__main__':
    main()
