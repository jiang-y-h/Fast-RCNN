import argparse

import numpy as np
import skimage
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from selectivesearch import selective_search
from torchvision import transforms

from fast_rcnn import FastRCNN


def cal_iou(a, b):
    a_min_x, a_min_y, a_max_x, a_max_y = a
    b_min_x, b_min_y, b_max_x, b_max_y = b
    if min(a_max_y, b_max_y) < max(a_min_y, b_min_y) or min(a_max_x, b_max_x) < max(a_min_x, b_min_x):
        return 0
    else:
        intersect_area = (min(a_max_y, b_max_y) - max(a_min_y, b_min_y) + 1) * \
            (min(a_max_x, b_max_x) - max(a_min_x, b_min_x) + 1)
        union_area = (a_max_x - a_min_x + 1) * (a_max_y - a_min_y + 1) + \
            (b_max_x - b_min_x + 1) * (b_max_y - b_min_y + 1) - intersect_area
    return intersect_area / union_area


def main():
    parser = argparse.ArgumentParser('parser for testing fast-rcnn')
    parser.add_argument('--jpg_path', type=str,
                        default='/devdata/project/ai_learn/COCO2017/val2017/000000241326.jpg')
    parser.add_argument('--save_path', type=str, default='sample.png')
    parser.add_argument('--save_type', type=str, default='png')
    parser.add_argument('--model', type=str, default='./model/fast_rcnn.pkl')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--scale', type=float, default=30.0)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--min_size', type=int, default=50)
    parser.add_argument('--cats', type=str, nargs='*', default=[
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    trained_net = torch.load(args.model)
    model = FastRCNN(num_classes=args.num_classes)
    model.load_state_dict(trained_net)
    if args.cuda:
        model.cuda()

    img = skimage.io.imread(args.jpg_path)
    h = img.shape[0]
    w = img.shape[1]
    _, ss_regions = selective_search(
        img, args.scale, args.sigma, args.min_size)
    rois = []
    for region in ss_regions:
        rect = list(region['rect'])
        rect[0] = rect[0] / w
        rect[1] = rect[1] / h
        rect[2] = rect[0] + rect[2] / w
        rect[3] = rect[1] + rect[3] / h
        rois.append(rect)
    img = Image.fromarray(img)
    img_tensor = img.resize([224, 224])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([
                                   0.485, 0.456, -.406], [0.229, 0.224, 0.225])])
    img_tensor = transform(img_tensor).unsqueeze(0)
    if args.cuda:
        img_tensor = img_tensor.cuda()
    rois = np.array(rois)
    roi_idx = [0] * rois.shape[0]

    prob, rela_loc = model.forward(img_tensor, rois, roi_idx)
    prob = torch.nn.Softmax(dim=-1)(prob).cpu().detach().numpy()
    # rela_loc = rela_loc.cpu().detach().numpy()[:, 1:, :].mean(axis=1)
    labels = []
    max_probs = []
    bboxs = []
    for i in range(len(prob)):
        if prob[i].max() > 0.8 and np.argmax(prob[i], axis=0) != 0:
            # proposal regions is directly used because of limited training epochs, bboxs predicted are not precise
            # bbox = [(rois[i][2] - rois[i][0]) * rela_loc[i][0] + 0.5 * (rois[i][2] + rois[i][0]),
            #         (rois[i][3] - rois[i][1]) * rela_loc[i][1] + 0.5 * (rois[i][3] + rois[i][1]),
            #         np.exp(rela_loc[i][2]) * rois[i][2],
            #         np.exp(rela_loc[i][3]) * rois[i][3]]
            # bbox = [bbox[0] - 0.5 * bbox[2],
            #         bbox[1] - 0.5 * bbox[3],
            #         bbox[0] + 0.5 * bbox[2],
            #         bbox[1] + 0.5 * bbox[3]]
            labels.append(np.argmax(prob[i], axis=0))
            max_probs.append(prob[i].max())
            rois[i] = [int(w * rois[i][0]), int(h * rois[i][1]),
                       int(w * rois[i][2]), int(w * rois[i][3])]
            bboxs.append(rois[i])
    labels = np.array(labels)
    max_probs = np.array(max_probs)
    bboxs = np.array(bboxs)
    order = np.argsort(-max_probs)
    labels = labels[order]
    max_probs = max_probs[order]
    bboxs = bboxs[order]

    nms_labels = []
    nms_probs = []
    nms_bboxs = []
    del_indexes = []
    for i in range(len(labels)):
        if i not in del_indexes:
            for j in range(len(labels)):
                if j not in del_indexes and cal_iou(bboxs[i], bboxs[j]) > 0.4:
                    del_indexes.append(j)
            nms_labels.append(labels[i])
            nms_probs.append(max_probs[i])
            nms_bboxs.append(bboxs[i])

    cat_dict = {(i + 1): args.cats[i] for i in range(len(args.cats))}
    cat_dict[0] = 'background'
    font = ImageFont.truetype('./fonts/chinese_cht.ttf', size=16)
    draw = ImageDraw.Draw(img)
    for i in range(len(nms_labels)):
        draw.polygon([(nms_bboxs[i][0], nms_bboxs[i][1]), (nms_bboxs[i][2], nms_bboxs[i][1]),
                      (nms_bboxs[i][2], nms_bboxs[i][3]), (nms_bboxs[i][0], nms_bboxs[i][3])], outline=(255, 0, 0))
        draw.text((nms_bboxs[i][0] + 5, nms_bboxs[i][1] + 5), '%s %.2f%%' % (
            cat_dict[nms_labels[i]], 100 * max_probs[i]), fill=(255, 0, 0), font=font)
    img.save(args.save_path, args.save_type)


if __name__ == '__main__':
    main()
