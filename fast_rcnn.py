import torch
import torch.nn as nn
import torchvision

from .roipooling import ROIPooling


class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        vgg = torchvision.models.vgg19_bn(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])
        self.roipool = ROIPooling(output_size=(7, 7))
        self.output = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.prob = nn.Linear(4096, num_classes+1)
        self.loc = nn.Linear(4096, 4 * (num_classes + 1))

        self.cat_loss = nn.CrossEntropyLoss()
        self.loc_loss = nn.SmoothL1Loss()

    def forward(self, img, rois, roi_idx):
        """

        :param img: img:批次内的图像
        :param rois: rois:[[单张图片内框体],[],[]]
        :param roi_idx: [2]*6------->[2, 2, 2, 2, 2, 2]
        :return:
        """
        res = self.features(img)
        res = self.roipool(res, rois, roi_idx)
        res = res.view(res.shape[0], -1)
        features = self.output(res)
        prob = self.prob(features)
        loc = self.loc(features).view(-1, self.num_classes+1, 4)
        return prob, loc
    
    def loss(self, prob, bbox, label, gt_bbox, lmb=1.0):
        """

        :param prob: 预测类别
        :param bbox:预测边界框
        :param label:真实类别
        :param gt_bbox:真实边界框
        :param lmb:
        :return:
        """
        loss_cat = self.cat_loss(prob, label)
        lbl = label.view(-1, 1, 1).expand(label.size(0), 1, 4)
        mask = (label != 0).float().view(-1, 1, 1).expand(label.shape[0], 1, 4)
        loss_loc = self.loc_loss(gt_bbox * mask, bbox.gather(1, lbl).squeeze(1) * mask)
        loss = loss_cat + lmb * loss_loc
        return loss, loss_cat, loss_loc

