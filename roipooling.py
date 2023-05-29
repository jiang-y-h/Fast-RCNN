import numpy as np
import torch
import torch.nn as nn


class ROIPooling(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size

    def forward(self, imgs, rois, roi_idx):
        """
        :param img: img:批次内的图像
        :param rois: rois:[[单张图片内框体],[],[]]
        :param roi_idx: [2]*6------->[2, 2, 2, 2, 2, 2]
        :return:
        """
        n = rois.shape[0]
        h = imgs.shape[2]
        w = imgs.shape[3]

        x1 = rois[:, 0]
        y1 = rois[:, 1]
        x2 = rois[:, 2]
        y2 = rois[:, 3]

        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        res = []
        for i in range(n):
            img = imgs[roi_idx[i]].unsqueeze(dim=0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.maxpool(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res

if __name__ == '__main__':
    import numpy as np
    img = torch.randn(2, 10, 224, 224)
    rois = np.array([[0.2, 0.2, 0.4, 0.4],
                    [0.5, 0.5, 0.7, 0.7],
                    [0.1, 0.1, 0.3, 0.3]])
    roi_idx = np.array([0, 0, 1])
    r = ROIPooling((7, 7))
    print(r.forward(img, rois, roi_idx).shape)
