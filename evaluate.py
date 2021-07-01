import torchvision
import torch.nn as nn
import torch
import numpy as np


def predict_label(model_clf_, test_loader, device):
    proper_mask_prediction = []
    for i, test_data in enumerate(test_loader):
        images, _, labels = test_data
        images = images.float().to(device)
        labels = labels.float().to(device)
        X_batch, y_batch_clf = images, labels

        sigmoid = nn.Sigmoid()
        y_pred_clf = model_clf_.forward(X_batch).to(device)
        y_pred_clf = torch.round(sigmoid(y_pred_clf))

        proper_mask_prediction += [True if x.item() else False for x in y_pred_clf]

    return np.array(proper_mask_prediction)


def predict_bbox(model_reg_, test_loader, device):
    bboxes_prediction = []
    for i, test_data in enumerate(test_loader):
        images, bboxes, _ = test_data
        images = images.float().to(device)
        bboxes = bboxes.to(device)
        X_batch, y_batch_reg = images, torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy')

        y_pred_reg = model_reg_.forward(X_batch).to(device)
        y_pred_reg = torch.round(torchvision.ops.box_convert(y_pred_reg, 'xyxy', 'xywh'))

        # tmp_1 = [x.detach().numpy() for x in y_pred_reg]
        # tmp_2 = [x.item() for x in y_pred_clf]
        bboxes_prediction += [x.cpu().detach().numpy() for x in y_pred_reg]

    return np.array(bboxes_prediction).T
