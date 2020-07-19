import os
import numpy as np
import cv2
import torch

from ..third_party.super_point_magic_leap.demo_superpoint import SuperPointFrontend


def SIFT_detect(img, nfeatures=1500, contrastThreshold=0.04):
    """ Compute SIFT feature points. """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures,
                                       contrastThreshold=contrastThreshold)
    keypoints = sift.detect(img, None)
    keypoints = [[k.pt[1], k.pt[0], k.response] for k in keypoints]
    keypoints = np.array(keypoints)
    return keypoints


def SP_detect(img, kp_net):
    """ Compute SuperPoint feature points. """
    keypoints, _, _ = kp_net.run(img.astype(np.float32) / 255.)
    keypoints = keypoints.squeeze()[[1, 0, 2], :].transpose()
    return keypoints


def load_SP_net(conf_thresh=0.015, cuda=torch.cuda.is_available(),
                nms_dist=4, nn_thresh=0.7):
    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../third_party/super_point_magic_leap/superpoint_v1.pth')
    kp_net = SuperPointFrontend(
        weights_path, nms_dist=nms_dist, conf_thresh=conf_thresh,
        nn_thresh=nn_thresh, cuda=cuda)
    return kp_net