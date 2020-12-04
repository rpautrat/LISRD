"""
Export features detections and descriptions for all images in a given folder.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as func
import cv2
from tqdm import tqdm

from .models import get_model
from .models.base_model import Mode
from .datasets.utils.data_reader import resize_and_crop
from .utils.pytorch_utils import keypoints_to_grid
from .models.keypoint_detectors import SIFT_detect, SP_detect, load_SP_net


base_config = {
    'learning_rate': 0.001, 'desc_size': 128,
    'tile': 3, 'n_clusters': 8, 'meta_desc_dim': 128,
    'compute_meta_desc': True, 'freeze_local_desc': False}


def export(images_list, model, checkpoint, keypoints_type,
           num_keypoints, detection_thresh, extension,
           resize=False, h=480, w=640):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = base_config
    config['name'] = model

    # Load the model
    net = get_model(config['name'])(None, config, device)
    net.load(checkpoint, Mode.EXPORT)
    net._net.eval()

    # Load the keypoint network if necessary
    if keypoints_type == 'superpoint':
        kp_net = load_SP_net(conf_thresh=detection_thresh)

    # Parse the data, predict the features, and export them in an npz file
    with open(images_list, 'r') as f:
        image_files = f.readlines()
    image_files = [path.strip('\n') for path in image_files]

    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = resize_and_crop(img, (h, w))
        img_size = img.shape
        if img_size[2] != 3:
            sys.exit('Export only available for RGB images.')
        cpu_gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = torch.tensor(img, dtype=torch.float, device=device)
        img = img.permute(2, 0, 1).unsqueeze(0) / 255.

        # Predict keypoints
        if keypoints_type == 'sift':
            cpu_gray_img = np.uint8(cpu_gray_img)
            keypoints = SIFT_detect(cpu_gray_img, nfeatures=num_keypoints,
                                    contrastThreshold=0.04)
        if keypoints_type == 'superpoint':
            keypoints = SP_detect(cpu_gray_img, kp_net)
        scores = keypoints[:, 2]

        grid_points = keypoints_to_grid(
            torch.tensor(keypoints[:, :2], dtype=torch.float, device=device),
            img_size[:2])
        keypoints = keypoints[:, [1, 0]]

        # Predict the corresponding descriptors
        inputs = {'image0': img}
        with torch.no_grad():
            outputs = net._forward(inputs, Mode.EXPORT, config)
        
        descs = outputs['descriptors']
        meta_descs = outputs['meta_descriptors']
        descriptors, meta_descriptors = [], []
        for k in descs.keys():
            desc = func.normalize(
                func.grid_sample(descs[k], grid_points),
                dim=1).squeeze().cpu().numpy().transpose(1, 0)
            descriptors.append(desc)
            meta_descriptors.append(
                meta_descs[k].squeeze().cpu().numpy())
        descriptors = np.stack(descriptors, axis=1)
        meta_descriptors = np.stack(meta_descriptors, axis=0)

        # Keep the best scores
        idxs = scores.argsort()[-num_keypoints:]

        with open(img_path + extension, 'wb') as output_file:
            np.savez(output_file, keypoints=keypoints[idxs],
                     descriptors=descriptors[idxs], scores=scores[idxs],
                     meta_descriptors=meta_descriptors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_list', type=str,
                         help='Path to a txt file containing the image paths.')
    parser.add_argument('model', type=str,
                         help="Model name.")
    parser.add_argument('--checkpoint', type=str, default=None,
                         help="Path to the model checkpoint.")
    parser.add_argument('--keypoints', type=str, default='sift',
                         help="Type of keypoints to use: "
                         + "'sift', or 'superpoint'.")
    parser.add_argument('--num_kp', type=int, default=2000,
                         help="Number of keypoints to use.")
    parser.add_argument('--detection_thresh', type=float, default=0.015,
                         help="Detection threshold for SuperPoint.")
    parser.add_argument('--resize', action='store_true', default=False,
                        help='Resize the images to a given dimension.')
    parser.add_argument('--h', type=int, default='480',
                        help='Image height.')
    parser.add_argument('--w', type=int, default='640',
                        help='Image width.')
    parser.add_argument('--extension', type=str, default=None,
                         help="Extension to add to each exported npz.")
    args = parser.parse_args()

    model = args.model
    checkpoint = os.path.expanduser(args.checkpoint)
    if not os.path.exists(checkpoint):
        sys.exit('Unable to find checkpoint', checkpoint)

    keypoints_type = args.keypoints
    if keypoints_type not in ['sift', 'superpoint']:
        sys.exit('Unknown keypoint method:', keypoints_type)
    num_keypoints = args.num_kp

    extension = args.extension if args.extension else model
    extension = '.' + extension

    export(args.images_list, model, checkpoint, keypoints_type,
           num_keypoints, args.detection_thresh, extension,
           args.resize, args.h, args.w)
