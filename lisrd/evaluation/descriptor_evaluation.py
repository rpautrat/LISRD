import numpy as np
import cv2
from tqdm import tqdm

from ..datasets import get_dataset
from ..datasets.utils.homographies import warp_points
from ..utils.geometry_utils import get_lisrd_desc_dist_numpy


def run_descriptor_evaluation(config):
    models_name = config['models_name']
    H_estimation = {m: [] for m in models_name}
    precision = {m: [] for m in models_name}
    recall = {m: [] for m in models_name}
    mma = {m: [] for m in models_name}

    dataset = get_dataset(config['name'])(config, 'cpu')
    data_loader = dataset.get_data_loader('test')
    for x in tqdm(data_loader):
        H = x['homography'][0].numpy()
        features = x['features']

        # Run the evaluation for each method
        for m in models_name:
            features[m] = {k: v[0].numpy() for k, v in features[m].items()}
            # Match the features with mutual nearest neighbor filtering
            desc_dist = get_desc_dist(features[m])
            nearest0 = nn_matcher(desc_dist)
            mutual_matches = nearest0 != -1
            m_kp0 = features[m]['keypoints0'][mutual_matches]
            m_kp1 = features[m]['keypoints1'][nearest0[mutual_matches]]

            # Compute the descriptor metrics
            # Homography estimation
            H_estimation[m].append(compute_H_estimation(m_kp0, m_kp1, H,
                                   x['img_size'],
                                   config['correctness_threshold']))

            # Precision
            kp_dist0 = np.linalg.norm(
                m_kp0 - warp_points(m_kp1, np.linalg.inv(H)), axis=1)
            kp_dist1 = np.linalg.norm(m_kp1 - warp_points(m_kp0, H), axis=1)
            precisions = []
            for threshold in range(1, config['max_mma_threshold'] + 1):
                precisions.append(compute_precision(kp_dist0, kp_dist1,
                                                    threshold))
            precision[m].append(precisions[config['correctness_threshold']-1])
            mma[m].append(np.array(precisions))

            # Recall
            kp_dist = np.linalg.norm(
                warp_points(features[m]['keypoints0'], H)[:, None]
                - features[m]['keypoints1'][None], axis=2)
            recall[m].append(compute_recall(kp_dist, nearest0,
                                            config['correctness_threshold']))

    for m in models_name:
        H_estimation[m] = np.mean(H_estimation[m])
        precision[m] = np.mean(precision[m])
        recall[m] = np.mean(recall[m])
        mma[m] = np.mean(np.stack(mma[m], axis=1), axis=1)
    return H_estimation, precision, recall, mma


def get_desc_dist(features):
    """ Given two lists of descriptors (and potentially meta descriptors),
        compute the descriptor distance between each pair of feature. """
    if 'meta_descriptors0' in features:
        # Use the LISRD meta_descriptors to weight the descriptor distance
        desc_dist = get_lisrd_desc_dist_numpy(
            features['descriptors0'], features['descriptors1'],
            features['meta_descriptors0'], features['meta_descriptors1'])
    else:
        # Compute a standard descriptor L2 distance
        desc_dist = np.linalg.norm(
            features['descriptors0'][:, None]
            - features['descriptors1'][None], axis=2)
    return desc_dist


def nn_matcher(desc_dist):
    """ Given a matrix of descriptor distances n_points0 x n_points1,
        return a np.array of size n_points0 containing the indices of the
        closest points in img1, and -1 if the nearest neighbor is not mutual.
    """
    nearest0 = np.argmin(desc_dist, axis=1)
    nearest1 = np.argmin(desc_dist, axis=0)
    non_mutual = nearest1[nearest0] != np.arange(len(nearest0))
    nearest0[non_mutual] = -1
    return nearest0


def compute_H_estimation(m_kp0, m_kp1, real_H, img_shape,
                         correctness_thresh=3):
    # Estimate the homography between the matches using RANSAC
    H, _ = cv2.findHomography(m_kp0[:, [1, 0]], m_kp1[:, [1, 0]], cv2.RANSAC)
    if H is None:
        return 0.

    # Compute the reprojection error of the four corners of the image
    corners = np.array([[0, 0, 1],
                        [img_shape[1] - 1, 0, 1],
                        [0, img_shape[0] - 1, 1],
                        [img_shape[1] - 1, img_shape[0] - 1, 1]])
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners / warped_corners[:, 2:]
    re_warped_corners = np.dot(warped_corners, np.transpose(np.linalg.inv(real_H)))
    re_warped_corners = re_warped_corners[:, :2] / re_warped_corners[:, 2:]
    mean_dist = np.mean(np.linalg.norm(re_warped_corners - corners[:, :2], axis=1))
    correctness = float(mean_dist <= correctness_thresh)

    return correctness


def compute_precision(kp_dist0, kp_dist1, correctness_threshold=3):
    """
    Compute the precision for a given threshold, averaged over the two images.
    kp_dist0 is the distance between the matched keypoints in img0 and the
    matched keypoints of img1 warped into img0. And vice-versa for kp_dist1.
    """
    precision = ((kp_dist0 <= correctness_threshold).mean()
                 + (kp_dist1 <= correctness_threshold).mean()) / 2
    return precision


def compute_recall(kp_dist, nearest0, correctness_threshold=3):
    """ Compute the matching recall for a given threshold.
    kp_dist is the distance between all the keypoints of img0
    warped into img1 and all the keypoints of img1. """
    mutual_matches = nearest0 != -1
    # Get the GT closest point
    closest = np.argmin(kp_dist, axis=1)
    correct_gt = np.amin(kp_dist, axis=1) <= correctness_threshold

    corr_closest = nearest0[mutual_matches] == closest[mutual_matches]
    corr_matches = corr_closest * correct_gt[mutual_matches]

    if (np.sum(correct_gt) > 0) and (np.sum(mutual_matches) > 0):
        recall = np.sum(corr_matches) / np.sum(correct_gt)
    else:
        recall = 0.
    return recall
