# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utils for Metrics
Source for Pose AUC Metrics: VGGT
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid


def thresh_inliers(gt, pred, thresh=1.03, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth dense map of size H x W x C.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth. Default: 1.03
        mask: Array of shape HxW with boolean values to indicate validity. For bool, False means invalid. Default: None
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]). Default: 1

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    # Compute the norms
    gt_norm = np.linalg.norm(gt, axis=-1)
    pred_norm = np.linalg.norm(pred, axis=-1)

    gt_norm_valid = (gt_norm) > 0
    if mask is not None:
        combined_mask = mask & gt_norm_valid
    else:
        combined_mask = gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_1 = np.nan_to_num(
            gt_norm / pred_norm, nan=thresh + 1, posinf=thresh + 1, neginf=thresh + 1
        )  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(
            pred_norm / gt_norm, nan=0, posinf=0, neginf=0
        )  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(
        np.float32
    )  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, combined_mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth dense map of size HxWxC.

    Args:
        gt: Ground truth map as numpy array of shape H x W x C.
        pred: Predicted map as numpy array of shape H x W x C.
        mask: Array of shape HxW with boolean values to indicate validity. For bool, False means invalid. Default: None
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]). Default: 1

    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    error_norm = np.linalg.norm(pred - gt, axis=-1)
    gt_norm = np.linalg.norm(gt, axis=-1)

    gt_norm_valid = (gt_norm) > 0
    if mask is not None:
        combined_mask = mask & gt_norm_valid
    else:
        combined_mask = gt_norm_valid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_ae = np.nan_to_num(error_norm / gt_norm, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, combined_mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae

