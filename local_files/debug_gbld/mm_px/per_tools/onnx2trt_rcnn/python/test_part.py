# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Inference functionality for most Detectron models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import logging
import numpy as np

from caffe2.python import core
from caffe2.python import workspace
import pycocotools.mask as mask_util

from detectron.core.config import cfg
from detectron.utils.timer import Timer
import detectron.core.test_retinanet as test_retinanet
import detectron.modeling.FPN as fpn
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.image as image_utils
import detectron.utils.keypoints as keypoint_utils

logger = logging.getLogger(__name__)

def pre_maskrcnn( inputs, outputs, im_scale ):
    #box_results_with_nms_and_limit
    boxes = inputs[0]
    scores = inputs[1]
    
    num_classes = cfg.MODEL.NUM_CLASSES   #   = 6 
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]  #  = 0.05
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )

        keep = box_utils.nms(dets_j, cfg.TEST.NMS)          # 0.5
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:    #100
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    im_rois = im_results[:, :-1]
    scores = im_results[:, -1]
    if  im_rois.shape[0] == 0 :
      return cls_boxes, im_rois
    
    rois = im_rois.astype(np.float, copy=False) * im_scale
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)  
    rois_blob = np.hstack((levels, rois))
    rois = rois_blob.astype(np.float32, copy=False)
    
    #rois = collect(inputs, self._train)
    cfg_key = 'TEST'
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N 

    lvl_min = 2 #cfg.FPN.ROI_MIN_LEVEL
    lvl_max = 5 #cfg.FPN.ROI_MAX_LEVEL
    #lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max) 
    w = (rois[:, 3] - rois[:, 1] + 1)
    h = (rois[:, 4] - rois[:, 2] + 1)
    areas = w * h 
    s = np.sqrt(areas)
    s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4
    # Eqn.(1) in FPN paper
    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    lvls = np.clip(target_lvls, lvl_min, lvl_max)

    #outputs[0].reshape(rois.shape)
    outputs[0] = rois

    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        #outputs[output_idx + 1].reshape(blob_roi_level.shape)
        #outputs[output_idx + 1].data[...] = blob_roi_level
        outputs[output_idx + 1] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    outputs[-1] = rois_idx_restore.astype(np.int32)
    print("pre_maskrcnn")
    for input in inputs: 
        print( "Input Shape:" + str( input.shape ) )
    outputnames=['gpu_0/mask_rois', 'gpu_0/mask_rois_fpn2', 'gpu_0/mask_rois_fpn3', 'gpu_0/mask_rois_fpn4', 'gpu_0/mask_rois_fpn5', 'gpu_0/mask_rois_idx_restore_int32']
    for i in range(0,6): 
        workspace.FeedBlob(outputnames[i], outputs[i])
        #print( "Output Shape:" + str( outputs[i].shape ) + " " + outputnames[i])
        #print(  outputs[i] )

    return cls_boxes, im_rois

def im_detect_all(model, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    # Handle RetinaNet testing separately for now
    if cfg.RETINANET.RETINANET_ON:
        cls_boxes = test_retinanet.im_detect_bbox(model, im, timers)
        return cls_boxes, None, None

    timers['im_detect_bbox'].tic()
    scores, boxes, im_scale = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, timers, boxes=box_proposals
        )
    timers['im_detect_bbox'].toc()

    timers['im_detect_maskall'].tic()
    testoutputs = [[0],[1],[2],[3],[4],[5]]
    testcls_boxes, testboxes = pre_maskrcnn( [boxes, scores], testoutputs, im_scale)
    if cfg.MODEL.MASK_ON and testboxes.shape[0] > 0:
      workspace.RunNet(model.mask_net.Proto().name)
      testmasks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')
      ).squeeze()
      testmasks = testmasks.reshape([-1, cfg.MODEL.NUM_CLASSES, cfg.MRCNN.RESOLUTION, cfg.MRCNN.RESOLUTION]) #[ -1, 6, 28, 28]
      print( "testboxes=" + str(testboxes.shape))
      print( "testmasks=" + str(testmasks.shape) )
      timers['im_detect_maskall'].toc()
      
      timers['misc_mask'].tic()
      cls_segms = segm_results(
            testcls_boxes, testmasks, testboxes, im.shape[0], im.shape[1]
        )
      timers['misc_mask'].toc()
    else:
        timers['im_detect_maskall'].toc()
        cls_segms = None
    
    #print( cls_boxes, masks )
    return testcls_boxes, cls_segms, None
    
    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        masks = im_detect_mask(model, im_scale, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1]
        )
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    cls_keyps = None
    
    #print( cls_boxes, masks )
    return cls_boxes, cls_segms, cls_keyps


def im_conv_body_only(model, im, target_scale, target_max_size):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale, _im_info = blob_utils.get_image_blob(
        im, target_scale, target_max_size
    )
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale


def im_detect_bbox(model, im, target_scale, target_max_size, timers=None, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs={}
    if hasattr( model, 'feed_data') and len(model.feed_data) >= 3 :
        inputs['im_info'] = model.feed_data[1]
        im_scale = model.feed_data[1][0][2]
        print( "read data from feed_data, im_scale=" + str(im_scale) )
        inputs['data'] = model.feed_data[2]
    else:
        inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)
    	
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if hasattr( model, 'roi_net'):
        print( "FeedBlob " + core.ScopedName('data') + " " + str(inputs['data'].shape) )
        timers['im_detect_bbox_part1_res_net'].tic()
        if hasattr( model, 'run_feed' ) and hasattr( model, 'feed_data'):
            model.run_feed( model.feed_data ) 
        elif hasattr( model, 'trt_net' ) :
        	model.trt_net( inputs['data'] )
        else:
            workspace.FeedBlob(core.ScopedName('data'), inputs['data'])
            workspace.RunNet(model.res_net.Proto().name)
        timers['im_detect_bbox_part1_res_net'].toc()
        
        #timers['part1_conv_body_net'].tic()
        #workspace.FeedBlob(core.ScopedName('data'), inputs['data'])
        #workspace.RunNet(model.conv_body_net.Proto().name)
        #timers['part1_conv_body_net'].toc()
        
        print( "FeedBlob " + core.ScopedName('im_info') + " " + str(inputs['im_info'].shape) )
        timers['im_detect_bbox_part2_roi_net'].tic()
        print( inputs['im_info'] )
        workspace.FeedBlob(core.ScopedName('im_info'), inputs['im_info'])
        workspace.RunNet(model.roi_net.Proto().name)
        timers['im_detect_bbox_part2_roi_net'].toc()
        print("Run roi_net");
    else:
        for k, v in inputs.items():
            workspace.FeedBlob(core.ScopedName(k), v)
            print( "FeedBlob " + core.ScopedName(k) + " " + str(v.shape) )
        workspace.RunNet(model.net.Proto().name)

    print( 'im_scale=', im_scale )
        
    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS
        )
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale


def im_detect_mask(model, im_scale, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')
	#print( boxes, im_scale )
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
        #print("masknet FeedBlob:"+ core.ScopedName(k) + " " + str( v) )
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')
    ).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def box_results_with_nms_and_limit(scores, boxes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms



def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale
