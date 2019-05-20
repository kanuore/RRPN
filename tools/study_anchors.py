# coding=utf-8

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
from rotation.rotate_polygon_nms import rotate_gpu_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math

from rotation.data_extractor import get_rroidb, test_rroidb, get_MSRA
# from eval.MSRA_eval import eval as MSRA_eval
from rotation.rt_test import r_im_detect
from rotation.merge_box import merge
import rotation.rt_test_crop as rt_crop
from rotation.generate_anchors import generate_anchors
from rotation.inside_judge import ind_inside, condinate_rotate
from rotation.rbbox_transform import rbbox_transform
from rotation.rbbox import angle_diff
# 注意网络中调用的是加速gpu单独实现的函数
from rotation.rbbox_overlaps import rbbx_overlaps

CLASSES = ('__background__',
           'text')

def vis_image(boxes,img,color):

    for idx in range(len(boxes)):
        cx,cy,h,w,angle = boxes[idx]
        #if h*w < 10000:continue
        print h,w,h*w
        lt = [cx - w/2, cy - h/2,1]
        rt = [cx + w/2, cy - h/2,1]
        lb = [cx - w/2, cy + h/2,1]
        rb = [cx + w/2, cy + h/2,1]
        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)
        if angle != 0:
            cos_cita = np.cos(np.pi / 180 * angle)
            sin_cita = np.sin(np.pi / 180 * angle)
        else:
            cos_cita = 1
            sin_cita = 0

        M0 = np.array([[1,0,0],
                        [0,1,0],
                        [-cx,-cy,1]])
        M1 = np.array([[cos_cita, sin_cita,0], 
                        [-sin_cita, cos_cita,0],
                        [0,0,1]])
        M2 = np.array([[1,0,0],
                        [0,1,0],
                        [cx,cy,1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)
        #print(rotated_pts)
        cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), color,3)
        cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), color,3)
        cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), color,3)
        cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), color,3)
        
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
    return img


if __name__ == "__main__":
    SHOW_ROI = True
    SHOW_ANCHORS = False
    anchor_scales = (2, 4, 8)
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    _feat_stride = 8

    bbox_para_num = 5
    _allowed_border = 0
    height, width = (63,63)
    A = _num_anchors
    # labels
    top0 = np.zeros((1, 1, A * height, width))
    # bbox_targets
    top1 = np.zeros((1, A * bbox_para_num, height, width))
    # bbox_inside_weights
    top2 = np.zeros((1, A * bbox_para_num, height, width))
    # bbox_outside_weights
    top3 = np.zeros((1, A * bbox_para_num, height, width))

    # im_info
    im_info = [500,500]
    #forward
    # list

    roidb = get_rroidb("train")[0]
    gt_boxes = np.asarray(roidb['boxes'])
    # 对gt过滤，因为gt是原坐标生成，旋转后可能超出界面
    gt1, gt2, gt3, gt4 = condinate_rotate(gt_boxes[:,0:5]) # coodinate project
    inds_inside = np.array(ind_inside(gt1, gt2, gt3, gt4, im_info[0], im_info[1])) # inside index
    gt_boxes = gt_boxes[inds_inside.astype(np.int32)]
    # 结束
    n_batch = np.ones(len(gt_boxes))
    gt_boxes = np.column_stack((gt_boxes,n_batch))
    # GT boxes (x_ctr, y_ctr, height, width, theta, label)
    # gt_boxes = np.array([[250.0,250.0,256.0,256.0,16,1.0]])#,[300.0,300.0,256.0,256.0,0.0,1.0]])



    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    #print shift_x.ravel().shape,shift_y.ravel().shape, np.zeros((3,width*height)).shape
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),np.zeros((3, width*height)))).transpose()
    #print shifts
    
    # add A anchors (1, A, 5) to
    # cell K shifts (K, 1, 5) to get
    # shift anchors (K, A, 5)
    # reshape to (K*A, 5) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, bbox_para_num)) +
                   shifts.reshape((1, K, bbox_para_num)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, bbox_para_num))
    total_anchors = int(K * A)
    
    # 32*32*3*3*6 =(55296, 5)
    print 'before : ',all_anchors.shape

    pt1, pt2, pt3, pt4 = condinate_rotate(all_anchors) # coodinate project
    inds_inside = np.array(ind_inside(pt1, pt2, pt3, pt4, im_info[0], im_info[1])) # inside index
    anchors = all_anchors[inds_inside, :]

    # after :  (8148, 5)
    print 'after : ',anchors.shape

    if SHOW_ANCHORS:
        for i in anchors:
            black = np.ones((500,500,3),dtype = np.uint8) * 255
            black = vis_image([i],black,color =(255,0,0))
            cv2.imshow('black',black)
            k = cv2.waitKey(0) & 0xff
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    if SHOW_ROI:

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)
        
        #overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32), np.ascontiguousarray(gt_boxes[:,0:5], dtype=np.float32))
        overlaps = rbbx_overlaps(anchors.astype(dtype=np.float32), gt_boxes[:,0:5].astype(dtype=np.float32))
        an_gt_diffs = angle_diff(anchors,gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1) # max overlaps of anchor compared with gts
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        
        gt_argmax_overlaps = overlaps.argmax(axis=0) # max overlaps of gt compared with anchors
        gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
        
        max_overlaps_angle_diff = an_gt_diffs[np.arange(len(inds_inside)), argmax_overlaps]                           
        gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps)&(an_gt_diffs<=cfg.TRAIN.R_POSITIVE_ANGLE_FILTER))[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU the angle diff abs must be less than 15
        labels[(max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP) & (max_overlaps_angle_diff <= 15)] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[(max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) | (max_overlaps_angle_diff > 15)] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1


        #print 'label = 0 : ',np.sum(labels==0)
        #print 'label = 1 : ',np.sum(labels==1)

        fg_rois = anchors[np.where(labels==1)]
        overlaps = overlaps[np.where(labels == 1)]
        #fg_rois = anchors
        image_file = roidb['image']
        gts = roidb['boxes']
        img = cv2.imread(image_file)
        img = vis_image(gts,img,color =(255,0,0))
        #cv2.imshow('boxes',img)
        #cv2.waitKey(0)
        
        for i in range(len(fg_rois)):
            print overlaps[i]
            image = img.copy()
            make_a_rois = fg_rois[i]
            make_a_rois = make_a_rois[np.newaxis,:]
            print make_a_rois
            image = vis_image(make_a_rois,image,color=(0,0,255))
            cv2.imshow('boxes',image)
            cv2.waitKey(0)
        