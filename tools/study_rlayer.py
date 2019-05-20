# coding=utf-8
import _init_paths
from fast_rcnn.config import cfg,cfg_from_file
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
from rotation.r_minibatch import r_get_minibatch, r_get_rotate_minibatch
def vis_image(img, boxes,RLAYER=False):
    if RLAYER: img = np.zeros((500,500,3),dtype=np.uint8)
    cv2.namedWindow("image")    
    for idx in range(len(boxes)):
        cx,cy,h,w,angle = boxes[idx]
        print ("######################",angle)
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
        M1 = np.array([[cos_cita, -sin_cita,0], 
                        [sin_cita, cos_cita,0],
                        [0,0,1]])
        M2 = np.array([[1,0,0],
                        [0,1,0],
                        [cx,cy,1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (0, 0, 255),3)

    cv2.imshow("image",img)

def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    #随机抽取一张图片
    db_inds = self._get_next_minibatch_inds()
    minibatch_db = [self._roidb[i] for i in db_inds]
    return r_get_rotate_minibatch(minibatch_db, self._num_classes) # D

if __name__ == '__main__':
    np.set_printoptions(suppress = True)
    cfg_from_file('experiments/cfgs/faster_rcnn_end2end.yml')
    roidb = get_rroidb('train')
    #for i in range(len(roidb)):
        #print roidb[i]
    #add_rbbox_regression_targets(roidb)
    a_roidb = roidb[1]
    print __file__,'a_roidb[gt]',a_roidb['boxes']
    print a_roidb['gt_classes']
    src = cv2.imread(a_roidb['image'])
    img = src[:]
    #cv2.imshow('src',img)
    #vis_image(img,a_roidb['boxes'])


    #blobs = r_get_minibatch([a_roidb], 2)
    blobs = r_get_rotate_minibatch([a_roidb], 3)
    #  ['gt_boxes', 'data', 'im_info']
    print 'blobs : ',blobs.keys()
    print 'im_info :',blobs['im_info']
    print 'gt_boxes : ',blobs['gt_boxes'][:,:5]
    # (0, 3, 1, 2)  == > ()
    images = blobs['data'].transpose(0,2,3,1)
    img = images[0]
    #cv2.imshow('test',image)

    vis_image(img,blobs['gt_boxes'][:,:5])
    #cv2.imshow('src',src)
    #vis_image(img,a_roidb['boxes'])
    cv2.waitKey(0)
    