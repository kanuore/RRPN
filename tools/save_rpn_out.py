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
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math
from rotation.data_extractor import get_rroidb, test_rroidb, get_MSRA
reload(cv2)
# from eval.MSRA_eval import eval as MSRA_eval
from rotation.rt_test import r_im_detect
from rotation.merge_box import merge
import rotation.rt_test_crop as rt_crop
from rotation.data_pick import vis_image

CLASSES = ('__background__',
           'text')
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [rrpn]',
                        choices=['rrpn', 'vgg16', 'zf'], default='rrpn')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    ROIS_SHOW = False
    RPN_SAVE = True
    NETS = {
    'rrpn': ('VGG16',
                  'VGG16_faster_rcnn.caffemodel'),
    'vgg16': ('VGG16',
                  'VGG16_faster_rcnn.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn.caffemodel')}
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    if args.demo_net == "rrpn":
        prototxt = os.path.join(cfg.RRPN_MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
        prototxt = '/home/lbk/ocr/RRPN-master/models/rrpn/VGG16/faster_rcnn_end2end/study_test_line2.prototxt'
    print "prototxt",prototxt
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              'vgg16_faster_rcnn_rpn.caffemodel')

    caffemodel = '/home/lbk/ocr/RRPN-master/vgg16_faster_lines_2.caffemodel'
    print "caffemodel",caffemodel
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # 未加入输入数据
    print ("net blobs : *****************************************")
    #查看网络中的blob形状
    for layer_name, blob in net.blobs.iteritems(): 
        print layer_name + '\t' + str(blob.data.shape)
    
    print ("net params : ****************************************")
    #查看网络中的参数情况
    for layer_name, param in net.params.iteritems(): 
        print layer_name + '\t' + str(param[0].data.shape)

    if RPN_SAVE:
        #roidb = get_rroidb("train")[0]
        #image_file = roidb['image']
        image_file = '/home/lbk/ocr/RRPN-master/data/demo/test_line_data/20.jpg'
        im_orig = cv2.imread(image_file)
        im_orig = cv2.resize(im_orig,(500,500))
        im = im_orig.astype(np.float32, copy=True)
        im -= cfg.PIXEL_MEANS
        im = im.transpose(2,0,1)
        im = im[np.newaxis,:,:,:]
        # shape for input (data blob is N x C x H x W), set data
        #net.blobs['data'].reshape(1, *image.shape)
        
        net.blobs['data'].data[...] = im
        
        im_info = np.array([
                            [500, 500, 1]
                            ],dtype=np.float32)

        net.blobs['im_info'].data[...] = im_info

        net.forward()
        out_rois = net.blobs['rois'].data[...]
        out_scores = net.blobs['scores'].data[...]

        out_prob = net.blobs['rpn_cls_prob_reshape'].data[:, 54:, :, :][0]
        out_prob = out_prob.transpose(1,2,0)
        out_max_index = np.argmax(out_prob,axis=2)
        out_max_value = np.max(out_prob,axis=2)
        
        np.save('rois_2.npy',out_rois)
        np.save('scores_2.npy',out_scores)
        np.save('index_2.npy',out_max_index)
        np.save('max_2.npy',out_max_value)
        print(image_file)
        #print('rois ',out_rois.shape)
        #print('scores ',out_scores.shape)
        #print('prob ',out_prob.shape)
        #print('out_max ',out_max_value.shape)
    if ROIS_SHOW:
      
        gt_boxes = roidb['boxes']

        n_batch = np.ones(11)
        gt_boxes = np.column_stack((n_batch,gt_boxes))
        print(gt_boxes.shape)
        print(type(out))
        print(out.keys())
        print(out['rpn_labels'].shape)
        net.blobs['gt_boxes'].data[...] = gt_boxes
        print('labels : ******************************************')
        print(out['labels'].shape)