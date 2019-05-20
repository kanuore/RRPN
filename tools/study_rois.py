# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
import math
import _init_paths
from rotation.rotate_cpu_nms import rotate_cpu_nms
from rotation.generate_anchors import generate_anchors
CLASSES = ('__background__',
           'text')
image_file = './data/datasets/ICDAR03/SceneTrialTrain/rotated/IMG_1261.JPG'
image_file = '/home/lbk/桌面/20.jpg'

def vis_image(boxes,image_file):
    image = cv2.imread(image_file)
    #img = cv2.resize(image,(500,500))   
    for idx in range(len(boxes)):
        cx,cy,h,w,angle = boxes[idx]
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
        # 在做旋转坐标系中，以x_w为正向，'全局逆时针'
        M1 = np.array([[cos_cita, -sin_cita,0], 
                        [sin_cita, cos_cita,0],
                        [0,0,1]])
        M2 = np.array([[1,0,0],
                        [0,1,0],
                        [cx,cy,1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)
        #print(rotated_pts)
        img = image.copy()
        cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (0, 0, 255),3)

        cv2.imshow("rectangle",img)
        cv2.waitKey(0)
    return img
# the net used
# [x_w,y_h,h,w],
def condinate_rotate(all_anchors,show = False):
    image = cv2.imread(image_file)
    # 这里其实是指定 x,y是w,h
    left_top = np.array((- all_anchors[:, 3] / 2, - all_anchors[:, 2] / 2)).T # left top
    left_bottom = np.array([- all_anchors[:, 3] / 2, all_anchors[:, 2] / 2]).T # left bottom
    right_top = np.array([all_anchors[:, 3] / 2, - all_anchors[:, 2] / 2]).T # right top
    right_bottom = np.array([all_anchors[:, 3] / 2, all_anchors[:, 2] / 2]).T # right bottom
    
    theta = all_anchors[:, 4]
    print 
    #positive angle when anti-clockwise rotation

    cos_theta = np.cos(np.pi / 180 * theta) # D
    sin_theta = np.sin(np.pi / 180 * theta) # D

    # [2, 2, n] n is the number of anchors
    #同上，‘全局逆时针’
    rotation_matrix = [cos_theta, -sin_theta, 
                        sin_theta, cos_theta]

    # coodinate rotation
    pt1 = pts_dot(left_top, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T
    pt2 = pts_dot(left_bottom, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T
    pt3 = pts_dot(right_top, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T
    pt4 = pts_dot(right_bottom, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T
    if show:
        img = image.copy()
        cv2.circle(img,(pt1[0,0],pt1[0,1]),3,(0,0,255),-1)
        cv2.circle(img,(pt2[0,0],pt2[0,1]),3,(0,0,255),-1)
        cv2.circle(img,(pt3[0,0],pt3[0,1]),3,(0,0,255),-1)
        cv2.circle(img,(pt4[0,0],pt4[0,1]),3,(0,0,255),-1)
        cv2.imshow("image",img)
    else:
        return pt1,pt2,pt3,pt4 

def pts_dot(pts, rotat_matrix):
    return np.array([ 
                        pts[:, 0] * rotat_matrix[0] + pts[:, 1] * rotat_matrix[2], 
                        pts[:, 0] * rotat_matrix[1] + pts[:, 1] * rotat_matrix[3]
                    ]).T


def show_rois():
    rois = np.load('/home/lbk/ocr/RRPN-master/rois.npy')
    for i in range(rois.shape[0]):
        make_a_rois = rois[i,1:]
        make_a_rois = make_a_rois[np.newaxis,:]
        print('rois:',make_a_rois)
        vis_image(make_a_rois,image_file)
        condinate_rotate(make_a_rois)
        cv2.waitKey(0)


def ind_inside(pt1, pt2, pt3, pt4, img_width, img_height):

    size = len(pt1)
    IMG_PADDING = 0
    padding_w = IMG_PADDING * img_width
    padding_h = IMG_PADDING * img_height
    iw = img_width+padding_w
    ih = img_height+padding_h

    #print type(pt1),pt1.shape
    pt = np.hstack((pt1,pt2,pt3,pt4))
    tmp = (pt[:,0:8:2]>-padding_w) & (pt[:,1:8:2]>-padding_h) & \
            (pt[:,0:8:2]<iw) & (pt[:,1:8:2]<ih)
    #ins = np.where(tmp[:,0]&tmp[:,1]&tmp[:,2]&tmp[:,3])[0].tolist()
    return tmp[:,0]&tmp[:,1]&tmp[:,2]&tmp[:,3]

def test():
    FILTER = True
    rois = np.load('/home/lbk/ocr/RRPN-master/rois.npy')
    print 'rois.shape : ',rois.shape

    scores = np.load('/home/lbk/ocr/RRPN-master/scores.npy')
    print 'scores.shape : ',scores.shape

    make_a_rois = rois[:,1:]
    dets = np.hstack((make_a_rois,scores)).astype(np.float32)

    keep = rotate_cpu_nms(dets, 0.1)
    make_a_rois = make_a_rois[keep, :]
    scores = scores[keep]
    
    print 'make_a_rois ',make_a_rois.shape
    make_a_rois = make_a_rois[np.where(scores > 0.94)[0]]
    #thre_index = np.where(scores > 0.9)[0]
    #_,_,h,w,_ = make_a_rois
    #if h * w > 16*16*9:continue
    #print scores
    print 'make_a_rois ',make_a_rois.shape

    #make_a_rois = make_a_rois[np.newaxis,::]
    pt1, pt2, pt3, pt4 = condinate_rotate(make_a_rois)
    inside_index = ind_inside(pt1, pt2, pt3, pt4, 500, 500)
    print 'inside_index ', inside_index
    if FILTER:
        vis_image(make_a_rois[inside_index],image_file)
        cv2.waitKey(0)
    else:
        vis_image(make_a_rois,image_file)
        cv2.waitKey(0)

def draw_points():
    index = np.load('/home/lbk/ocr/RRPN-master/index.npy')
    print 'index.shape : ',index.shape

    value = np.load('/home/lbk/ocr/RRPN-master/max.npy')
    print 'value.shape : ',value.shape
    height, width = value.shape

    shift_x = np.arange(0, width) * 8
    shift_y = np.arange(0, height) * 8
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    black = np.zeros((500,500),np.uint8)
    angle = [-30.0, 0.0, 30.0, 60.0, 90.0, 120.0]

    for i in range(63):
        for j in range(63):

            x_1 = shift_x[i][j]+4
            y_1 = shift_y[i][j]+4
            point_value = value[i][j] * 4
            point_angle = angle[(index[i][j]+1) % 6]
            print point_angle
            cos_cita = np.cos(np.pi / 180 * point_angle)
            sin_cita = np.sin(np.pi / 180 * point_angle)
            
            x_2 = int(shift_x[i][j]+4+ point_value*cos_cita)
            y_2 = int(shift_y[i][j]+4+ point_value*sin_cita)
            # 当角度过小，y坐标取整使其归，从而造成显示有偏差
            if value[i][j] > 0.7:
                y_2 = int(shift_y[i][j]+4+ point_value*sin_cita)+1


            cv2.line(black,(x_1,y_1),(x_2,y_2),1)
    cv2.imshow('black',255*(1-black))
if __name__ == '__main__':
    test()
    #draw_points()
    #cv2.waitKey(0)
    #anchors = generate_anchors()
    #print(anchors)