#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, sys, cv2
import glob
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def get_detections(net, im, conf_thresh=0.8, nms_thresh=0.3):
    scores, boxes = im_detect(net, im)
    
    final_dets = {}
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        
        inds = np.where(dets[:, -1] >= conf_thresh)[0]
        if len(inds) > 0:
            final_dets[cls] = dets[inds]
            
    return final_dets

def draw_bbox(ax, img, dets_dict, title):
    ax.imshow(img[:, :, (2, 1, 0)], aspect='equal')
    ax.set_title(title, fontsize=14)
    
    for cls_name, dets in dets_dict.items():
        for i in range(len(dets)):
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=2)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.2f}'.format(cls_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')
    ax.axis('off')

def parse_args():
    parser = argparse.ArgumentParser(description='Compare two models')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int)
    parser.add_argument('--proto', dest='prototxt', required=True, help='Path to test.prototxt')
    parser.add_argument('--base', dest='base_model', required=True, help='Original .caffemodel')
    parser.add_argument('--new', dest='new_model', required=True, help='Finetuned .caffemodel')
    parser.add_argument('--num', dest='num_imgs', default=10, type=int, help='Number of images to visualize')
    parser.add_argument('--data', dest='img_dir', required=True, help='Path to input images directory')
    parser.add_argument('--out', dest='out_dir', default='comparison_results', help='Path to save output results')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cfg.TEST.HAS_RPN = True
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print("Loading Baseline Model: {}".format(args.base_model))
    net_base = caffe.Net(args.prototxt, args.base_model, caffe.TEST)

    print("Loading Finetuned Model: {}".format(args.new_model))
    net_new = caffe.Net(args.prototxt, args.new_model, caffe.TEST)

    print("Reading images from: {}".format(args.img_dir))
    img_list = glob.glob(os.path.join(args.img_dir, '*.jpg'))
    
    if len(img_list) == 0:
        print("Error: No .jpg images found in {}".format(args.img_dir))
        sys.exit(1)

    np.random.shuffle(img_list) 
    process_list = img_list[:args.num_imgs]

    print("Starting comparison on {} images...".format(len(process_list)))

    for i, img_path in enumerate(process_list):
        print("[{}/{}] Processing {}".format(i+1, len(process_list), os.path.basename(img_path)))
        im = cv2.imread(img_path)

        dets_base = get_detections(net_base, im)
        dets_new = get_detections(net_new, im)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        draw_bbox(axes[0], im, dets_base, "Original Model")
        draw_bbox(axes[1], im, dets_new, "Finetuned Model")
        
        plt.tight_layout()
        
        save_path = os.path.join(args.out_dir, 'compare_{}'.format(os.path.basename(img_path)))
        plt.savefig(save_path)
        plt.close()
        
    print("Comparison done! Check results in '{}' folder.".format(args.out_dir))
