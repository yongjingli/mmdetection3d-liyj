#!/usr/bin/env python2

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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
#import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()
#sys.path.append('/home/nvidia/caffe2_TRT/detectron')

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import detectron.utils.net as net_utils
from detectron.modeling import model_builder
from test_part import im_detect_all

def initialize_model_from_cfg(weights_file, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils.initialize_gpu_from_weights_file(
        model, weights_file, gpu_id=gpu_id,
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    net_utils.print_net( model )
    return model
    
import ctypes
import numpy as np
import numpy.ctypeslib as npct

libonnxtrt = None
trtOutData = None
createEngine = None
runEngine = None
def runtensorrt( inputData ):
    global libonnxtrt,trtOutData,createEngine,runEngine, args
    if libonnxtrt is None:
        libonnxtrt = ctypes.CDLL("libonnxtrt.so")
        createEngine = libonnxtrt.CreateEngine
        createEngine.restype = ctypes.c_int
        createEngine(args.trtmodel) #res_net_fp16.trt
        #createEngine(b"/home/nvidia/res_net_dbg.trt")
        # must be a double array, with single dimension that is contiguous
        array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
        runEngine = libonnxtrt.RunEngine
        runEngine.restype = ctypes.c_int
        runEngine.argtypes = [ctypes.c_int, array_1d_float, array_1d_float]
        trtOutData = np.ones( [92892240], dtype = np.float32 )

    trtoutput = [ ["gpu_0/rpn_bbox_pred_fpn2",[1, 12, 192, 336], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn6", [1, 12, 12, 21], 4 ],
        ["gpu_0/rpn_cls_probs_fpn5", [1, 3, 24, 42], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn3", [1, 12, 96, 168], 4 ],
        ["gpu_0/rpn_cls_probs_fpn4", [1, 3, 48, 84], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn4", [1, 12, 48, 84], 4 ],
        ["gpu_0/rpn_cls_probs_fpn3", [1, 3, 96, 168], 4 ],
        ["gpu_0/rpn_cls_probs_fpn2", [1, 3, 192, 336], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn5", [1, 12, 24, 42], 4 ],
        ["gpu_0/rpn_cls_probs_fpn6", [1, 3, 12, 21], 4 ],
        ["gpu_0/fpn_res2_2_sum",  [1, 256, 192, 336], 4],
        ["gpu_0/fpn_res3_3_sum", [1, 256, 96, 168], 4 ],
        ["gpu_0/fpn_res4_5_sum", [1, 256, 48, 84], 4 ],        
        ["gpu_0/fpn_res5_2_sum", [1, 256, 24, 42], 4 ],
        ]

    result = runEngine(1, inputData, trtOutData)
    outdict={}
    offset=0;
    t = time.time()
    for outinfo in trtoutput:
        blocksize = outinfo[1][0]*outinfo[1][1]*outinfo[1][2]*outinfo[1][3]
        blockdata = trtOutData[offset:offset+blocksize]
        outdict[ outinfo[0] ] = np.reshape( blockdata, outinfo[1] )
        offset = offset+blocksize
        #print( outinfo[0] + " out: " + str(blockdata [0:16]) )
        #print( 'FeedBlob:'+ outinfo[0] + ' data:' + str(outdict[ outinfo[0] ].shape) )
        workspace.FeedBlob( outinfo[0], outdict[ outinfo[0] ])
    print('runEngine FeedBlob: {:.3f}s'.format(time.time() - t))
    return outdict
    
def runtrtnet( inputdata ):
    inputdata = inputdata.reshape(-1)
    print( 'shape:'+ str(inputdata.shape) + ' type:' + str(type(inputdata[0])) )
    outdict = runtensorrt( inputdata )
    return

def runfeeddata( inputData ): # [im, data, im_info, output14]
    trtoutput = [ ["gpu_0/rpn_bbox_pred_fpn2",[1, 12, 192, 336], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn6", [1, 12, 12, 21], 4 ],
        ["gpu_0/rpn_cls_probs_fpn5", [1, 3, 24, 42], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn3", [1, 12, 96, 168], 4 ],
        ["gpu_0/rpn_cls_probs_fpn4", [1, 3, 48, 84], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn4", [1, 12, 48, 84], 4 ],
        ["gpu_0/rpn_cls_probs_fpn3", [1, 3, 96, 168], 4 ],
        ["gpu_0/rpn_cls_probs_fpn2", [1, 3, 192, 336], 4 ],
        ["gpu_0/rpn_bbox_pred_fpn5", [1, 12, 24, 42], 4 ],
        ["gpu_0/rpn_cls_probs_fpn6", [1, 3, 12, 21], 4 ],
        ["gpu_0/fpn_res2_2_sum",  [1, 256, 192, 336], 4],
        ["gpu_0/fpn_res3_3_sum", [1, 256, 96, 168], 4 ],
        ["gpu_0/fpn_res4_5_sum", [1, 256, 48, 84], 4 ],        
        ["gpu_0/fpn_res5_2_sum", [1, 256, 24, 42], 4 ],
        ]
    index=3;
    for outinfo in trtoutput:
        print( 'FeedBlob:'+ outinfo[0] + ' data:' + str( outinfo[1]) )
        workspace.FeedBlob( outinfo[0], inputData[ index ])
        index = index+1
    return 
    
g_model = None
def processimg( arrays ):
    global g_model
    print( 'feed_data len=' + str(len(arrays)) )
    g_model.feed_data = arrays
    im = arrays[0]
    print( 'im shape:'+ str(im.shape) + ' type:' + str(type(im[0][0][0])) )    
    if len( arrays ) == 2 or  len( arrays ) == 3:  # [im, im_info, data] , [im, im_info] resize im to data by python
        g_model.trt_net = runtrtnet
    if len( arrays ) == 17 : # [im, data, im_info, output14]
        g_model.run_feed = runfeeddata
              
    timers = defaultdict(Timer)
    t = time.time()                    
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = im_detect_all(
             g_model, im, None, timers=timers
        )
    print('Inference time: {:.3f}s'.format(time.time() - t))         
    for k, v in timers.items():
        print(' | {}: {:.3f}s'.format(k, v.average_time))
                
    return cls_boxes, cls_segms, cls_keyps
            
def main_init(args):
    global g_model, g_timers
    logger = logging.getLogger(__name__)
    logger.setLevel( logging.WARNING )
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'
        
    splitedNodes = ['gpu_0/fpn_res2_2_sum',  'gpu_0/fpn_res3_3_sum', 'gpu_0/fpn_res4_5_sum', 'gpu_0/fpn_res5_2_sum', 
                'gpu_0/rpn_cls_probs_fpn2', 'gpu_0/rpn_bbox_pred_fpn2', 'gpu_0/rpn_cls_probs_fpn3', 'gpu_0/rpn_bbox_pred_fpn3', 'gpu_0/rpn_cls_probs_fpn4', 'gpu_0/rpn_bbox_pred_fpn4', 'gpu_0/rpn_cls_probs_fpn5', 'gpu_0/rpn_bbox_pred_fpn5', 'gpu_0/rpn_cls_probs_fpn6', 'gpu_0/rpn_bbox_pred_fpn6']
      
    #if args.convert : convertONNX(outputnames=splitedNodes)
    model = initialize_model_from_cfg(args.weights)
              
    #roi_input = ['gpu_0/im_info', 'gpu_0/fpn_inner_res2_2_sum_lateral',  'gpu_0/fpn_inner_res3_3_sum_lateral', 'gpu_0/fpn_inner_res4_5_sum_lateral', 'gpu_0/fpn_res5_2_sum' ]
    roi_input = splitedNodes + ['gpu_0/im_info']
    roi_output = ['gpu_0/cls_prob', 'gpu_0/bbox_pred']
    roi_net, roi_outputBlob = model.net.ClonePartial('',roi_input, roi_output)
    #print( roi_net.Proto() )
    model.roi_net = roi_net.Clone('roi_net')
    print( model.roi_net.Proto().name )
    workspace.CreateNet(model.roi_net)

    res_net_input = ['gpu_0/data']
    #res_net_output = ['gpu_0/fpn_inner_res2_2_sum_lateral',  'gpu_0/fpn_inner_res3_3_sum_lateral', 'gpu_0/fpn_inner_res4_5_sum_lateral', 'gpu_0/fpn_res5_2_sum' ]
    res_net_output = splitedNodes
    #res_net_output += ["gpu_0/conv1", "gpu_0/pool1"]
    res_net, res_outputBlob = model.net.ClonePartial('',res_net_input, res_net_output)
    #print( res_net.Proto() )
    model.res_net = res_net.Clone('res_net')
    print( model.res_net.Proto().name )
    workspace.CreateNet(model.res_net)
    #if args.tensorrt : model.trt_net = runtrtnet
    g_model = model
	        
def main_process(args):  
    global g_model, g_timers
    logger = logging.getLogger(__name__)
    logger.setLevel( logging.DEBUG )
      
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()                          
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        if args.tensorrt :
            cls_boxes, cls_segms, cls_keyps = processimg( [im, [[0,0,0]]] )
        else :
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = im_detect_all(
                g_model, im, None, timers=timers
                )
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))        
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
            #if args.export: export(workspace, model.res_net, model.params) 
            '''
            res_inputData = workspace.FetchBlob('gpu_0/data')
            res_inputData.tofile('gpu_0/data.bin')
            for blob in res_outputBlob: 
                res_outputData = workspace.FetchBlob( blob )
                print('Out:' + str(blob) + ' shape:' + str(res_outputData.shape) )
                res_outputData.tofile(str(blob)+'.txt',sep=',')
            '''    
        	
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            ext=args.output_ext,
            out_when_no_box=args.out_when_no_box
        )

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--tensorrt',
        dest='tensorrt',
        help='run with tensorrt',
        action='store_true'
    )    
    return parser.parse_args()
   
  
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)
args = parse_args()
args.output_dir = '/tmp/detectron'
args.output_ext = 'pdf'
args.image_ext = 'jpg'
args.cfg = '/home/nvidia/caffe2_TRT/model/e2e_mask_rcnn_R-50-FPN_2x.yaml'
args.weights = '/home/nvidia/caffe2_TRT/model/model_iter179999.pkl'
args.im_or_folder = '/home/nvidia/caffe2_TRT/test_pic/test'
#args.tensorrt = True
args.trtmodel=b"res_net_fp16.trt"
args.out_when_no_box = False

main_init(args)
         
if __name__ == '__main__':
    main_process(args)
