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
#
# Copyright (c) 2018, Xiaopeng, Inc. 
# 2018.8.15 Caizw: add functions to export pb format and convert onnx format
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
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

#sys.path.append('/home/nvidia/caffe2_TRT/detectron')
#sys.path.append('/home/nvidia/caffe2_TRT/maskrcnn_c/python')

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--export',
        dest='export',
        help='export to init_net.pb, predict_net.pb',
        action='store_true'
    )
    parser.add_argument(
        '--convert',
        dest='convert',
        help='convert to onnx to model.onnx',
        action='store_true'
    ) 
    parser.add_argument(
        '--selnet',
        dest='selnet',
        help='select net like [roi_net|model_net],704,1344',
        default='model_net,704,1344',
        type=str
    )       
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

import onnx
import maskrcnn_frontend
from caffe2.proto import caffe2_pb2
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew
from caffe2.python.predictor.mobile_exporter import Export
def export(workspace, net, params, init_net_name="init_net.pb", predict_net_name="predict_net.pb" ):
    print("Export " + net.Proto().name)   
    extra_params = []
    extra_blobs = []
    for blob in workspace.Blobs():
        name = str(blob)
        if name.endswith("_rm") or name.endswith("_riv"):
            extra_params.append(name)
            extra_blobs.append(workspace.FetchBlob(name))
    for name, blob in zip(extra_params, extra_blobs):
        workspace.FeedBlob(name, blob)
        params.append(name)
    init_net, predict_net = Export(workspace, net, params)
    
    tmpinput = predict_net.external_input
    newinput = []
    for tmp in tmpinput :
      if tmp not in newinput : newinput.append( tmp )
    for i in range( len(tmpinput) ): tmpinput.remove( tmpinput[0] )
    predict_net.external_input.extend( newinput ) 
    #print(predict_net.external_input) 
 
    # Delete unsupport op
    device_opts = core.DeviceOption(caffe2_pb2.CUDA,0) # CPU / CUDA / CUDNN
    for o in predict_net.op :
        o.engine = ''
        o.device_option.CopyFrom(device_opts);
        for ta in o.arg :
            if ta.name=='exhaustive_search' or ta.name=='grad_input_indices' or  ta.name=='grad_output_indices' :
                o.arg.remove( ta )
    
    for o in predict_net.op:
        if o.type == 'StopGradient' :
            predict_net.op.remove( o )
        if o.type == 'RoIAlign':
            o.name = 'RoIAlign:' + o.input[0] + ',' + o.output[0] 
        if o.type == 'BatchPermutation':
            o.name += 'BatchPermutation:' + o.input[0] + ',' + o.output[0]  
           
    for o in init_net.op :
        o.engine = ''
        o.device_option.CopyFrom(device_opts);
       
    #delete useless data
    allop = list(init_net.op)
    for o in allop:
        if o.output[0] not in predict_net.external_input :
            print('Remove output: ' + o.output[0] )
            init_net.op.remove( o )
    for name in predict_net.external_output:print("Use output: "+name)
           
    with open(init_net_name, 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(predict_net_name, 'wb') as f:
        f.write(predict_net.SerializeToString())
    onnxname = net.Proto().name+'.onnx'
    #ConvPb2ONNX( init_net, predict_net, onnxname )

roi_net_valinfo = {"gpu_0/rpn_bbox_pred_fpn2":(onnx.TensorProto.FLOAT, (1, 12, 192, 336)),
        "gpu_0/rpn_bbox_pred_fpn6":(onnx.TensorProto.FLOAT, (1, 12, 12, 21)),
        "gpu_0/rpn_cls_probs_fpn5":(onnx.TensorProto.FLOAT, (1, 3, 24, 42)),
        "gpu_0/rpn_bbox_pred_fpn3":(onnx.TensorProto.FLOAT, (1, 12, 96, 168)),
        "gpu_0/rpn_cls_probs_fpn4":(onnx.TensorProto.FLOAT, (1, 3, 48, 84)),
        "gpu_0/rpn_bbox_pred_fpn4":(onnx.TensorProto.FLOAT, (1, 12, 48, 84)),
        "gpu_0/rpn_cls_probs_fpn3":(onnx.TensorProto.FLOAT, (1, 3, 96, 168)),
        "gpu_0/rpn_cls_probs_fpn2":(onnx.TensorProto.FLOAT, (1, 3, 192, 336)),
        "gpu_0/rpn_bbox_pred_fpn5":(onnx.TensorProto.FLOAT, (1, 12, 24, 42)),
        "gpu_0/rpn_cls_probs_fpn6":(onnx.TensorProto.FLOAT, (1, 3, 12, 21)),
        "gpu_0/fpn_res2_2_sum":(onnx.TensorProto.FLOAT, (1, 256, 192, 336)),
        "gpu_0/fpn_res3_3_sum":(onnx.TensorProto.FLOAT, (1, 256, 96, 168)),
        "gpu_0/fpn_res4_5_sum":(onnx.TensorProto.FLOAT, (1, 256, 48, 84)),       
        "gpu_0/fpn_res5_2_sum":(onnx.TensorProto.FLOAT, (1, 256, 24, 42)),
        'gpu_0/im_info': (onnx.TensorProto.FLOAT, (1, 3, 1, 1))
        }
roi_net_output = []  #"gpu_0/roi_feat"

mask_net_valinfo = {"gpu_0/fpn_res2_2_sum":(onnx.TensorProto.FLOAT, (1, 256, 192, 336)),
        "gpu_0/fpn_res3_3_sum":(onnx.TensorProto.FLOAT, (1, 256, 96, 168)),
        "gpu_0/fpn_res4_5_sum":(onnx.TensorProto.FLOAT, (1, 256, 48, 84)),       
        "gpu_0/fpn_res5_2_sum":(onnx.TensorProto.FLOAT, (1, 256, 24, 42)),
        'gpu_0/mask_rois_fpn3':(onnx.TensorProto.FLOAT, (6, 5)),
        'gpu_0/mask_rois_fpn2':(onnx.TensorProto.FLOAT, (2, 5)),
        'gpu_0/mask_rois_fpn5':(onnx.TensorProto.FLOAT, (1, 5)),
        'gpu_0/mask_rois_fpn4':(onnx.TensorProto.FLOAT, (1,5)),
        'gpu_0/mask_rois_idx_restore_int32':(onnx.TensorProto.FLOAT, (10,1))
        }
mask_net_output = ["gpu_0/mask_fcn_probs"]

model_net_output=["gpu_0/rois", "gpu_0/bbox_pred", "gpu_0/cls_prob", 
        'gpu_0/fpn_res2_2_sum',  'gpu_0/fpn_res3_3_sum', 'gpu_0/fpn_res4_5_sum', 'gpu_0/fpn_res5_2_sum' ,
        ]
''' 
model_net_output+=["gpu_0/roi_feat","gpu_0/fc6","gpu_0/fc7","gpu_0/cls_score"] 
model_net_output+=["gpu_0/rpn_cls_probs_fpn2","gpu_0/rpn_bbox_pred_fpn2",
                   "gpu_0/rpn_cls_probs_fpn3","gpu_0/rpn_bbox_pred_fpn5",
                   "gpu_0/rpn_cls_probs_fpn4","gpu_0/rpn_bbox_pred_fpn4",
                   "gpu_0/rpn_cls_probs_fpn5","gpu_0/rpn_bbox_pred_fpn5",
                   "gpu_0/rpn_cls_probs_fpn6","gpu_0/rpn_bbox_pred_fpn6"]
'''
                       
#Convert pb to onnx    
def ConvPb2ONNX(init_net, predict_net, onnxname='model.onnx' ):
    print("Convert " + onnxname) 

    # Save the ONNX model   
    data_type = onnx.TensorProto.FLOAT
    data_shape = (1, 3, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE) # 576, 704, 768 ; 768,1344; 640,960
    im_info_shape = (1, 3) # 640, 704, 736, 768
    value_info = {'gpu_0/data': (data_type, data_shape), 'gpu_0/im_info': (data_type, im_info_shape)}
    #value_info =  roi_net_valinfo #mask_net_valinfo
    #onnxname = 'roi_net.onnx'
    
    onnx_model = maskrcnn_frontend.caffe2_net_to_onnx_model(predict_net,init_net,value_info)
    onnx.checker.check_model(onnx_model)
    print("Save " + onnxname) 
    onnx.save(onnx_model, onnxname)

def convertONNX(outputnames=[], init_net_name="init_net.pb", predict_net_name="predict_net.pb" ):
    print("convertONNX ")   
    predict_net = caffe2_pb2.NetDef()
    with open(predict_net_name, 'rb') as f:
        predict_net.ParseFromString(f.read())
        #print(predict_net.external_input)     
    tmpinput = predict_net.external_input
    newinput = []
    for tmp in tmpinput :
      if tmp not in newinput : newinput.append( tmp )
    for i in range( len(tmpinput) ): tmpinput.remove( tmpinput[0] )
    predict_net.external_input.extend( newinput ) 
    print(predict_net.external_input) 
    
    if len(outputnames) > 0  :
        tmpoutput = predict_net.external_output
        for i in range( len(tmpoutput) ): tmpoutput.remove( tmpoutput[0] ) 
        for outname in outputnames:
            if outname not in predict_net.external_output : predict_net.external_output.append( outname )  
    print(predict_net.external_output) # 
    
    init_net = caffe2_pb2.NetDef()
    with open(init_net_name, 'rb') as f:
        init_net.ParseFromString(f.read())
    # Save the ONNX model
    ConvPb2ONNX( init_net, predict_net )

def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel( logging.WARNING )
    merge_cfg_from_file(args.cfg)
    modelsize = args.selnet.split(',')
    cfg.NUM_GPUS = 1
    cfg.TEST.SCALE = int(modelsize[1])  # 256x448, 320x576, 640x960
    cfg.TEST.MAX_SIZE = int(modelsize[2])
    print( cfg.PIXEL_MEANS )
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'
        
    splitedNodes = ['gpu_0/fpn_res2_2_sum',  'gpu_0/fpn_res3_3_sum', 'gpu_0/fpn_res4_5_sum', 'gpu_0/fpn_res5_2_sum', 
                'gpu_0/rpn_cls_probs_fpn2', 'gpu_0/rpn_bbox_pred_fpn2', 'gpu_0/rpn_cls_probs_fpn3', 'gpu_0/rpn_bbox_pred_fpn3', 'gpu_0/rpn_cls_probs_fpn4', 'gpu_0/rpn_bbox_pred_fpn4', 'gpu_0/rpn_cls_probs_fpn5', 'gpu_0/rpn_bbox_pred_fpn5', 'gpu_0/rpn_cls_probs_fpn6', 'gpu_0/rpn_bbox_pred_fpn6']
      
    model = infer_engine.initialize_model_from_cfg(args.weights)
    if args.convert : 
        convertONNX(model_net_output) #mask_net_output #roi_net_output 
        #convertONNX(roi_net_output, "init_roi_net.pb", "predict_roi_net.pb") 
        
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
              
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
                
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    logger.setLevel(level = logging.DEBUG)
    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        res_inputData = workspace.FetchBlob('gpu_0/data')
        print('In:' + 'gpu_0/data' + ' shape:' + str(res_inputData.shape) )
        res_inputData.tofile('gpu_0/'+os.path.basename(im_name)+'.bin')
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
            if args.export: 
                export(workspace, model.net, model.params) 
                with open('gpu_0/masknetweight.bin', 'wb') as wf: 
                    totweightsize = 0
                    for blob in ["gpu_0/_[mask]_fcn1_w","gpu_0/_[mask]_fcn1_b", "gpu_0/_[mask]_fcn2_w","gpu_0/_[mask]_fcn2_b",
                        "gpu_0/_[mask]_fcn3_w","gpu_0/_[mask]_fcn3_b", "gpu_0/_[mask]_fcn4_w","gpu_0/_[mask]_fcn4_b",
                        "gpu_0/conv5_mask_w","gpu_0/conv5_mask_b",  "gpu_0/mask_fcn_logits_w", "gpu_0/mask_fcn_logits_b"] :
                        res_outputData = workspace.FetchBlob( blob )
                        wsize = res_outputData.size;
                        print("Save Weight " + blob + " size=" , wsize)
                        wf.write(res_outputData.tobytes())
                        totweightsize += wsize
                    endstr = "masknet " + str(totweightsize) + " weight"
                    print("Save Weight " + endstr)
                    wf.write( bytes(endstr))
            for blob in res_outputBlob: 
                res_outputData = workspace.FetchBlob( blob )
                print('Out:' + str(blob) + ' shape:' + str(res_outputData.shape) )
                #res_outputData.tofile(str(blob)+'.txt',sep=',')
            '''            
            for blob in ["gpu_0/rois", "gpu_0/bbox_pred","gpu_0/cls_prob","gpu_0/roi_feat","gpu_0/fc6","gpu_0/fc7","gpu_0/cls_score",
            			 "gpu_0/fpn_res4_5_sum", "gpu_0/fpn_res5_2_sum",
                         "gpu_0/_[mask]_fcn1","gpu_0/_[mask]_fcn2",
                         "gpu_0/_[mask]_fcn3","gpu_0/_[mask]_fcn4",
                         "gpu_0/conv5_mask","gpu_0/mask_fcn_logits","gpu_0/mask_fcn_probs"]: 
                res_outputData = workspace.FetchBlob( blob )
                print('SaveOut:' + str(blob) + ' shape:' + str(res_outputData.shape) )
                res_outputData.tofile(str(blob)+'.txt',sep=',')                

            res_inputData = workspace.FetchBlob('gpu_0/_[mask]_roi_feat')
            print('In:' + 'gpu_0/_[mask]_roi_feat' + ' shape:' + str(res_inputData.shape) )
            res_inputData.tofile('gpu_0/_[mask]_roi_feat.bin')    
            '''
            #break
        	
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


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
