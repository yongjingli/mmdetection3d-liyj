#Copyright (c) 2019, Xpeng Motor. All rights reserved.
#remove useless slice and add pad ( 237x457 -> 240x460 )
import sys,getopt
import onnx
from onnx import helper, shape_inference, numpy_helper
from onnx import TensorProto, AttributeProto
import onnx.utils
from onnx import optimizer
import numpy as np

def splitelOnnxModel(model,model_prefix, nameMap=None):
    NodeList = []
    inList = []    
    for i,node in enumerate(model.graph.input) : 
        node.name = model_prefix + node.name
        if nameMap and nameMap.__contains__(node.name) : 
            node.name = nameMap[node.name]
        if i==0 :
            print("input:", node)
            height = int(node.type.tensor_type.shape.dim[2].dim_value)
            width = int(node.type.tensor_type.shape.dim[3].dim_value)
            if (height % 4) != 0 or  (width%4) != 0 :
                nameMap['0'] = '0pad'
                h_pad =  ( 4096 - height)%4
                w_pad = ( 4096 - width)%4 
                curNode = helper.make_node(
                    'Pad', # node name
                    ['0'], # inputs
                    ['0pad'], # outputs
                    mode='constant', # attributes
                    value=0,
                    pads=[0, 0, 0, 0 ,   0, 0, h_pad, w_pad],
                )
                print(curNode)
                NodeList.append( curNode )
            curNode = helper.make_node(
               'Split', # node name
                inputs=['0pad'], 
                outputs=['split0', 'split1'], 
                axis=1 )
            #nameMap['374'] = 'split0'
            #nameMap['378'] = 'split1'
            #print(curNode)
            #NodeList.append( curNode )
            
        inList.append(node)
    #im_info = helper.make_tensor_value_info("gpu_0/im_info_0",TensorProto.FLOAT,(1,3))
    #inList.append(im_info)
        
    for i in range(len(model.graph.node)): #
        curNode = model.graph.node[i];
        if curNode.name == "" :
            curNode.name = model_prefix + str(i)
        for j,input in enumerate( curNode.input ): 
            input_name = model_prefix + input
            if nameMap and nameMap.__contains__(input) : input_name = nameMap[input]
            curNode.input[j] = input_name
        for j,output in enumerate(model.graph.node[i].output):
            output_name = model_prefix + output
            if nameMap and nameMap.__contains__(output) : output_name = nameMap[output]
            curNode.output[j] = output_name

        if  curNode.op_type == "Slice"  :
            #if i < 10 :
            #    print("Remove Slice")
            #    continue
            #print( curNode.attribute )
            print( "Slice" ,curNode.attribute[2].ints[0] , curNode.attribute[1].ints[0] )
            if 0 == curNode.attribute[2].ints[0] and curNode.attribute[1].ints[0] > 1000:
                nameMap[curNode.output[0]] = curNode.input[0] 
                print("Skip Slice");
                continue
            elif curNode.attribute[1].ints[0] > 1000 : curNode.attribute[1].ints[0] = -1
            
        NodeList.append( curNode )
        print(curNode.name, curNode.op_type, curNode.input, curNode.output )
        
    initList = []    
    for node in model.graph.initializer : 
        node.name = model_prefix + node.name
        if nameMap.__contains__(node.name) : 
            node.name = nameMap[node.name]
            #print("init:", node.name)
        initList.append(node)

    outList = []   
    for node in model.graph.output : 
        node.name = model_prefix + node.name
        if nameMap and nameMap.__contains__(node.name) : node.name = nameMap[node.name]
        outList.append(node)
    return NodeList, initList, inList, outList

def createMulitiOnnxModel( NodeList, linkers ):
    NodeList = []
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == "" :
            model.graph.node[i].name = model_pre + str(i)
        NodeNameList.append(model.graph.node[i].name)
        print(model.graph.node[i].name, model.graph.node[i].op_type, model.graph.node[i].input, model.graph.node[i].output )
    return NodeNameList

def reform_onnx(onnxname, nameMap):
    nodeList_comb, initList_comb, inList_comb, outList_comb = [],[],[],[]
    model = onnx.load(onnxname)
    nodeList, initList, inList, outList = splitelOnnxModel( model, "", nameMap)
    nodeList_comb.extend( nodeList )
    initList_comb.extend( initList )
    inList_comb.extend( inList )
    outList_comb.extend( outList )
    
    graph_comb = helper.make_graph(
                nodeList_comb,
                "xp_model_pad",
                inputs=inList_comb,  
                outputs=outList_comb,  
                initializer=initList_comb,  
            )
    model_comb = helper.make_model(graph_comb, producer_name='pytorch-onnx')
    inferred_model = onnx.shape_inference.infer_shapes(model_comb)
    inferred_model = optimizer.optimize( inferred_model )
    print("xp_model_pad onnx build success!")
    onnx.save( inferred_model, "xp_model_pad_split.onnx" )

import argparse
if __name__ == "__main__" :   
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--mapidx",
        type=int,
        default=-1,
        help="index of nameMap list",
    )
    args = parser.parse_args()
    nameMap = {}

    reform_onnx("reformed.onnx", nameMap)

