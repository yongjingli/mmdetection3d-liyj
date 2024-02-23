## @package onnx
# Module caffe2.python.onnx.frontend

"""Caffe2 Protobuf to ONNX converter

To run this, you will need to have Caffe2 installed as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import collections
import logging
import re

from caffe2.python import core as caffe2_core
from caffe2.proto import caffe2_legacy_pb2
from enum import Enum
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model
import numpy as np

from caffe2.python.onnx.helper import c2_native_run_net
from caffe2.python.onnx.error import Unsupported

import caffe2.python._import_c_extension as C

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Caffe2Frontend(object):
    # This number controls the semantics of the operators we target.  Whenever
    # ONNX makes a BC breaking change to semantics of operators, having this set
    # to an accurate number will prevent our models form exporting.  However,
    # we should strive to keep this up-to-date as much as possible.
    target_opset_version = 8

    _renamed_operators = {
        'SpatialBN': 'BatchNormalization',
        'Conv1D': 'Conv',
        'Conv2D': 'Conv',
        'Conv3D': 'Conv',
        'ConvTranspose1D': 'ConvTranspose',
        'ConvTranspose2D': 'ConvTranspose',
        'ConvTranspose3D': 'ConvTranspose',
        'MaxPool1D': 'MaxPool',
        'MaxPool2D': 'MaxPool',
        'MaxPool3D': 'MaxPool',
        'AveragePool1D': 'AveragePool',
        'AveragePool2D': 'AveragePool',
        'AveragePool3D': 'AveragePool',
        'AffineChannel':  'BatchNormalization', # add by caizw@20180817
        'UpsampleNearest': 'Upsample', # add by caizw@20180820
        'RoIAlign':  'ATen', # add by caizw@20180901
        'BatchPermutation': 'ATen', # add by caizw@20180901        
        'Python': 'ATen', # add by caizw@20180901  
        'GroupNorm': 'ATen', # add by caizw@20181211         
    }

    # caffe2 arguments that are completely removed in onnx
    _blacklist_caffe2_args = {
        'order': {b'NCHW'},
        'cudnn_exhaustive_search': {0, 1},
        'exhaustive_search': {0, 1}, # add by caizw@20180817
        'use_cudnn': {0, 1},
    }

    _global_renamed_args = {
        'kernels': 'kernel_shape',
    }

    _per_op_renamed_args = {
        'Squeeze': {'dims': 'axes'},
        'Transpose': {'axes': 'perm'},
    }

    _special_operators = {}

    # Dummy name generator
    _dummy_name = C.DummyName()

    @classmethod
    def dummy_name(cls):
        return cls._dummy_name.new_dummy_name()

    @classmethod
    def _common_caffe2_arg_to_onnx_attr(cls, op_def, arg):
        # name
        op_type = op_def.type
        if op_type in cls._per_op_renamed_args:
            name = cls._per_op_renamed_args[op_type].get(
                arg.name, arg.name)
        else:
            name = cls._global_renamed_args.get(arg.name, arg.name)

        # value
        if arg.HasField('f'):
            value = arg.f
        elif arg.HasField('i'):
            value = arg.i
        elif arg.HasField('s'):
            value = arg.s
        elif arg.floats:
            value = arg.floats
        elif arg.ints:
            value = arg.ints
        elif arg.strings:
            value = arg.strings
        else:
            raise ValueError('Could not find data field in arg: {}'.format(arg))

        if name in cls._blacklist_caffe2_args:
            assert value in cls._blacklist_caffe2_args[arg.name]
            return None

        return helper.make_attribute(name, value)

    @classmethod
    def caffe2_arg_to_onnx_attr(cls, op_def, arg):
        return cls._common_caffe2_arg_to_onnx_attr(op_def, arg)

    @classmethod
    def _common_caffe2_op_to_onnx_node(cls, op_def, shapes):
        node_def = NodeProto()
        node_def.name = op_def.name

        node_def.op_type = cls._renamed_operators.get(op_def.type, op_def.type)

        node_def.input.extend(op_def.input)
        node_def.output.extend(op_def.output)

        attrs = filter(None, [cls.caffe2_arg_to_onnx_attr(op_def, arg)
                              for arg in op_def.arg])
        node_def.attribute.extend(attrs)

        if op_def.type == 'AffineChannel' : 
            node_def.input.extend([op_def.input[2].replace('_bn_b','_bn_m'), op_def.input[1].replace('_bn_s','_bn_v')])
            node_def.attribute.extend([helper.make_attribute('epsilon', 1e-5)]) 
            #print( "Bf:" + op_def.type + " in1:"+op_def.input[1] + " in2:" + op_def.input[2])
            #print( "Af:" + node_def.op_type + " in3:"+node_def.input[3] + " in4:" + node_def.input[4]) 
            #old arg: node_def.attribute.extend([helper.make_attribute("is_test", 1), helper.make_attribute("consumed_inputs", [0,0,0,1,1])])
        
        if op_def.type == 'UpsampleNearest' :
            print("Append:" + op_def.type + " add attr scales")
            scaleattr = node_def.attribute.pop()
            node_def.attribute.extend([helper.make_attribute('scales', [1.0,1.0,2.0,2.0])] ) 
                  
        return node_def

    @classmethod
    def caffe2_op_to_onnx_node(cls, op_def, shapes):
        if C.support_onnx_export(op_def.type):
            node_strs, tensor_strs = C.export_to_onnx(cls._dummy_name, op_def.SerializeToString(), shapes)
            nodes = []
            for s in node_strs:
                node = NodeProto()
                node.ParseFromString(s)
                nodes.append(node)
            const_tensors = []
            for s in tensor_strs:
                tensor = TensorProto()
                tensor.ParseFromString(s)
                const_tensors.append(tensor)
            return nodes, const_tensors
        elif op_def.type in cls._special_operators:
            translator = getattr(cls, cls._special_operators[op_def.type])
        else:
            translator = cls._common_caffe2_op_to_onnx_node
        nodes = translator(op_def, shapes)
        const_tensors = []
        if isinstance(nodes, tuple):
            nodes, const_tensors = nodes
        if not isinstance(nodes, collections.Iterable):
            nodes = [nodes]
        return nodes, const_tensors

    @staticmethod
    def _all_names_in_net(net):
        if net is None:
            return set()

        names = set()
        names.update(net.external_input)
        names.update(net.external_output)
        for op in net.op:
            names.update(op.input)
            names.update(op.output)
        return names

    @staticmethod
    def _extract_value_info(tensor):
        return make_tensor_value_info(
            name=tensor.name,
            elem_type=tensor.data_type,
            shape=tensor.dims)

    @classmethod
    def caffe2_net_to_onnx_graph(cls,
                                 predict_net,
                                 init_net=None,
                                 value_info=None):
        if value_info is None:
            value_info = {}
        if not isinstance(value_info, dict):
            raise ValueError('Please pass value_info as a '
                             'name -> (type, shape) dictionary')

        cls._filter_fake_init(init_net, value_info)
        cls._ssa_rewrite(predict_net, init_net, value_info)

        if init_net:
            initializer = cls.caffe2_init_net_to_initializer(init_net)
            value_info.update({init.name: (init.data_type, init.dims)
                               for init in initializer})
        else:
            initializer = []

        # Check whether we have got type shape info of all input
        missing = (set(list(predict_net.external_input)) -
                   set(value_info.keys()))
        if missing:
            raise RuntimeError('Could not find value info of inputs: {}'.format(
                ', '.join(missing)))

        inputs = {}
        for name in predict_net.external_input:
            elem_type, shape = value_info[name]
            if name == 'gpu_0/im_info' :
              inputs[name] = [768,1344,1280/1333.0].astype(
                mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type]);
            else:
              inputs[name] = np.random.randn(*shape).astype(
                mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])

        ws, outputs = c2_native_run_net(
            init_net,
            predict_net,
            inputs)

        for name in predict_net.external_output:
            output = outputs[name]
            elem_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output.dtype]
            shape = output.shape
            value_info[name] = (elem_type, shape)

        graph_def = GraphProto()
        graph_def.name = predict_net.name
        graph_def.initializer.extend(initializer)
        # This is a mapping from Caffe2 names to ONNX names
        graph_def.input.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in predict_net.external_input)

        inputappend=[]	#add by caizw@20180822 init for batchnorm( m=0, v=1 )	
        for name in predict_net.external_input :
            if '_bn_b' in name :
                inputappend.append(name.replace('_bn_b','_bn_m'))
            if '_bn_s' in name :
                inputappend.append(name.replace('_bn_s','_bn_v'))
        graph_def.input.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in inputappend)
            
        cls._dummy_name.reset(cls._all_names_in_net(predict_net) | cls._all_names_in_net(init_net))

        for op in predict_net.op:
            shapes = {}
            for name in itertools.chain(op.input, op.output):
                blob = ws.FetchBlob(name)
                if hasattr(blob, 'shape'):
                    shapes[name] = blob.shape
            nodes, const_tensors = cls.caffe2_op_to_onnx_node(op, shapes=shapes)
            graph_def.node.extend(nodes)
            graph_def.initializer.extend(const_tensors)
            graph_def.input.extend([cls._extract_value_info(tensor) for tensor in const_tensors])

        all_output = set(sum((list(node.output) for node in graph_def.node),
                             [init.name for init in graph_def.initializer]))
        redundant_output = set(vi.name for vi in graph_def.output) - all_output
        if redundant_output:
            logger.warning(
                'There are graph output not produced by any node or initializer: {}'
                '! Will drop them.'.format(', '.join(redundant_output)))
        graph_def.output.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in predict_net.external_output
            if name in all_output)

        return graph_def

    @classmethod
    def caffe2_init_net_to_initializer(cls, init_net):
        ws, _ = c2_native_run_net(init_net=None, predict_net=init_net, inputs=[])
        output_names = []
        for op in init_net.op:
            output_names.extend(op.output)
        initializer = [numpy_helper.from_array(ws.FetchBlob(name), name=name)
                       for name in sorted(set(output_names))]
        initappend=[] #add by caizw@20180822 init for batchnorm( m=0, v=1 )
        for tensor in initializer:
            if '_bn_b' in tensor.name :
                initappend.append(numpy_helper.from_array(np.zeros(tensor.dims,dtype = np.float32), name=tensor.name.replace('_bn_b','_bn_m')))
            if '_bn_s' in tensor.name :
                initappend.append(numpy_helper.from_array(np.ones(tensor.dims,dtype = np.float32), name=tensor.name.replace('_bn_s','_bn_v')))
        for tensor in initappend: print( "Append:"+tensor.name + "," + str(tensor.dims) ) 
        initializer.extend( initappend )       
        return initializer

    @classmethod
    def _filter_fake_init(cls, init_net, value_info):
        if init_net:
            fake_inits = [op for op in init_net.op
                          if len(op.output) == 1 and op.output[0] in value_info and
                          re.match('GivenTensor.*Fill|ConstantFill', op.type)]
            for fake_init in fake_inits:
                init_net.op.remove(fake_init)
            del fake_inits[:]
            del fake_inits

    @classmethod
    def ssa_rewrite(cls, net, init_net, value_info):
        return cls._ssa_rewrite(net, init_net, value_info)

    @classmethod
    def _ssa_rewrite(cls, net, init_net, value_info):
        def ssa_name(name, version):
            return '{}_{}'.format(name, version)

        if init_net:
            for op in init_net.op:
                assert re.match('GivenTensor.*Fill', op.type), "type is {}, \n{}".format(op.type, op)
                assert len(op.output) == 1
                op.output[0] = ssa_name(op.output[0], 0)
            init_net.external_input[:] = [ssa_name(name, 0)
                                          for name in init_net.external_input]
            init_net.external_output[:] = [ssa_name(name, 0)
                                           for name in init_net.external_output]
        if value_info:
            ssa_value_info = {ssa_name(name, 0): value
                              for name, value in value_info.items()}
            value_info.clear()
            value_info.update(ssa_value_info)
        net.external_input[:] = [ssa_name(name, 0)
                                 for name in net.external_input]
        ssa, blob_versions = caffe2_core.get_ssa(net)
        assert len(net.op) == len(ssa)
        for op, (versioned_inputs, versioned_outputs) in zip(net.op, ssa):
            op.input[:] = [ssa_name(name, version)
                           for name, version in versioned_inputs]
            op.output[:] = [ssa_name(name, version)
                            for name, version in versioned_outputs]
        net.external_output[:] = [ssa_name(name, blob_versions[name])
                                  for name in net.external_output]

    @classmethod
    def caffe2_net_to_onnx_model(cls, *args, **kwargs):
        opset_id = OperatorSetIdProto()
        opset_id.domain = ''  # ONNX default domain
        opset_id.version = cls.target_opset_version
        model = make_model(cls.caffe2_net_to_onnx_graph(*args, **kwargs),
                           opset_imports=[opset_id],  # current supported opset version
                           producer_name='onnx-caffe2',  # producer name
                           )
        checker.check_model(model)
        return model


caffe2_net_to_onnx_graph = Caffe2Frontend.caffe2_net_to_onnx_graph
caffe2_net_to_onnx_model = Caffe2Frontend.caffe2_net_to_onnx_model
caffe2_init_net_to_initializer = Caffe2Frontend.caffe2_init_net_to_initializer
ssa_rewrite = Caffe2Frontend.ssa_rewrite