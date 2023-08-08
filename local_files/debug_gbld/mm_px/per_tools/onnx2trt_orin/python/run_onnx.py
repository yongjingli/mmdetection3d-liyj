
import argparse
import os, sys
import time

try:
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
except ImportError:
    print ('Please install onnxruntime with \'pip install onnxruntime\' first !')
try:
    import numpy as np
except ImportError:
    print ('Please install numpy with \'pip install numpy\' first !')
try:
    from PIL import Image
except ImportError:
    print ('Please install pillow with \'pip install pillow\' first !')

def file_check(f):
    if not os.path.isfile(f):
        print ('File %s is not exist !' % (f))
        sys.exit('')
    if not os.access(f, os.R_OK):
        print ('File %s can not be read !' % (f))
        sys.exit('')

def print_binding(binding_idx, bds, out=True):
    dim_s = "{}".format(bds.shape[1])
    for i in range(2, len(bds.shape)):
        dim_s = dim_s + ",{}".format(bds.shape[i])
    output_s = ''
    if out:
        output_s = 'Out'
    else:
        output_s = 'In'
    s = "Binding[{}] {}:{} {}x[{}],4".format(binding_idx, output_s, \
        bds.name, bds.shape[0], dim_s)
    print (s)
    return binding_idx + 1
        
def print_output(i, name, batch, data, data_shape):
    print_num = 64
    # eltcount = 1
    # for elt in data_shape[1:]:
    #     eltcount = eltcount * elt
    np_data = np.array(data)
    b_data = np_data[batch].flatten()
    eltcount = len(b_data)
    
    print('[%d]%s.batchId: %d eltCount:%d Data:' % (i, name, batch, eltcount))
    print('the elt count is: %d'% (eltcount))
    s_str = ''
    for i in range(print_num):
        s_str = s_str + str(b_data[i]) + '\t, '
    s_str = s_str + ' ... '
    if eltcount > print_num:
        for i in range(eltcount-print_num, eltcount):
            s_str = s_str + str(b_data[i]) + '\t, '
    print(s_str)
    
    
def main(onnx_path, image_path=None, cache_image=False, image_cache_path=None):
    file_check(onnx_path)
    # file_check(image_path)
    onnx_session = onnxruntime.InferenceSession(onnx_path) 

    batch_size = 1
    input_shape = [1,12,240,320]
    outputs_names = []
    outputs_shapes = []
    
    for inputs in onnx_session.get_inputs():
        # binding_idx = print_binding(binding_idx, inputs)
        if inputs.name == 'image':
            input_shape = inputs.shape
            batch_size = inputs.shape[0]
    for outputs in onnx_session.get_outputs():
        outputs_names.append(outputs.name)
        outputs_shapes.append(outputs.shape)
    print ('Tot batch number is %d' % (batch_size))
    print ('Read ONNX model %s' % (onnx_path))
    
    x = None
    input_shape[0] = 1    
    if image_path == None:
        x = np.zeros(input_shape).astype(np.float32)
    else:
        # input image
        f_tail = os.path.splitext(image_path)[-1][1:]
        if f_tail in ['png', 'jpg', 'bmp', 'jpeg']:
            im = Image.open(image_path)
            im_np = np.array(im)
            im_height, im_width, channel = im_np.shape
            if input_shape[1] == 12:
                if 3 == channel and input_shape[2]*2 == im_height and input_shape[3]*2 == im_width:
                    # image = (np.array(im1).astype(np.float32) - np.array([102.9801, 115.9465, 122.7717])) / 255.0
                    image = np.array(im).astype(np.float32) / 255.0
                    # image = (np.array(im1).astype(np.float32)) 
                    image_shape = image.shape

                    print ('readImage %s by pillow, size = %d (%dx%dx%d)' % (image_path, \
                        image_shape[0]*image_shape[1]*image_shape[2], \
                        image_shape[0], image_shape[1], image_shape[2]))

                    #res2channel:  3 channel to 12 channel
                    x = np.concatenate((image[::2, ::2, :], image[1::2, ::2, :], \
                        image[::2, 1::2, :], image[1::2, 1::2, :]), axis=-1).transpose(2,0,1).astype(np.float32)
                    x = x[np.newaxis, :]
                    if cache_image: 
                        if image_cache_path is not None:
                            x.tofile(image_cache_path)
                        else:
                            x.tofile(image_path + '.bin')    
                else:
                    print ('Bad image size ! Image with shape of [%dx%dx%d] is required.' % (input_shape[2]*2, input_shape[3]*2, input_shape[1]))
                    sys.exit('')
            # elif input_shape[1] == 3:
            else:
                print ('Unsupported onnx model as channel == ' + str(input_shape[1]))
        elif f_tail == 'bin':
            x = np.fromfile(image_path, dtype=np.float32).reshape(input_shape)  
        else:
            print ('Unsupported image type ! \(support: .png .jpg .bin\)')
            sys.exit('')

    binding_idx = 0
    for inputs in onnx_session.get_inputs():
        binding_idx = print_binding(binding_idx, inputs, out=False)
    for outputs in onnx_session.get_outputs():
        binding_idx = print_binding(binding_idx, outputs)
    
    if x is not None:
        xx = x
        for i in range(batch_size-1):
            xx = np.concatenate((xx[:,:,:,:], x[:,:,:,:]))
        
        t_start = time.time()
        onnx_inputs = {onnx_session.get_inputs()[0].name: xx}
        onnx_outs = onnx_session.run(None, onnx_inputs)
        t_end = time.time()
        dur_ms = (t_end - t_start) * 1e3

        print ('Tot outputSize=%d B' % (0))
        print ('CH.0 time = %f' % (dur_ms))
        print ('Checking output.')
        print ('Tot time = %f' % (dur_ms))
        print ('TestAll 1 in %.3fms range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, 90%%< %.3fms' % \
            (dur_ms, dur_ms, dur_ms, dur_ms, dur_ms, dur_ms))
        
        for i, name in enumerate(outputs_names):
            for bc in range(batch_size):            
                if name.find('filter') == -1:
                    print_output(i, name, bc, onnx_outs[i], outputs_shapes[i])
                else:
                    print ('todo')
        
   
        localtime = time.asctime( time.localtime(time.time()) )
        print ('Completion time', localtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model", type=str, required=True, help="ONNX model path e.g. /path/to/model.onnx ")
    parser.add_argument("-i", "--image_path", type=str, default='', help="Test image \(.png .jpg\) path e.g. /path/to/image.png. When empty, inputs will be set zeros.")

    args = parser.parse_args()
    if args.image_path == '':
        main(os.path.abspath(args.onnx_model))
    else:
        main(os.path.abspath(args.onnx_model), os.path.abspath(args.image_path))

