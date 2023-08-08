#!/bin/sh
# please check following parameters
# model name
onnx_model_name=cngp_main_v7.1.0
data_type=fp16
data_size=16
custom_model_output=notsr
calib_file=${onnx_model_name}_XPU_KL_main_narrow

# dirs
exec_dir=./bin
model_dir=./model
cuda_lib_dir=/usr/lib:/usr/local/lib:/usr/lib/cudalibs
NVTX_dir=/usr/lib/cudalibs/libcupti.so.11.4
# nvcc_dir=/usr/local/cuda/bin
lib_dir=./bin
python_dir=.
test_image=./300.n-604223e8-66d1-322b-a917-c5b2f1369f51.png

# exec and lib name
convert_exec_name=onnx2trt_orin
profile_exec_name=profile_model
enqueue_exec_name=test_multi_models
lib_name=libonnxtrt_orin.so

# all outputs are lld, mod_filter, sod_filter, tl_filter, bd, tfl2_filter, tsr_filter, mod3d_filter
output_num_option="-O lld -O mod_filter -O sod_filter -O tl_filter -O bd -O tfl2_filter -O mod3d_filter"

# other options
batch_size=2
check_result=1
gemv_ver=2
upsample_ver=2

# start from here, no need to change and get modified cuda graph automatically
# model name
model_type=${data_type}b${batch_size}
origin_model_name=${onnx_model_name}_${model_type}
custom_model_name=${onnx_model_name}_${model_type}_${custom_model_output}

# step 1, get layer information of onnx model in txt file
if [ ! -f "${onnx_model_name}.txt" ]; then
    echo "${exec_dir}/${convert_exec_name} -t ${onnx_model_name}.txt ${model_dir}/${onnx_model_name}.onnx"
    ${exec_dir}/${convert_exec_name} -t ${onnx_model_name}.txt ${model_dir}/${onnx_model_name}.onnx
fi
echo "\033[32mstep 1 done, get layer information of \033\033[35m${onnx_model_name}.onnx\033\033[32m model\033[0m"

# step 2, convert full model with all output
if [ ! -f "${model_dir}/${origin_model_name}.trt" ]; then
    echo "TRT_UPSAMPLE=${upsample_ver} TRT_GEMV=${gemv_ver} ${exec_dir}/${convert_exec_name} -o ${model_dir}/${origin_model_name}.trt -d ${data_size} -b ${batch_size} -l ${model_dir}/${onnx_model_name}.onnx 2>&1 | tee ${model_dir}/${origin_model_name}_convert.txt"
    TRT_UPSAMPLE=${upsample_ver} TRT_GEMV=${gemv_ver} ${exec_dir}/${convert_exec_name} -o ${model_dir}/${origin_model_name}.trt -d ${data_size} -b ${batch_size} -l ${model_dir}/${onnx_model_name}.onnx 2>&1 | tee ${model_dir}/${origin_model_name}_convert.txt
fi
echo "\033[32mstep 2 done, get converted \033\033[35m${origin_model_name}\033\033[32m model\033[0m"

# step 3, get remain layers of modified model
if [ ! -f "${onnx_model_name}_${custom_model_output}.txt" ]; then
    echo "${lib_dir}/${convert_exec_name} ${output_num_option} -G ${onnx_model_name}_${custom_model_output}.txt ${onnx_model_name}.txt"
    ${lib_dir}/${convert_exec_name} ${output_num_option} -G ${onnx_model_name}_${custom_model_output}.txt ${onnx_model_name}.txt
fi
echo "\033[32mstep 3 done, get remain layers of modified model from \033\033[35m${onnx_model_name}.onnx\033\033[32m model\033[0m"

# step 4, get profile output of full model
if [ ! -f "${origin_model_name}_profile.txt" ]; then
    echo "sudo LD_LIBRARY_PATH=${cuda_lib_dir} NVTX_INJECTION64_PATH=${NVTX_dir} ${lib_dir}/${profile_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -t 2 -m CudaGraph=True 2>&1 | tee ${origin_model_name}_profile.txt"
    sudo LD_LIBRARY_PATH=${cuda_lib_dir} NVTX_INJECTION64_PATH=${NVTX_dir} ${lib_dir}/${profile_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -t 2 -m CudaGraph=True | tee ${origin_model_name}_profile.txt
fi
echo "\033[32mstep 4 done, get profile output of full model from \033\033[35m${origin_model_name}.trt\033\033[32m model\033[0m"

# step 5, get chronological node table between layers and kernels of full model
# step 6, get deleted node list of customized output
if [ ! -f "${origin_model_name}_node_table.txt" ] || [ ! -f "${custom_model_name}_deleted_nodes.txt" ]; then
    echo "python3 ${python_dir}/cuda_graph.py mark_nodes ${batch_size} ${onnx_model_name} ${model_type} ${custom_model_output}"
    python3 ${python_dir}/cuda_graph.py mark_nodes ${batch_size} ${onnx_model_name} ${model_type} ${custom_model_output}
fi
echo "\033[32mstep 5 done, get node table of \033\033[35m${origin_model_name}\033\033[32m model\033[0m"
echo "\033[32mstep 6 done, get deleted node list of \033\033[35m${custom_model_name}\033\033[32m model\033[0m"

# step 7, run model with customized output by cuda graph and check results
if [ ${check_result} -eq 1 ]; then
    echo "\033[32mstep 9 start, check results of \033\033[35m${test_image}\033[0m"
    if [ ! -f "${origin_model_name}_result.txt" ]; then
        echo "${exec_dir}/${enqueue_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -m CudaGraph=True -t 2 -b ${batch_size} -g -i ${test_image} 2>&1 | tee ${origin_model_name}_result.txt"
        ${exec_dir}/${enqueue_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -m CudaGraph=True -t 2 -b ${batch_size} -g -i ${test_image} 2>&1 | tee ${origin_model_name}_result.txt
    fi
    echo "\033[32mget full model result of \033\033[35m${test_image}\033[0m"
    
    if [ ! -f "${custom_model_name}_result.txt" ]; then
        echo "${exec_dir}/${enqueue_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -m CudaGraph=True,CudaGraphNodeFile=${custom_model_name}_deleted_nodes.txt -t 2 -b ${batch_size} -g -i ${test_image} 2>&1 | tee ${custom_model_name}_result.txt"
        ${exec_dir}/${enqueue_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -m CudaGraph=True,CudaGraphNodeFile=${custom_model_name}_deleted_nodes.txt -t 2 -b ${batch_size} -g -i ${test_image} 2>&1 | tee ${custom_model_name}_result.txt
    fi
    echo "\033[32mget modified model result of \033\033[35m${test_image}\033[0m"

    echo "\033[32mchecking results:\033[0m"
    echo "python ${python_dir}/cuda_graph.py check_results ${batch_size} ${data_type} ${origin_model_name} ${custom_model_output} ${output_num_option}"
    python3 ${python_dir}/cuda_graph.py check_results ${batch_size} ${data_type} ${origin_model_name} ${custom_model_output} ${output_num_option}
    echo "\033[32mstep 7 done\033[0m"
else
    echo "${exec_dir}/${enqueue_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -m CudaGraph=True,CudaGraphNodeFile=./${custom_model_name}_deleted_nodes.txt -t 2 -b ${batch_size}"
    ${exec_dir}/${enqueue_exec_name} -s ${lib_dir}/${lib_name} -e ${model_dir}/${origin_model_name}.trt -m CudaGraph=True,CudaGraphNodeFile=./${custom_model_name}_deleted_nodes.txt -t 2 -b ${batch_size}
fi