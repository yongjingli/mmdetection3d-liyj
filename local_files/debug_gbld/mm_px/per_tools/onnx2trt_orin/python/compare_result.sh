#!/bin/sh

# default values
compare_script=./compare_result.py
onnxrt_script=./run_onnx.py
batch_size=1
input_type_1=0
input_type_2=0
thread_type=2
python_exec=python2
deviation_threshold=0.000001

# print help
Help() {
cat << EOF
-------------------------------------------------------------------Help script to compare result-------------------------------------------------------------------
Usage:
    sh compare_result.sh [-h|--help] [-M|--mode COMPARE_MODE] [-l|--lib LIB_PATH] [-e|--engine ENGINE_PATH] [-E|--testExec TEST_EXEC_PATH] ...

    [-h|--help]                                             Show this message
    [-M|--mode ONNX | LIB | MODEL]                          Specitfy the compare mode
                                                            ONNX:  compare TensorRT model results to ONNX model, now only support zero input
                                                                   If system support both ONNXRT and TensorRT, can compare the results by one command
                                                            LIB:   compare new library results to old library
                                                            MODEL: compare 1 model to another model, or compare 1 model with different input type
    [-l|--lib ./libonnxtrt1.so,./libonnxtrt2.so]            Specitfy the library path, allow only one lib "./libonnxtrt.so"
    [-e|--engine ./engine1.trt,./engine2.trt]               Specitfy the TensorRT engine path, allow only one engine "./engine.trt"
    [-Z|--engineZoo path/to/engines]                        Specitfy the TensorRT engine path, will test all trt engine in this folder
                                                            Now, only support use it on LIB mode of comparison, and all engine names are different
    [-C|--checkpoint path/to/onnx/checkpoint]               Specitfy the ONNX checkpoint file path
    [-E|--testExec path/to/test/executable]                 Specitfy the test executable file path
    [-c|--compare path/to/compare_result.py]                Specitfy the python script of comparison, default=./compare_result.py
    [-r|--runONNX path/to/run_onnx.py]                      Specitfy the python script of running checkpoint by ONNX runtime, default=./run_onnx.py
    [-i|--input path/to/input/file]                         Specitfy the input image/binary file/ppm file/...
    [-I|--inputType [0..5] or [0..5],[0..5]]                Specitfy the input type of twice execution, default=0,0
                                                            0:cpu float32, 1:gpu float32, 2: gpu 3HW (CHW) float32,
                                                            3: gpu 3HW (CHW) int8, 4:lidar CPU pcd, 5:cpu 3HW uint8
    [-b|--batch batch/size]                                 Specitfy the batch size, default=1
    [-m|--modelConfig model/config]                         Specitfy the model config, ex: Prioity=High,CudaGraph=True
    [-o|--output path/to/output/result/txt/file]            Specitfy the output txt file
                                                            If enable, will only run once and save the results of model 1
    [-O|--outputFolder path/to/output/file/folder]          Specitfy the output file folder path, to save results of all engines
                                                            Must use it with -Z, if enable, every engine will only run once
    [-t|--text path/to/compared/txt/file]                   Specitfy the txt file to be compared
                                                            If enable, will read this file for comparison to model 1
    [-T|--textFolder path/to/text/file/folder]              Specitfy the txt file folder path, to load result files from -O
                                                            Must use it with -Z, if enable, every engine in engine zoo will only run once
    [-d|--deviation deviation/threshold]                    Specitfy max allowed deviation, default=1e-6
    [-D|--thread thread/type]                               Specitfy the thread type, default=2
    [-P|--python python, python2 or python3]                Specitfy the installed python bin, default=python2

Sample:
    sh compare_result.sh -M ONNX -C ./checkpoint.pb -r ./run_onnx.py -o ./onnx_result.txt -l ./libonnxtrt.so -e ./test.trt -E ./test_multi_models -b 2 -P python3
    sh compare_result.sh -M ONNX -C ./checkpoint.pb -r ./run_onnx.py -o ./onnx_result.txt -P python3
    sh compare_result.sh -M ONNX -l ./libonnxtrt.so -e ./test.trt -E ./test_multi_models -I 0 -b 2 -t ./onnx_result.txt
    sh compare_result.sh -M LIB -l ./libonnxtrt_1.so,./libonnxtrt_2.so -e ./test.trt -E ./test_multi_models -c ./compare_result.py -i ./test_image.jpg
    sh compare_result.sh -M LIB -l ./libonnxtrt_1.so -Z ./engineZoo -E ./test_multi_models -O ./textFolder
    sh compare_result.sh -M LIB -l ./libonnxtrt_2.so -Z ./engineZoo -E ./test_multi_models -T ./textFolder
    sh compare_result.sh -M MODEL -l ./libonnxtrt.so -e ./test.trt -E ./test_multi_models -c ./compare_result.py -i ./test_image.jpg -I 0,3 -b 2
    sh compare_result.sh -M MODEL -l ./libonnxtrt.so -e ./test.trt -E ./test_multi_models -I 0 -i ./test_image.jpg -o result.txt
    sh compare_result.sh -M MODEL -l ./libonnxtrt.so -e ./test.trt -E ./test_multi_models -I 3 -i ./test_image.jpg -t result.txt
    sh compare_result.sh -M MODEL -l ./libonnxtrt.so -e ./test_1.trt,./test_2.trt -E ./test_multi_models -d 0.0001 -P python -m CudaGraph=True
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
EOF
}

# run ONNX comparison
ONNX_Mode() {
    # check and print paras
    # 0: -C, 1: -e, 2: -C -e
    compare_flag=0
    if [ "${checkpoint_path}" != "" ] && [ "${engine_path_1}" != "" ]; then
        compare_flag=2
    elif [ "${checkpoint_path}" != "" ]; then
        compare_flag=0
    elif [ "${engine_path_1}" != "" ]; then
        compare_flag=1
    else
        echo "Haven't specitfy checkpoint file or engine file"; Help; exit 1
    fi

    if [ "${compare_flag}" = "0" ] || [ "${compare_flag}" = "2" ]; then
        echo "\033[32mCheckpoint:\033[0m\t${checkpoint_path}"
        if [ ! -f "${checkpoint_path}" ]; then
            echo "Cannot find checkpoint file"; Help; exit 1
        fi

        echo "\033[32mONNXRT Script:\033[0m\t${onnxrt_script}"
        if [ ! -f "${onnxrt_script}" ]; then
            echo "Cannot find ONNXRT script"; Help; exit 1
        fi
    fi

    if [ "${compare_flag}" = "1" ] || [ "${compare_flag}" = "2" ]; then
        echo "\033[32mLib Path:\033[0m\t${lib_path_1}"
        if [ ! -f "${lib_path_1}" ]; then
            echo "Cannot find library file"; Help; exit 1
        fi

        echo "\033[32mEngine Path:\033[0m\t${engine_path_1}"
        if [ ! -f "${engine_path_1}" ]; then
            echo "Cannot find engine file"; Help; exit 1
        fi
    fi

    echo "\033[32mInput File:\033[0m\t${input_file}"

    if [ "${compare_flag}" = "1" ] || [ "${compare_flag}" = "2" ]; then
        echo "\033[32mInput Type:\033[0m\t${input_type_1}"
    fi

    echo "\033[32mBatch Size:\033[0m\t${batch_size}"

    echo "\033[32mMax Deviation:\033[0m\t$(printf "%.4f" $(echo "scale=5;${deviation_threshold}*100"|bc))%"

    if [ "${compare_flag}" = "1" ] || [ "${compare_flag}" = "2" ]; then
        if [ "${model_config}" != "" ]; then
            echo "\033[32mModel Config:\033[0m\t${model_config}"
        fi
    fi

    if [ "${compare_flag}" = "0" ] || [ "${compare_flag}" = "1" ]; then
        if [ "${output_path}" = "" ] && [ "${text_path}" = "" ]; then
            echo "Haven't specitfy output file or text file"; Help; exit 1
        fi
    else
        if [ "${output_path}" != "" ]; then
            echo "No need specitfy output file with both -C and -e, ignore -o config"
            output_path=""
        fi
        if [ "${text_path}" != "" ]; then
            echo "No need specitfy text file with both -C and -e,  ignore -t config"
            text_path=""
        fi
    fi

    if [ "${output_path}" != "" ]; then
        echo "\033[32mOutput File:\033[0m\t${output_path}"
    fi

    if [ "${text_path}" != "" ]; then
        if [ "${output_path}" != "" ]; then
            echo "\033[32mText File:\033[0m\tExist -o config, ignore -t config"
            break
        else
            echo "\033[32mText File:\033[0m\t${text_path}"
            if [ ! -f "${text_path}" ]; then
                echo "Cannot find result text file"; Help; exit 1
            fi
        fi
    fi

    if [ "${compare_flag}" = "1" ] || [ "${compare_flag}" = "2" ]; then
        echo "\033[32mThread Type:\033[0m\t${thread_type}"
    fi

    if [ "${compare_flag}" = "0" ] || [ "${compare_flag}" = "2" ]; then
        if [ "${python_exec}" != "python3" ]; then
            echo "run_onnx.py need python3 to run, change Python Exec to python3"
            python_exec=python3
        fi
    fi
    echo "\033[32mPython Exec:\033[0m\t/usr/bin/${python_exec}"
    if [ ! -x "/usr/bin/${python_exec}" ]; then
        echo "Cannot find python exec"; Help; exit 1
    fi

    # start run models
    echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "\033[32mStart comparison...\033[0m"
    if [ ! -f "${input_file}" ]; then
        echo "Null input file, use zero input"
    fi

    if [ -f "${input_file}" ]; then
        for i in `seq 0 $(( ${batch_size}-1 ))`
        do
            input_file_arg="${input_file_arg} -i ${input_file}"
        done
        input_file_arg_py="-i ${input_file}"
    fi

    if [ "${model_config}" != "" ]; then
        model_config_arg="-m ${model_config}"
    fi

    # run onnx checkpoint
    if [ "${compare_flag}" = "0" ]; then
        # record checkpoint result
        if [ "${output_path}" != "" ]; then
            # run checkpoint
            echo "\033[32mRunning Checkpoint:\033[0m /usr/bin/${python_exec} ${onnxrt_script} -o ${checkpoint_path} ${input_file_arg_py} 2>&1 | tee ${output_path}"
            # add ${batch_size}
            /usr/bin/${python_exec} ${onnxrt_script} -o ${checkpoint_path} ${input_file_arg_py} 2>&1 | tee ${output_path}
            echo "\033[32mDone\033[0m"
            return 0
        fi

        echo "\033[32mRunning Checkpoint:\033[0m /usr/bin/${python_exec} ${onnxrt_script} -o ${checkpoint_path} ${input_file_arg_py} 2>&1 | tee result_1.txt"
        # add ${batch_size}
        /usr/bin/${python_exec} ${onnxrt_script} -o ${checkpoint_path} ${input_file_arg_py} 2>&1 | tee result_1.txt

        compare_file_1="./result_1.txt"
        compare_file_2="${text_path}"

        echo "\033[32mCompare result between\033[0m ${compare_file_1} \033[32mand\033[0m ${compare_file_2}"
        echo "\033[32mComparing Command:\033[0m /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}"
        /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}
        echo "\033[32mDone\033[0m"
    fi

    # run trt engine
    if [ "${compare_flag}" = "1" ]; then
        # record engine result
        if [ "${output_path}" != "" ]; then
            # run engine
            echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_path}"
            ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_path}
            echo "\033[32mDone\033[0m"
            return 0
        fi

        echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_1.txt"
        ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_1.txt

        compare_file_1="./result_1.txt"
        compare_file_2="${text_path}"

        echo "\033[32mCompare result between\033[0m ${compare_file_1} \033[32mand\033[0m ${compare_file_2}"
        echo "\033[32mComparing Command:\033[0m /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}"
        /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}
        echo "\033[32mDone\033[0m"
    fi

    # run checkpoint and engine
    if [ "${compare_flag}" = "2" ]; then
        echo "\033[32mRunning Checkpoint:\033[0m /usr/bin/${python_exec} ${onnxrt_script} -o ${checkpoint_path} ${input_file_arg_py} 2>&1 | tee result_1.txt"
        # add ${batch_size}
        /usr/bin/${python_exec} ${onnxrt_script} -o ${checkpoint_path} ${input_file_arg_py} 2>&1 | tee result_1.txt
        compare_file_1="./result_1.txt"

        echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_2.txt"
        ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_2.txt

        compare_file_2="./result_2.txt"

        # compare result
        echo "\033[32mCompare result between\033[0m ${compare_file_1} \033[32mand\033[0m ${compare_file_2}"
        echo "\033[32mComparing Command:\033[0m /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}"
        /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}
        echo "\033[32mDone\033[0m"
    fi
}

# run library comparison
LIB_Mode() {
    # check and print paras
    echo "\033[32mLib Path 1:\033[0m\t${lib_path_1}"
    if [ ! -f "${lib_path_1}" ]; then
        echo "Cannot find library file"; Help; exit 1
    fi
    echo "\033[32mLib Path 2:\033[0m\t${lib_path_2}"
    if [ ! -f "${lib_path_2}" ]; then
        echo "Cannot find library file"; Help; exit 1
    fi

    if [ "${engine_path_1}" != "" ] && [ "${engine_zoo_path}" != "" ]; then
        echo "Cannot support -e and -Z at the same time"
        Help
        exit 1
    fi
    if [ "${engine_zoo_path}" = "" ]; then
        echo "\033[32mEngine Path 1:\033[0m\t${engine_path_1}"
        if [ ! -f "${engine_path_1}" ]; then
            echo "Cannot find engine file"; Help; exit 1
        fi
        echo "\033[32mEngine Path 2:\033[0m\t${engine_path_2}"
        if [ ! -f "${engine_path_2}" ]; then
            echo "Cannot find engine file"; Help; exit 1
        fi
    else
        echo "\033[32mEngineZoo Path:\033[0m\t${engine_zoo_path}"
        if [ ! -d "${engine_zoo_path}" ]; then
            echo "Cannot find engine zoo folder"; Help; exit 1
        fi
        engine_num=0
        for file in $(find ${engine_zoo_path} -name "*.*" | awk '$0 ~ /trt$/ {print i$0}')
        # for file in $(ls ${engine_zoo_path} | awk '$0 ~ /trt$/ {print i$0}' i=${engine_zoo_path}'/')
        do
            engine_num=$((${engine_num}+1))
            [ -f $file ] && echo $file
        done
        if [ ${engine_num} = 0 ]; then
            echo "Cannot find engine file in engine zoo"; Help; exit 1
        else
            echo "Find ${engine_num} TensorRT engines"
        fi
    fi

    echo "\033[32mTest Exec:\033[0m\t${test_exec}"
    if [ ! -f "${test_exec}" ]; then
        echo "Cannot find test executable file"; Help; exit 1
    fi

    echo "\033[32mCompare Script:\033[0m\t${compare_script}"
    if [ ! -f "${compare_script}" ]; then
        echo "Cannot find compare script"; Help; exit 1
    fi

    echo "\033[32mInput File:\033[0m\t${input_file}"

    echo "\033[32mInput Type 1:\033[0m\t${input_type_1}"
    echo "\033[32mInput Type 2:\033[0m\t${input_type_2}"

    echo "\033[32mBatch Size:\033[0m\t${batch_size}"

    echo "\033[32mMax Deviation:\033[0m\t$(printf "%.4f" $(echo "scale=5;${deviation_threshold}*100"|bc))%"

    if [ "${model_config}" != "" ]; then
        echo "\033[32mModel Config:\033[0m\t${model_config}"
    fi

    # 0: -e, 1: -e -o, 2: -e -t 
    # 3: -Z, 4: -Z -O, 5: -Z -T
    compare_flag=0
    if [ "${engine_zoo_path}" != "" ]; then
        compare_flag=3

        if [ "${output_path}" != "" ]; then
            echo "\033[32mOutput File:\033[0m\tExist -Z config, ignore -o config"
        fi
        if [ "${output_folder_path}" != "" ]; then
            echo "\033[32mOutput Folder:\033[0m\t${output_folder_path}"
            if [ ! -d "${output_folder_path}" ]; then
                mkdir -p ${output_folder_path}
                echo "Folder not exist, create"
            fi
            compare_flag=4
        fi

        if [ "${text_path}" != "" ]; then
            echo "\033[32mText File:\033[0m\tExist -Z config, ignore -t config"
        fi
        if [ "${text_folder_path}" != "" ]; then
            if [ "${output_folder_path}" != "" ]; then
                echo "\033[32mText Folder:\033[0m\tExist -O config, ignore -T config"
                break
            else
                echo "\033[32mText Folder:\033[0m\t${text_folder_path}"
                if [ ! -d "${text_folder_path}" ]; then
                    echo "Cannot find result text folder"; Help; exit 1
                fi
                compare_flag=5
            fi
        fi
    else
        if [ "${output_path}" != "" ]; then
            echo "\033[32mOutput File:\033[0m\t${output_path}"
            compare_flag=1
        fi
        if [ "${output_folder_path}" != "" ]; then
            echo "\033[32mOutput Folder:\033[0m\t$Exist -e config, ignore -O config"
        fi

        if [ "${text_path}" != "" ]; then
            if [ "${output_path}" != "" ]; then
                echo "\033[32mText File:\033[0m\tExist -o config, ignore -t config"
                break
            else
                echo "\033[32mText File:\033[0m\t${text_path}"
                if [ ! -f "${text_path}" ]; then
                    echo "Cannot find result test file"; Help; exit 1
                fi
                compare_flag=2
            fi
        fi
        if [ "${text_folder_path}" != "" ]; then
            echo "\033[32mText Folder:\033[0m\tExist -e config, ignore -T config"
        fi
    fi

    echo "\033[32mThread Type:\033[0m\t${thread_type}"

    echo "\033[32mPython Exec:\033[0m\t${python_exec}"
    if [ ! -x "/usr/bin/${python_exec}" ]; then
        echo "Cannot find python exec"; Help; exit 1
    fi

    # start run models
    echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "\033[32mStart comparison...\033[0m"
    if [ ! -f "${input_file}" ]; then
        echo "Null input file, use zero input"
    fi

    if [ -f "${input_file}" ]; then
        for i in `seq 0 $(( ${batch_size}-1 ))`
        do
            input_file_arg="${input_file_arg} -i ${input_file}"
        done
    fi

    if [ "${model_config}" != "" ]; then
        model_config_arg="-m ${model_config}"
    fi

    # compare flag = 1
    if [ "${compare_flag}" = "1" ]; then
        echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_path}"
        ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_path}
        echo "\033[32mDone\033[0m"
        return 0;
    fi

    # compare flag = 0 or 2
    if [ "${compare_flag}" = "0" ] || [ "${compare_flag}" = "2" ]; then
        # run model 1
        echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_1.txt"
        ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_1.txt
        compare_file_1="./result_1.txt"

        if [ "${text_path}" != "" ]; then
            # use given text file
            compare_file_2="${text_path}"
        else
            # run model 2
            echo "\033[32mRunning Model 2:\033[0m ${test_exec} -e ${engine_path_2} -s ${lib_path_2} ${input_file_arg} -b ${batch_size} -g -G ${input_type_2} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_2.txt"
            ${test_exec} -e ${engine_path_2} -s ${lib_path_2} ${input_file_arg} -b ${batch_size} -g -G ${input_type_2} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_2.txt
            compare_file_2="./result_2.txt"
        fi

        # compare result
        echo "\033[32mCompare result between\033[0m ${compare_file_1} \033[32mand\033[0m ${compare_file_2}"
        echo "\033[32mComparing Command:\033[0m /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}"
        /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}

        echo "\033[32mDone\033[0m"
        return 0;
    fi

    # compare flag = 4
    if [ "${compare_flag}" = "4" ]; then
        for engine_file in $(find ${engine_zoo_path} -name "*.*" | awk '$0 ~ /trt$/ {print i$0}')
        do
            output_file=`echo ${engine_file} | awk -F "/" '{print $NF}'`
            output_file=`echo ${output_file} | awk -F ".trt" '{print $1}'`
            output_file="${output_folder_path}/${output_file}_result_out.txt"
            
            echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_file} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_file}"
            ${test_exec} -e ${engine_file} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_file}
        done

        echo "\033[32mDone\033[0m"
        return 0;
    fi

    # compare flag = 3 or 5
    if [ "${compare_flag}" = "3" ] || [ "${compare_flag}" = "5" ]; then
        for engine_file in $(find ${engine_zoo_path} -name "*.*" | awk '$0 ~ /trt$/ {print i$0}')
        do
            file=`echo ${engine_file} | awk -F "/" '{print $NF}'`
            file=`echo ${file} | awk -F ".trt" '{print $1}'`

            # run model 1
            echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_file} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ./${file}_result_1.txt"
            ${test_exec} -e ${engine_file} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ./${file}_result_1.txt
            compare_file_1="./${file}_result_1.txt"

            if [ "${compare_flag}" = "5" ]; then
                compare_file_2="${text_folder_path}/${file}_result_out.txt"
            else
                # run model 2
                echo "\033[32mRunning Model 2:\033[0m ${test_exec} -e ${engine_file} -s ${lib_path_2} ${input_file_arg} -b ${batch_size} -g -G ${input_type_2} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ./${file}_result_2.txt"
                ${test_exec} -e ${engine_file} -s ${lib_path_2} ${input_file_arg} -b ${batch_size} -g -G ${input_type_2} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ./${file}_result_2.txt
                compare_file_2="./${file}_result_2.txt"
            fi

            echo "\033[32mCompare result between\033[0m ${compare_file_1} \033[32mand\033[0m ${compare_file_2}"
            echo "\033[32mComparing Command:\033[0m /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold} 2>&1 | tee ./${file}_compare.txt"
            /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold} 2>&1 | tee ./${file}_compare.txt
        done

        echo "\033[32mDone\033[0m"
        return 0;
    fi
}

# run MODEL comparison
MODEL_Mode() {
    # check and print paras
    echo "\033[32mLib Path 1:\033[0m\t${lib_path_1}"
    if [ ! -f "${lib_path_1}" ]; then
        echo "Cannot find library file"; Help; exit 1
    fi
    echo "\033[32mLib Path 2:\033[0m\t${lib_path_2}"
    if [ ! -f "${lib_path_2}" ]; then
        echo "Cannot find library file"; Help; exit 1
    fi

    echo "\033[32mEngine Path 1:\033[0m\t${engine_path_1}"
    if [ ! -f "${engine_path_1}" ]; then
        echo "Cannot find engine file"; Help; exit 1
    fi
    echo "\033[32mEngine Path 2:\033[0m\t${engine_path_2}"
    if [ ! -f "${engine_path_2}" ]; then
        echo "Cannot find engine file"; Help; exit 1
    fi

    echo "\033[32mTest Exec:\033[0m\t${test_exec}"
    if [ ! -f "${test_exec}" ]; then
        echo "Cannot find test executable file"; Help; exit 1
    fi

    echo "\033[32mCompare Script:\033[0m\t${compare_script}"
    if [ ! -f "${compare_script}" ]; then
        echo "Cannot find compare script"; Help; exit 1
    fi

    echo "\033[32mInput File:\033[0m\t${input_file}"

    echo "\033[32mInput Type 1:\033[0m\t${input_type_1}"
    echo "\033[32mInput Type 2:\033[0m\t${input_type_2}"

    echo "\033[32mBatch Size:\033[0m\t${batch_size}"

    echo "\033[32mMax Deviation:\033[0m\t$(printf "%.4f" $(echo "scale=5;${deviation_threshold}*100"|bc))%"

    if [ "${model_config}" != "" ]; then
        echo "\033[32mModel Config:\033[0m\t${model_config}"
    fi

    if [ "${output_path}" != "" ]; then
        echo "\033[32mOutput File:\033[0m\t${output_path}"
    fi

    if [ "${text_path}" != "" ]; then
        if [ "${output_path}" != "" ]; then
            echo "\033[32mText File:\033[0m\tExist -o config, ignore -t config"
            break
        else
            echo "\033[32mText File:\033[0m\t${text_path}"
            if [ ! -f "${text_path}" ]; then
                echo "Cannot find result text file"; Help; exit 1
            fi
        fi
    fi

    echo "\033[32mThread Type:\033[0m\t${thread_type}"

    echo "\033[32mPython Exec:\033[0m\t${python_exec}"
    if [ ! -x "/usr/bin/${python_exec}" ]; then
        echo "Cannot find python exec"; Help; exit 1
    fi

    # start run models
    echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "\033[32mStart comparison...\033[0m"
    if [ ! -f "${input_file}" ]; then
        echo "Null input file, use zero input"
    fi

    if [ -f "${input_file}" ]; then
        for i in `seq 0 $(( ${batch_size}-1 ))`
        do
            input_file_arg="${input_file_arg} -i ${input_file}"
        done
    fi

    if [ "${model_config}" != "" ]; then
        model_config_arg="-m ${model_config}"
    fi

    # record model 1 result
    if [ "${output_path}" != "" ]; then
        # run model 1
        echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_path}"
        ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee ${output_path}
        echo "\033[32mDone\033[0m"
        return 0
    fi

    # run model 1
    echo "\033[32mRunning Model 1:\033[0m ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_1.txt"
    ${test_exec} -e ${engine_path_1} -s ${lib_path_1} ${input_file_arg} -b ${batch_size} -g -G ${input_type_1} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_1.txt
    compare_file_1="./result_1.txt"

    if [ "${text_path}" != "" ]; then
        # use given text file
        compare_file_2="${text_path}"
    else
        # run model 2
        echo "\033[32mRunning Model 2:\033[0m ${test_exec} -e ${engine_path_2} -s ${lib_path_2} ${input_file_arg} -b ${batch_size} -g -G ${input_type_2} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_2.txt"
        ${test_exec} -e ${engine_path_2} -s ${lib_path_2} ${input_file_arg} -b ${batch_size} -g -G ${input_type_2} -T ${thread_type} -t 1 ${model_config_arg} 2>&1 | tee result_2.txt
        compare_file_2="./result_2.txt"
    fi

    # compare result
    echo "\033[32mCompare result between\033[0m ${compare_file_1} \033[32mand\033[0m ${compare_file_2}"
    echo "\033[32mComparing Command:\033[0m /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}"
    /usr/bin/${python_exec} ${compare_script} ${batch_size} ${compare_file_1} ${compare_file_2} ${deviation_threshold}
    echo "\033[32mDone\033[0m"
}

# read paras
while true;
do
    case $1 in
        -h|--help):
            Help; exit 0; ;;
        -M|--mode):
            compare_mode=$2; shift 2; ;;
        -l|--lib):
            lib_path_1=${2%,*}
            lib_path_2=${2#*,}
            shift 2
            ;;
        -e|--engine):
            engine_path_1=${2%,*}
            engine_path_2=${2#*,}
            shift 2
            ;;
        -Z|--engineZoo):
            engine_zoo_path=$2; shift 2; ;;
        -C|--checkpoint):
            checkpoint_path=$2; shift 2; ;;
        -E|--testExec):
            test_exec=$2; shift 2; ;;
        -c|--compare):
            compare_script=$2; shift 2; ;;
        -r|--runONNX):
            onnxrt_script=$2; shift 2; ;;
        -i|--input):
            input_file=$2; shift 2; ;;
        -I|--inputType):
            input_type_1=${2%,*}
            input_type_2=${2#*,}
            shift 2
            ;;
        -b|--batch):
            batch_size=$2; shift 2; ;;
        -m|--modelConfig):
            model_config=$2; shift 2; ;;
        -o|--output):
            output_path=$2; shift 2; ;;
        -O|--outputFolder):
            output_folder_path=$2; shift 2; ;;
        -t|--text):
            text_path=$2; shift 2; ;;
        -T|--textFolder):
            text_folder_path=$2; shift 2; ;;
        -d|--deviation):
            deviation_threshold=$2; shift 2; ;;
        -D|--thread):
            thread_type=$2; shift 2; ;;
        -P|--python):
            python_exec=$2; shift 2; ;;
        "")
            break; ;;
        *)
            echo "\033[31munrecognized arguments\033[0m ${1}"; Help; exit 1; ;;
    esac
done

echo "\033[32mCompare Mode:\033[0m\t${compare_mode}"
if [ "$compare_mode" != "ONNX" ] && [ "$compare_mode" != "MODEL" ] && [ "$compare_mode" != "LIB" ]; then
    echo "Cannot recognize compare mode"
    Help
    exit 1
fi

if [ "$compare_mode" = "ONNX" ]; then
    ONNX_Mode
    exit 0
fi

if [ "$compare_mode" = "LIB" ]; then
    LIB_Mode
    exit 0
fi

if [ "$compare_mode" = "MODEL" ]; then
    echo "In MODEL mode, suggest use only same libonnx2trt.so"
    MODEL_Mode
    exit 0
fi