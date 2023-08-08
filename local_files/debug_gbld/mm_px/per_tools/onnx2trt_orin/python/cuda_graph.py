import sys
import struct
import os

def mark_nodes(argv):
    batch_size  = int(argv[1])
    onnx_name   = argv[2]
    model_type  = argv[3]
    custom_name = argv[4]

    model_name   = onnx_name + "_" + model_type
    profile_name = model_name + "_profile.txt"
    table_file   = model_name + "_node_table.txt"
    custom_file  = onnx_name + "_" + custom_name + ".txt"
    deleted_nodes_file = model_name + "_" + custom_name + "_deleted_nodes.txt"

    # read name of layers from txt file chronologically
    with open(profile_name, 'r', encoding='unicode_escape') as f:
        profile_layers = [i[:-1] for i in f.readlines()]

    table = []
    i_kernel = 0
    i_layer = 0
    i = 0
    range_name = []
    start_flag = 0
    for i in range(0, len(profile_layers)):
        if profile_layers[i].find("::enqueue") != -1 and start_flag == 1:
            while True:
                i = i + 1
                if profile_layers[i].find("GraphNode Cloned") != -1:
                    break
                layer = profile_layers[i].split("\t")
                if layer[1] == "name = ExecutionContext::enqueue":
                    continue
                if layer[0] == "#CUPTI_CBID_NVTX_nvtxDomainRangePushEx":
                    name = layer[1].split(" = ")
                    range_name = name[1]
                    i_layer = i_layer + 1
                if layer[0] == "GraphNode Create":
                    if len(layer) == 4:
                        table.append([i_kernel, range_name, layer[3]])
                    else:
                        table.append([i_kernel, range_name])
                    i_kernel = i_kernel + 1
            break
        if profile_layers[i].find("::enqueue") != -1:
            start_flag += 1

    # write node table into txt file
    with open(table_file, 'w') as f:
        for i in table:
            if len(i) == 3:
                f.write(str(i[0]) + "  " + i[1] + "  " + i[2] + "\n")
            else:
                f.write(str(i[0]) + "  " + i[1] + "\n")

    # read name of notsr layers from txt file chronologically
    with open(custom_file, 'r') as f:
        custom_layers = [i[:-1] for i in f.readlines()]

    # find deleted kernel
    kernel_flag = [False] * i_kernel
    for i in range(0, len(custom_layers)):
        layer = custom_layers[i]
        for j in range(0, i_kernel):
            table_tmp = table[j][1] + " "
            layer_a = layer + " "
            layer_b = layer + "["
            if table_tmp.find(layer_a) != -1:
                kernel_flag[j] = True
            if table_tmp.find(layer_b) != -1:
                kernel_flag[j] = True

    deleted_nodes = []
    for i in range(0, i_kernel):
        if kernel_flag[i] == False:
            deleted_nodes.append(i)

    # write deleted node file into txt file
    with open(deleted_nodes_file, 'w') as f:
        for i in range(0, batch_size - 1):
            f.write("\n")
        for i in range(0, len(deleted_nodes) - 1):
            f.write(str(deleted_nodes[i]) + " ")
        if len(deleted_nodes) - 1 < 0:
            f.write("\n")
        else:
            f.write(str(deleted_nodes[len(deleted_nodes) - 1]) + "\n")

def mark_nodes_model(argv):
    batch_size  = int(argv[1])
    onnx_name   = argv[2]
    model_type  = argv[3]
    custom_name = argv[4]

    model_name   = onnx_name + "_" + model_type
    model_file   = model_name + ".trt"
    profile_name = model_name + "_profile.txt"
    table_file   = model_name + "_node_table.txt"
    custom_file  = onnx_name + "_" + custom_name + ".txt"

    # read name of layers from txt file chronologically
    with open(profile_name, 'r', encoding='unicode_escape') as f:
        profile_layers = [i[:-1] for i in f.readlines()]

    table = []
    i_kernel = 0
    i_layer = 0
    i = 0
    range_name = []
    start_flag = 0
    for i in range(0, len(profile_layers)):
        if profile_layers[i].find("::enqueue") != -1 and start_flag == 1:
            while True:
                i = i + 1
                if profile_layers[i].find("GraphNode Cloned") != -1:
                    break
                layer = profile_layers[i].split("\t")
                if layer[1] == "name = ExecutionContext::enqueue":
                    continue
                if layer[0] == "#CUPTI_CBID_NVTX_nvtxDomainRangePushEx":
                    name = layer[1].split(" = ")
                    range_name = name[1]
                    i_layer = i_layer + 1
                if layer[0] == "GraphNode Create":
                    if len(layer) == 4:
                        table.append([i_kernel, range_name, layer[3]])
                    else:
                        table.append([i_kernel, range_name])
                    i_kernel = i_kernel + 1
            break
        if profile_layers[i].find("::enqueue") != -1:
            start_flag += 1

    # write node table into txt file
    with open(table_file, 'w') as f:
        for i in table:
            if len(i) == 3:
                f.write(str(i[0]) + "  " + i[1] + "  " + i[2] + "\n")
            else:
                f.write(str(i[0]) + "  " + i[1] + "\n")

    # read name of notsr layers from txt file chronologically
    with open(custom_file, 'r') as f:
        custom_layers = [i[:-1] for i in f.readlines()]

    # find deleted kernel
    kernel_flag = [False] * i_kernel
    for i in range(0, len(custom_layers)):
        layer = custom_layers[i]
        for j in range(0, i_kernel):
            table_tmp = table[j][1] + " "
            layer_a = layer + " "
            layer_b = layer + "["
            if table_tmp.find(layer_a) != -1:
                kernel_flag[j] = True
            if table_tmp.find(layer_b) != -1:
                kernel_flag[j] = True

    deleted_nodes = []
    for i in range(0, i_kernel):
        if kernel_flag[i] == False:
            deleted_nodes.append(i)

    # write deleted node info at the end of model
    model = open(model_file, 'rb')
    print("write deleted node list to model: " + model_file)
    model.read(16)
    record_length = struct.unpack("i", model.read(4))[0] + 0x18
    print("record length: " + str(record_length))
    file_length = os.path.getsize(model_file)
    print("file length:   " + str(file_length))
    if record_length != file_length:
        print("length recorded in model does not equal binary file length")
    
    print(len(deleted_nodes))
    list_length = 4 + 4 * batch_size + 4 * len(deleted_nodes)
    print("Written list_length: " + str(list_length))
    num_byte = 0
    with open(model_file, 'ab') as f:
        a = struct.pack("4s", bytes("List".encode('utf-8')))
        f.write(a)
        num_byte += 4
        a = struct.pack("i", list_length)
        f.write(a)
        num_byte += 4
        a = struct.pack("i", batch_size)
        f.write(a)
        num_byte += 4
        for i in range(0, batch_size):
            if i != batch_size - 1:
                a = struct.pack("i", 0)
                f.write(a)
                num_byte += 4
            else:
                a = struct.pack("i", len(deleted_nodes))
                f.write(a)
                num_byte += 4
                for j in deleted_nodes:
                    a = struct.pack("i", j)
                    f.write(a)
                    num_byte += 4
    print("Write " + str(num_byte) + " byte in model " + model_file)
    print("After write, the size of model is " + str(os.path.getsize(model_file)))
    

def find_next_output(file, output_name, start_line):
    i = start_line
    while i < len(file):
        if file[i].find(output_name) != -1:
            return i
        i = i + 1
    return -1

def check_results(argv):
    batch_size  = int(argv[1])
    data_type   = argv[2]
    model_name  = argv[3]
    custom_name = argv[4]

    origin_result_file = model_name + "_result.txt"
    custom_result_file = model_name + "_" + custom_name + "_result.txt"

    # read result of origin outputs
    with open(origin_result_file, 'r') as f:
        origin_results = [i[:-1] for i in f.readlines()]

    # collect output
    output_flag = []
    for i in origin_results:
        if i.find("Binding[") != -1:
            tmp = i.split(" ")
            tmp = tmp[1].split(":")
            if tmp[1] != "image":
                output_flag.append([tmp[1], 0])
    for i in argv:
        for j in output_flag:
            if i == j[0]:
                j[1] = 1
                break

    # read result of modified outputs
    with open(custom_result_file, 'r') as f:
        custom_results = [i[:-1] for i in f.readlines()]

    # from now a means origin result, b means customized result
    a_now = find_next_output(origin_results, output_flag[0][0] + ".batchId: 0", 0)
    b_now = find_next_output(custom_results, output_flag[0][0] + ".batchId: 0", 0)
    for i in range(0, len(output_flag)):
        for batch in range(0, batch_size):
            output_name = output_flag[i][0] + "." + str(batch)
            # find current output result range of line
            if i == len(output_flag) - 1 and batch == batch_size - 1:
                a_next = len(origin_results) - 1
                # a_next = len(origin_results)
                b_next = len(custom_results) - 1
                # b_next = len(custom_results)
            else:
                next_batch = batch + 1
                if next_batch == batch_size:
                    a_next = find_next_output(origin_results, output_flag[i + 1][0] + ".batchId: 0", a_now)
                    b_next = find_next_output(custom_results, output_flag[i + 1][0] + ".batchId: 0", b_now)
                else:
                    a_next = find_next_output(origin_results, output_flag[i][0] + ".batchId: " + str(next_batch), a_now)
                    b_next = find_next_output(custom_results, output_flag[i][0] + ".batchId: " + str(next_batch), b_now)
            
            # check result
            if output_flag[i][1] == 1:            
                print("checking output \033[35m" + output_name + "\033[0m")
                # directly check
                if output_flag[i][0] == "lld" or output_flag[i][0] == "bd":
                    if a_next - a_now != b_next - b_now:
                        print(output_name + ": line number in full model doesn't equal line in modified model")
                        print("a: " + str(a_next - a_now) + " b: " + str(b_next - b_now))
                    for a, b in zip(range(a_now, a_next), range(b_now, b_next)):
                        if origin_results[a] != custom_results[b]:
                            print(output_name + ": line " + str(a) + " in full model doesn't equal line " + str(b) + " in modified model")
                            print("a: " + origin_results[a])
                            print("b: " + custom_results[b])
                # filter check
                else:
                    if a_next - a_now != b_next - b_now:
                        print(output_name + ": line number in full model doesn't equal line in modified model")
                        print("a: " + str(a_next - a_now) + " b: " + str(b_next - b_now))
                    
                    # store origin result
                    a_results = []
                    for a in range(a_now + 1, a_next):
                        tmp = origin_results[a].split("\t, ")
                        idx = tmp[0].split(": ")
                        res = []
                        res.append(int(idx[1][4:-1]))
                        res.append(idx[2])
                        for ch in tmp[1:-1]:
                            res.append(ch)
                        a_results.append(res)

                    right = 0
                    wrong = 0
                    # get customized results and check
                    for b in range(b_now + 1, b_next):
                        tmp = custom_results[b].split("\t, ")                    
                        idx = tmp[0].split(": ")
                        res = []
                        res.append(int(idx[1][4:-1]))
                        res.append(idx[2])
                        for ch in tmp[1:-1]:
                            res.append(ch)
                        flag = False
                        if data_type == "int8":
                            for a_res in a_results:
                                if a_res[0] == res[0]:
                                    flag = True
                                    sum = 0
                                    num = 0
                                    for a, b in zip(a_res[1:], res[1:]):
                                        if a[0] == " ":
                                            x = float(a[5:])
                                        else:
                                            x = float(a)
                                        if b[0] == " ":
                                            y = float(b[5:])
                                        else:
                                            y = float(b)
                                        sum = sum + abs((x - y) / x)
                                        num = num + 1
                                        # if abs((x - y) / x) > 0.1:
                                        #     flag = False
                                        #     print(x)
                                        #     print(y)
                                    sum = sum / num
                                    if sum > 0.05:
                                        flag = False
                                        print(res[0], sum)
                                    break
                        else:
                            for a_res in a_results:
                                if res == a_res:
                                    flag = True
                                    break
                        if not flag:
                            wrong = wrong + 1
                            # print("cannot find equal results of id " + str(res[0]) + " in origin results")
                        else:
                            right = right + 1
                    if wrong != 0:
                        print("right result: " + str(right))
                        print("wrong result: " + str(wrong))
                print("check output \033[35m" + output_name + "\033[0m done!")
            else:
                print("don't check output \033[35m" + output_name + "\033[0m")
            
            a_now = a_next
            b_now = b_next

if __name__ == "__main__":
    function_name = sys.argv[1]
    if function_name == "mark_nodes":
        print("start mark unnecessary nodes of cuda graph")
        mark_nodes(sys.argv[1:])
    elif function_name == "mark_nodes_model":
        print("start mark unnecessary nodes of cuda graph")
        mark_nodes_model(sys.argv[1:])
    elif function_name == "check_results":
        print("start check results")
        check_results(sys.argv[1:])
    else:
        print("wrong first argv, please use \033[35mmark_nodes\033[0m or \033[35mcheck_result\033[0m")