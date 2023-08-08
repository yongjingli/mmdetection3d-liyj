import sys

def find_next_output(file, output_name, start_line):
    i = start_line
    while i < len(file):
        if file[i].find(output_name) != -1:
            return i
        i = i + 1
    return -1

def find_elt_count(file, line):
    s = file[line].find('eltCount')
    if s == -1:
        return -1
    tmp = file[line][s:].split(" ")
    tmp = tmp[0].split(":")
    return int(tmp[1])

def csv_to_float_list(line):
    data_str = line.replace('\t','').replace('...','').replace(' ','').split(",")
    data = []
    for s in data_str:
        if s != '':
            try:
                a=float(s)
            except:
                print ('err', line, len(s), s)
            data.append(float(s))
    return data

def block_to_float_list(file, s_line, e_line):
    data_dict = {}
    for a in range(s_line, e_line):
        tmp = file[a].split(': ')
        idx = int(tmp[1].replace('\t','').replace(' ','')[4:-1])
        data = csv_to_float_list(tmp[2])
        data_dict[idx] = data
    return data_dict

def relative_deviation(a, b):
    if len(a)==0 or len(b)==0:
        return 0.0
    if len(a) != len(b):
        return Inf
    s = 0.0
    n = 0.0
    for i in range(len(a)):
        s = s + abs((a[i] - b[i]) / a[i])
        n = n + 1
    return s / n

def get_segment_dict(lines, output_names, batch_size):
    segment_dict = {}
    start_lines = []
    for i in range(0, len(output_names)):
        for batch in range(0, batch_size):
            s_line = find_next_output(lines, output_names[i][0] + ".batchId: " + str(batch), 0)
            # patch for old version on pdk5112 or pdk 526
            if s_line == -1:
                s_line = find_next_output(lines, output_names[i][0] + "." + str(batch), 0)
            if s_line != -1 and s_line != 0:
                segment_dict[output_names[i][0] + '.' + str(batch)] = [s_line, -1]
                start_lines.append(s_line)
    sorted_lines = sorted(start_lines)
    last_line = len(lines) - 1
    if len(lines[-1]) == 0:
        last_line = last_line - 1
    while lines[last_line].find(',') == -1 or lines[last_line].find('\t') == -1:
        last_line = last_line - 1
    last_line = last_line + 1
    if not last_line > sorted_lines[-1]:
        last_line = sorted_lines[-1] # + 1
    sorted_lines.append(last_line)
    for i in range(0, len(output_names)):
        for batch in range(0, batch_size):
            st_line = segment_dict[output_names[i][0] + '.' + str(batch)][0]
            st_idx = sorted_lines.index(st_line, 0, len(sorted_lines))
            if st_idx != -1:
                segment_dict[output_names[i][0] + '.' + str(batch)][1] = sorted_lines[st_idx+1]
    return segment_dict

def check_results(argv):
    relative_deviation_thres = 1e-6
    batch_size  = int(argv[1])
    compare_file1 = argv[2]
    compare_file2 = argv[3]
    if len(argv) >= 5:
        relative_deviation_thres  = float(argv[4])
    print ('Relative Deviation Thres: %f' % (relative_deviation_thres))
    # read result of origin outputs
    with open(compare_file1, 'r') as f:
        origin_results = [i[:-1] for i in f.readlines()]

    # collect output
    output_flag = []
    for i in origin_results:
        if i.find("Binding[") != -1:
            tmp = i.split(" ")
            tmp = tmp[1].split(":")
            if tmp[0].find("In") == -1:
                output_flag.append([tmp[1], 1])
    # patch for dla
    if len(output_flag) == 0:
        output_flag_keys = {}
        for i in origin_results:
            if i.find("Buf[") != -1:
                tmp = i.split('\t')
                tmp = tmp[1].split(':')
                if tmp[0].find("In") == -1:
                    s = tmp[1].split(' ')[0]
                    if not output_flag_keys.get(s):
                        output_flag_keys[s] = 1
        for key, val in output_flag_keys.items():
            output_flag.append([key, val]) 

    # read result of modified outputs
    with open(compare_file2, 'r') as f:
        custom_results = [i[:-1] for i in f.readlines()]

    # from now a means origin result, b means customized result
    a_segment_dict = get_segment_dict(origin_results, output_flag, batch_size)
    b_segment_dict = get_segment_dict(custom_results, output_flag, batch_size)

    for i in range(0, len(output_flag)):
        for batch in range(0, batch_size):
            output_name = output_flag[i][0] + "." + str(batch)
            # find current output result range of line
            a_now, a_next = a_segment_dict[output_name]
            b_now, b_next = b_segment_dict[output_name]

            # check result
            if output_flag[i][1] == 1:            
                print("checking output \033[35m" + output_name + "\033[0m")
                # directly check
                if output_flag[i][0].find("filter") == -1:
                    if a_next - a_now != b_next - b_now:
                        print(output_name + ": line number in Text A doesn't equal to line number in Text B")
                    # check elt Count
                    eltcount_eq = False
                    if find_elt_count(origin_results, a_now) == find_elt_count(custom_results, b_now):
                        eltcount_eq = True
                    # check data
                    while a_now < a_next:
                        if origin_results[a_now].find(',') != -1 and origin_results[a_now].find('\t') != -1:
                            break
                        a_now = a_now + 1
                    while b_now < b_next:
                        if custom_results[b_now].find(',') != -1 and custom_results[b_now].find('\t') != -1:
                            break
                        b_now = b_now + 1
                    rel_dev = 0.0
                    check_res = False
                    a_data = []
                    b_data = []
                    if a_next > a_now and b_next > b_now:
                        a_data = csv_to_float_list(origin_results[a_now])
                        b_data = csv_to_float_list(custom_results[b_now])
                        if eltcount_eq and len(a_data)==len(b_data):
                            rel_dev = relative_deviation(a_data, b_data)
                        else:
                            rel_dev = float('inf')
                    elif a_next == a_now and b_next == b_now:
                        rel_dev = 0.0
                    else:
                        rel_dev = float('inf')
                    if rel_dev < relative_deviation_thres:
                        check_res = True
                    if not check_res:
                        if a_next > a_now:
                            print("\033[31m" + output_name + ' divergence:\033[0m Relative Deviation: ' + str(rel_dev) + ' A[0]:' + str(a_data[0]) + ' B[0]:' + str(b_data[0]))
                        else:
                            print("\033[31mText A and text B have different size at line " + output_name + '\033[0m')
                    else:
                        print("checked \033[35m" + output_name + "\033[0m, no mismatch")  
                # filter check
                else:
                    if a_next - a_now != b_next - b_now:
                        print(output_name + ": line number in text A doesn't equal line in text B")
                    
                    # check elt Count
                    # eltcount_eq = False
                    # if find_elt_count(origin_results, a_now) == find_elt_count(custom_results, b_now):
                    #     eltcount_eq = True
                    while a_now < a_next:
                        if origin_results[a_now].find(',') != -1 and origin_results[a_now].find('\t') != -1:
                            break
                        a_now = a_now + 1
                    while b_now < b_next:
                        if custom_results[b_now].find(',') != -1 and custom_results[b_now].find('\t') != -1:
                            break
                        b_now = b_now + 1
                        
                    right = 0
                    wrong = 0
                    wrong_dict = {}                        
                    if a_next > a_now and b_next > b_now:
                        # check data
                        a_data_dict = block_to_float_list(origin_results, a_now, a_next)
                        b_data_dict = block_to_float_list(custom_results, b_now, b_next)
                        for a_key, a_value in a_data_dict.items():
                            b_value = b_data_dict.get(a_key)
                            if b_value is None:
                                wrong = wrong + 1
                                continue
                            rel_dev = relative_deviation(a_value, b_value)
                            if len(a_value)==len(b_value) and rel_dev < relative_deviation_thres:
                                right = right + 1
                            else:
                                wrong = wrong + 1
                                wrong_dict[a_key] = [rel_dev, a_value[0], b_value[0]]
                    elif a_next == a_now and b_next == b_now:
                        wrong = 0
                    else:
                        wrong = -1
                    
                    if wrong != 0:
                        if a_next > a_now:
                            print("\033[31m" + output_name + " divergence with " + str(wrong) + " indexes:\033[0m")
                            for key, val in wrong_dict.items():
                                print('\033[33mIdx:' + str(key) + '\033[0m Relative Deviation: ' + str(val[0]) + ' A[0]:' + str(val[1]) + ' B[0]:'+ str(val[2]))
                        else:
                            print("\033[31mText A and text B have different size at line " + output_name + '\033[0m')

                    else :
                        print("checked \033[35m" + str(a_next - a_now - 1) + "\033[0m elements, no mismatch")
                print("check output \033[35m" + output_name + "\033[0m done!")
            else:
                print("don't check output \033[31m" + output_name + "\033[0m")






if __name__ == "__main__":
    print("start check results")
    check_results(sys.argv)