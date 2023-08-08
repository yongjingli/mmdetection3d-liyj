import sys

def main():

    target=open(sys.argv[1]+"/deploy_"+sys.argv[2]+".yaml", "w+")

    target.write("general:\n")
    target.write("    trt_file_path : /xpilot/bin/perception/xp_model_qnx_fp16_"+sys.argv[2]+".trt\n")
    target.write("    lib_file_path : /xpilot/bin/perception/libonnxtrt.so\n")
    target.write("    anchor_cell_file_path : /xpilot/bin/perception/anchor_cell_"+sys.argv[2]+".bin\n")
    target.write("    exttrt_file_name : LaneMOD\n")

    try:
        lld=open("lld_params.yaml", "r")
        target.write("LLD:\n")
        
        line=[line.rstrip('\n') for line in lld]
        for l in line :
            target.write('    '+l+'\n')
    except:
        print("lld_params.yaml not found")


    try:
        mod=open("mod_params.yaml", "r")
        target.write("MOD:\n")

        line=[line.rstrip('\n') for line in mod]
        for l in line :
            target.write('    '+l+'\n')
    except:
        print("mod_params.yaml not found")


    try:
        ds=open("ds_params.yaml", "r")
        target.write("DS:\n")
        line=[line.rstrip('\n') for line in ds]
        for l in line :
            target.write('    '+l+'\n')
    except:
        print("ds_params.yaml not found")
    

    try:
        ihb=open("ihb_params.yaml", "r")
        target.write("IHB:\n")
        target.write("    enable: true\n")
        line=[line.rstrip('\n') for line in ihb]
        for l in line :
            target.write('    '+l+'\n')
    except:
        target.write("IHB:\n")
        target.write("    enable: false\n")
        print("ihb_params.yaml not found")


if __name__=='__main__':
    main()