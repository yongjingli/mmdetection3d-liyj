# 在加载官方权重的时候,出现权重shape不匹配的情况
# size mismatch for pts_middle_encoder.conv_input.0.weight: copying a param with shape
def convert_bev_checkpoints():
    import torch
    path = '/home/dell/liyongjing/programs/mmdetection3d-liyj/checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
    s_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_convert.pth"
    model = torch.load(path)

    for key in model['state_dict'].keys():
        if (key == 'pts_middle_encoder.conv_input.0.weight') or (key == 'pts_middle_encoder.conv_out.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight') or (('pts_middle_encoder.encoder_layers.encoder_layer' in key) and ('conv' in key)):
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 0, 1)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 1, 2)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 2, 3)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 3, 4)

    torch.save(model, s_path)

def convert_mvxnet_checkpoints():
    import torch
    path = '/home/dell/liyongjing/programs/mmdetection3d-liyj/checkpoints/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
    s_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/checkpoints/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a_convert.pth"
    model = torch.load(path)

    for key in model['state_dict'].keys():
        if (key == 'pts_middle_encoder.encoder_layers.encoder_layer1.0.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.0.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.1.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.0.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.1.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer4.0.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer4.1.0.weight') or \
                (key == 'pts_middle_encoder.encoder_layers.encoder_layer4.2.0.weight') or \
                (key == 'pts_middle_encoder.conv_input.0.weight') or \
                (key == 'pts_middle_encoder.conv_out.0.weight') or \
            (('pts_middle_encoder.encoder_layers.encoder_layer' in key) and ('conv' in key)):
            print(key)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 0, 1)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 1, 2)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 2, 3)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 3, 4)

    torch.save(model, s_path)

if __name__ == "__main__":
    print("Start")
    # convert_bev_checkpoints()
    convert_mvxnet_checkpoints()
    print("End")