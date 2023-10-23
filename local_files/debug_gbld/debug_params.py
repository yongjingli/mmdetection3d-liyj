import torch
from torchvision.models import resnet50, resnet101
from thop import profile
from thop import clever_format     
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.registry import MODELS


def cal_resnet50_params():
    # model = resnet50()
    model = resnet101()
    model = model.float()
    # input = torch.randn(1, 3, 608, 960)
    input = torch.randn(6, 3, 736, 1280)
    # input = input.float16()
    flops, params = profile(model, inputs=(input, ))

    pflops, params = clever_format([flops, params], "%.3f")
    print("flops:", pflops)

    # 这里拿到的是参数的个数
    print("params:", params)

    
def cal_multi_head_att():
    class Model(nn.Module):
        def __init__(self, embed_dim=768, head_num=12, dropout=0.1):
            super(Model, self).__init__()
            self.mha = torch.nn.MultiheadAttention(embed_dim, head_num, dropout, batch_first=True)

        def forward(self, query, key, value):
            out = self.mha(query, key, value)
            return out

    # input = torch.randn(1, 3, 608, 960)
    model = Model(embed_dim=300, head_num=12,)


    query = torch.rand(10, 64, 300)
    # batch_size 为 64，有 10 个词，每个词的 Key 向量是 300 维
    key = torch.rand(10, 64, 300)
    # batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
    value = torch.rand(10, 64, 300)

    # model 是没有参数量的，只是提供attention的计算方式
    print(query.shape, key.shape, value.shape, )
    attn_output, attn_output_weights = model(query, key, value)

    flops, params = profile(model, inputs=(query, key, value))
    print(attn_output.shape, attn_output_weights.shape)
    pflops, params = clever_format([flops, params], "%.3f")
    print("flops:", pflops)

    # 这里拿到的是参数的个数
    print("params:", params)


def cal_self_model_params():
    class SelfModel(nn.Module):
        def __init__(self, embed_dim=768, head_num=12, dropout=0.1):
            super(SelfModel, self).__init__()
            self.mha = torch.nn.MultiheadAttention(embed_dim, head_num, dropout, batch_first=True)

            backbone_cfg = {'type': 'mmdet.ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1,
             'norm_cfg': {'type': 'BN', 'requires_grad': False}, 'norm_eval': True, 'style': 'caffe',
             'init_cfg': {'type': 'Pretrained', 'checkpoint': 'open-mmlab://detectron2/resnet50_caffe'}, 'dcn': None,
             'stage_with_dcn': (False, False, True, True)}
            neck_cfg = {'type': 'mmdet.FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'start_level': 0, 'num_outs': 4}


            # regnet
            # backbone_cfg = dict(
            #                     # _delete_=True,
            #                     type='RegNet',
            #                     arch='regnetx_3.2gf',
            #                     out_indices=(0, 1, 2, 3),
            #                     frozen_stages=1,
            #                     norm_cfg=dict(type='BN', requires_grad=True),
            #                     norm_eval=True,
            #                     style='pytorch',
            #                     init_cfg=dict(
            #                         type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf'))
            # neck_cfg = dict(
            #         type='FPN',
            #         in_channels=[96, 192, 432, 1008],
            #         out_channels=256,
            #         num_outs=5)

            self.backbone = MODELS.build(backbone_cfg)
            self.neck = MODELS.build(neck_cfg)

            # bev query,只是编码了，不参与参数量
            self.bev_h = 150
            self.bev_w = 150
            self.embed_dims = 256
            self.num_query = 900
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)


            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            self.code_weights = nn.Parameter(torch.tensor(
                self.code_weights, requires_grad=False), requires_grad=False)

            # 都添加一个卷积看看是否会增加参数量
            self.m = nn.Conv2d(3, 256, (3, 3), stride=(1, 1))

            #
            #
            cls_branch = []
            self.num_reg_fcs = 2
            self.cls_out_channels = 10
            self.code_size = 10
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
            fc_cls = nn.Sequential(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

            num_layers = 3
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_layers)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])



        # def forward(self, query, key, value):
        #     out = self.mha(query, key, value)
        #     return out

        def forward(self, batch_inputs):
            x = self.backbone(batch_inputs)
            # return x
            x = self.neck(x)

            a = self.bev_embedding
            b = self.query_embedding

            c = self.code_weights

            # 不运行的话参数量就不会增加
            d = self.m(batch_inputs)
            # e = self.cls_branches[0](d)
            # print(e.shape)

            return x, a, b, c

    img_input = torch.randn(1, 3, 608, 960)


    model = SelfModel(embed_dim=300, head_num=12,)
    query = torch.rand(10, 64, 300)
    # batch_size 为 64，有 10 个词，每个词的 Key 向量是 300 维
    key = torch.rand(10, 64, 300)
    # batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
    value = torch.rand(10, 64, 300)

    # model 是没有参数量的，只是提供attention的计算方式
    # att forward
    # print(query.shape, key.shape, value.shape, )
    # attn_output, attn_output_weights = model(query, key, value)
    # flops, params = profile(model, inputs=(query, key, value))
    # print(attn_output.shape, attn_output_weights.shape)

    # img forward
    img_output = model(img_input)
    flops, params = profile(model, inputs=(img_input, ))



    # flops, params = clever_format([flops, params], "%.3f")
    print("flops:", flops)

    # 这里拿到的是参数的个数
    print("params:", params)


def debug_Linear():
    class Model(nn.Module):
        def __init__(self, embed_dim=768, head_num=12, dropout=0.1):
            super(Model, self).__init__()
            self.connected_layer = Linear(in_features=64 * 64 * 3, out_features=100)

        def forward(self, input):
            out = self.connected_layer(input)
            return out


    connected_layer = nn.Linear(in_features=64 * 64 * 3, out_features=1)
    # connected_layer = Linear(in_features=64 * 64 * 3, out_features=1)
    #
    # # 假定输入的图像形状为[64,64,3]
    input = torch.randn(1, 64, 64, 3)
    #
    # # 将四维张量转换为二维张量之后，才能作为全连接层的输入
    input = input.view(1, 64 * 64 * 3)
    print(input.shape)
    # output = connected_layer(input)  # 调用全连接层
    # print(output.shape)

    # img forward
    model = Model()
    img_output = model(input)
    print(img_output.shape)
    flops, params = profile(model, inputs=(input, ))
    # flops, params = clever_format([flops, params], "%.3f")
    print("flops:", flops)
    # 这里拿到的是参数的个数
    print("params:", params)


def debug_linear_params():
    class Model(nn.Module):
        def __init__(self, embed_dim=768, head_num=12, dropout=0.1):
            super(Model, self).__init__()
            self.connected_layer = nn.Linear(in_features=3, out_features=100, bias=False)
            self.connected_layer2 = nn.Linear(in_features=3, out_features=100, bias=True)
            # self.connected_layer = Linear(in_features=3, out_features=100)

        def forward(self, input):
            out = self.connected_layer(input)
            # out2 = self.connected_layer2(input)
            return out

    model = Model()
    # input = torch.randn(1, 64, 64, 3)
    input = torch.randn(1, 64, 3)
    out = model(input)
    print(out.shape)

    flops, params = profile(model, inputs=(input, ))
    # flops, params = clever_format([flops, params], "%.3f")
    print("flops:", flops)
    # 这里拿到的是参数的个数
    print("params:", params)



def debug_linear_params_mmcv():
    class Model(nn.Module):
        def __init__(self, embed_dim=768, head_num=12, dropout=0.1):
            super(Model, self).__init__()
            # self.connected_layer = nn.Linear(in_features=3, out_features=100, bias=False)
            # self.connected_layer = nn.Linear(in_features=3, out_features=100, bias=True)
            self.connected_layer = Linear(in_features=3, out_features=100)
            self.connected_layer2 = Linear(in_features=3, out_features=100)

        def forward(self, input, meta=None):
            out = self.connected_layer(input)
            # out2 = self.connected_layer2(input)
            return out

    model = Model()
    # # input = torch.randn(1, 64, 64, 3)
    input = torch.randn(1, 64, 3)
    # out = model(input)
    # print(out.shape)
    #
    # flops, params = profile(model, inputs=(input, ))
    # # flops, params = clever_format([flops, params], "%.3f")
    # print("flops:", flops)
    # # 这里拿到的是参数的个数
    # print("params:", params)
    from mmcv.cnn.utils.flops_counter import add_flops_counting_methods
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()

    # input
    input_shape = (1, 100, 3)
    try:
        batch = torch.ones(()).new_empty(
            (1, *input_shape),
            dtype=next(flops_model.parameters()).dtype,
            device=next(flops_model.parameters()).device)
    except StopIteration:
        # Avoid StopIteration for models which have no parameters,
        # like `nn.Relu()`, `nn.AvgPool2d`, etc.
        batch = torch.ones(()).new_empty((1, *input_shape))

    meta = "hahahah"
    _ = flops_model(batch, meta)
    flops_count, params_count = flops_model.compute_average_flops_cost()
    flops_model.stop_flops_count()

    flops, params = flops_count, params_count
    print(f'Flops: {flops}\nParams: {params}')


if __name__ == "__main__":
    print("STart")
    # cal_resnet50_params()
    # cal_multi_head_att()
    # cal_self_model_params()
    # debug_Linear()

    # 采用mm里的Linear算出来的计算量和参数量均为0
    # 采用pytorch里的nn.Linear则可以算出来, 如果不算bias，参数量为in_channel * out_channel,
    # 如果不算bias，参数量为in_channel * out_channel + out_channel
    # debug_linear_params()

    # 采用mmcv的方式计算
    # debug_linear_params_mmcv()
    print("End")
