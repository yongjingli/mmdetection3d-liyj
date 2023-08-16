# 对于gbld
#    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        batch_imgs = batch_inputs_dict['imgs']
        # get flops
        # batch_imgs = batch_inputs_dict    # 修改输入

config=./projects/GrasslandBoundaryLine2D/work_dirs/gbld_debug_no_dcn/gbld_debug_config_no_dcn.py
pyton tools/analysis_tools/get_flops.py $config \
--modality image \
--shape 304 480