from xpilot_vision.tasks.ap_lld.losses.loss import AP_LLDLoss
from xpilot_vision.tasks.base.heads.head import BaseHead
from xpilot_lightning.machine_learning.tasks.builder import HEADS


@HEADS.register_module
class AP_LLDHead(BaseHead):
    def __init__(self, global_config, task_config, loss_fun=AP_LLDLoss):
        BaseHead.__init__(self, global_config, task_config, loss_fun)

        from xpilot_vision.models.heads.common_head import AP_LaneHead

        self.head = AP_LaneHead(
            global_config,
            task_config.skip_layer,
            task_config.global_layer,
            task_config.scale_factor,
            task_config.crop_size,
            task_config.buffer_layer,
        )

    def forward(
        self, feats, cam_id, is_onnx=False, onnx_with_anomaly=True, return_loss=False, **kwargs
    ):
        y_hats = self.head(feats, cam_id, is_onnx)
        if not return_loss:
            return y_hats
        labels = kwargs["labels"]
        masks = kwargs["masks"]
        loss, loss_info, batch_loss = self.loss(y_hats, labels, mask=masks)
        return {"outputs": y_hats, "loss": loss, "loss_info": loss_info, "batch_loss": batch_loss}
