from xpilot_lightning.machine_learning.tasks.base_linear.heads.head import LinearBaseHead
from xpilot_lightning.machine_learning.tasks.builder import HEADS
from xpilot_vision.tasks.lld2lightning.losses.loss import LLD2LIGHTNINGLoss


@HEADS.register_module()
class LLD2LIGHTNINGHead(LinearBaseHead):
    def __init__(
        self, global_config, task_config, loss_func=LLD2LIGHTNINGLoss, freeze_module: bool = False
    ):
        super().__init__(global_config, task_config, loss_func, freeze_module)
