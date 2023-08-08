from xpilot_lightning.machine_learning.tasks.base_linear.losses.loss import LinearBaseLoss
from xpilot_lightning.machine_learning.tasks.builder import LOGGERS


@LOGGERS.register_module()
class LLD2LIGHTNINGLoss(LinearBaseLoss):
    ...
