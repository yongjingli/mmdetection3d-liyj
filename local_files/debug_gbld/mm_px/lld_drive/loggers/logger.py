from xpilot_lightning.machine_learning.tasks.base_linear.loggers.logger import LinearBaseLogger
from xpilot_lightning.machine_learning.tasks.builder import LOGGERS


@LOGGERS.register_module()
class LLD2LIGHTNINGLogger(LinearBaseLogger):
    ...
