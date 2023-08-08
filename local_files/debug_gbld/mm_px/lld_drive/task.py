from xpilot_lightning.machine_learning.tasks.base_linear.task import LinearBaseTask
from xpilot_lightning.machine_learning.tasks.builder import TASKS
from xpilot_vision.tasks.lld2lightning.attributes.builder import LLD2LIGHTNINGATTRIBUTES


@TASKS.register_module()
class LLD2LIGHTNINGTask(LinearBaseTask):
    def __init__(self, global_config, task_config, name, attribute_registry=LLD2LIGHTNINGATTRIBUTES):
        super().__init__(global_config, task_config, name, attribute_registry)
