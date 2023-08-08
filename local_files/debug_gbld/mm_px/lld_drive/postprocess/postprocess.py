from xpilot_lightning.machine_learning.tasks.base_linear.postprocesses.postprocess import (
    LinearBasePostProcess,
)
from xpilot_lightning.machine_learning.tasks.builder import POSTPROCESSES


@POSTPROCESSES.register_module()
class LLD2LIGHTNINGPostProcessing(LinearBasePostProcess):
    ...
