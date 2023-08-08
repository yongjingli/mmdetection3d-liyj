from xpilot_lightning.machine_learning.tasks.base_linear.datasets.dataset import LinearBaseDataset
from xpilot_lightning.machine_learning.tasks.builder import DATASETS


@DATASETS.register_module()
class LLD2LIGHTNINGDataset(LinearBaseDataset):
    ...
