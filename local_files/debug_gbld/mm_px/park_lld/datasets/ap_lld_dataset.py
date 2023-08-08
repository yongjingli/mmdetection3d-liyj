from xpilot_lightning.machine_learning.tasks.base.datasets.image_base_dataset import ImageBaseDataset
from xpilot_lightning.machine_learning.tasks.builder import DATASETS
from xpilot_lightning.data.xdata_helpers import XDatasetLoaderImageHelper, SQLFilterConverter

import logging
import numpy as np

@DATASETS.register_module
class AP_LLDDataset(ImageBaseDataset):
    def __init__(self, global_config,
                 task_config,
                 preprocess,
                 dataset_name,
                 camera_id,
                 status,
                 label_task_name,
                 phase,
                 **kwargs):
        super().__init__(
            global_config=global_config,
            task_config=task_config,
            preprocess=preprocess,
            dataset_name=dataset_name,
            camera_id=camera_id,
            status=status,
            label_task_name=label_task_name,
            phase=phase,
            **kwargs)
        self.preprocess.cam_id = self.camera_id
        # self.preprocess.reset_params()
        self.camera_id = camera_id

    def _label_preprocess_vision(self, data_blob):
        # label, mask, aux = self.preprocess.process(label_json, augment, self.label_task)
        # label, mask, aux = self.preprocess.process(data_blob["label"], {}, self.camera_id, data_blob['metadata'])
        label, mask, aux = self.preprocess.process(data_blob["label"], {}, self.camera_id,
                                                   data_blob['metadata']['uuid'])

        data_blob["label"] = np.array(label, dtype=np.float32)
        data_blob["mask"] = np.array(mask, dtype=np.float32)
        return data_blob
