from xpilot_vision.tasks.base.config_parsers.parser import BaseConfigParser
from xpilot_lightning.machine_learning.tasks.builder import CONFIGPARSERS


@CONFIGPARSERS.register_module
class AP_LLDConfigParser(BaseConfigParser):
    def __init__(self, global_config, config):
        super().__init__(global_config=global_config, config=config)

    def _load_constants(self):
        # label related
        self.need_validate = True
        # Drawing
        self.colors = [(0, 0, 1), (0, 1, 0), (1, 0, 1),
                       (1, 0, 0), (0, 1, 1), (1, 1, 0),
                       (0, 0, 0.5), (0.5, 0, 0), (0, 0.5, 0),
                       (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5),
                       (0.25, 0, 0), (0, 0.25, 0), (0, 0, 0.25),
                       (0.25, 0.25, 0), (0, 0.25, 0.25), (0.25, 0, 0.25),
                       (1, 0.5, 0), (0.5, 1, 0), (0, 0.5, 1)]
        # Validation Use
        self.validation_xboard = {
            "scale": 8,
            "kpi_dict": {
                'f1': True,
                'iou': True,
                'line_type': True,
                'result_quality': True
            }
        }
