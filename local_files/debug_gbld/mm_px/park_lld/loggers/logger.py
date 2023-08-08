import torch

from xpilot_vision.tasks.base.loggers.logger import BaseLogger
from xpilot_lightning.machine_learning.tasks.builder import LOGGERS


@LOGGERS.register_module
class AP_LLDLogger(BaseLogger):
    def __init__(self, global_config, task_config, name="ap_lld"):
        super().__init__(global_config, task_config, name=name)

    def scalar_log(self, loss, iteration, phase, log_writer, kpi, sublosses):
        BaseLogger.scalar_log(self, loss, iteration, phase, log_writer, kpi, sublosses)
        # Log KPI
        if kpi:
            with torch.no_grad():
                name = self.name + "/" + phase + "/"
                _ = (
                    kpi.str_xboard_result_dict()
                )  # run this to generate result and store in kpi.kpi_table_list
                for k, v in kpi.kpi_table_list.items():
                    log_writer.add_scalar(name + "avg_" + k, v[0], iteration)
