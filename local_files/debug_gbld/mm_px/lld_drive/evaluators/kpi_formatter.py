from xpilot_lightning.machine_learning.tasks.builder import KPI_FORMATTERS
from xpilot_lightning.machine_learning.tasks.base.kpi_formatters.kpi_formatter import (
    BaseKPIFormatter,
)
from xpilot_lightning import const


@KPI_FORMATTERS.register_module()
class LLD2LIGHTNINGKPIFormatter(BaseKPIFormatter):
    _task_name = "LLD"

    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)
        self.group_names = list(self.task_config.agg_map.keys())
        self.main_table_kpis_to_show = [
            # [feature name, kpi, table row name]
            ["lane_iou", "iou", "lane_iou"],
            ["exist", "accuracy", "lane_exist_accuracy"],
            ["line_type", "accuracy", "line_type_accuracy"],
            ["lane_iou_0-300", "iou", "lane_iou_0-300"],
            ["lane_iou_300-380", "iou", "lane_iou_300-380"],
            ["lane_iou_380-10000", "iou", "lane_iou_380-10000"],
            ["merge2left", "accuracy", "lane_merge2left_accuracy"],
            ["merge2right", "accuracy", "lane_merge2right_accuracy"],
            ["num_left_lanes", "accuracy", "num_left_lanes_accuracy"],
            ["num_right_lanes", "accuracy", "num_right_lanes_accuracy"],
            ["y_start", "rmse", "line_y_start_rmse"],
            ["y_end", "rmse", "line_y_end_rmse"],
        ]
        self.center_pr_table_kpis_to_show = [
            ["merge2left", "center_merge2left"],
            ["merge2right", "center_merge2right"],
            ["entirely_imagined_left_line", "center_entirely_imagined_left_line"],
            ["entirely_imagined_right_line", "center_entirely_imagined_right_line"],
            ["cross2noncross_left_line", "center_cross2noncross_left_line"],
            ["cross2noncross_right_line", "center_cross2noncross_right_line"],
            ["dashed_slow", "center_dashed_slow"],
            ["solid_slow", "center_solid_slow"],
            ["inferenced_line", "center_inferenced_line"],
            ["dash_solid", "center_dash_solid"],
            ["solid_dash", "center_solid_dash"],
            ["line_crossable", "center_line_crossable"],
            ["line_noncrossable", "center_line_non_crossable"],
        ]

    def display_cm(self, confusion_matrix):
        assert len(confusion_matrix) == len(confusion_matrix[0])
        assert len(confusion_matrix) == len(self.task_config.global_info)
        fmt_title = "|" + "{:^9}|" * (1 + len(confusion_matrix)) + "\n"
        fmt_row = "|{:^9}|" + "{:^9d}|" * len(confusion_matrix) + "\n"
        result_string = "Global case confusion matrix\n"
        result_string += fmt_title.format("t\\p", *self.task_config.global_info)
        for c, row in zip(self.task_config.global_info, confusion_matrix):
            result_string += fmt_row.format(c, *row)
        return result_string

    def format_kpi(self, kpi, dataset_identifier) -> str:
        cam_id = dataset_identifier.camera_id
        if cam_id:
            result_string = (
                "=" * 20 + "\t" + "LLD KPI benchmark {}".format(cam_id) + "\t" + "=" * 20 + "\n"
            )
        else:
            result_string = "=" * 20 + "\t" + "LLD KPI benchmark" + "\t" + "=" * 20 + "\n"
        for time in ["whole_day", "night", "day"]:
            timed_kpi = kpi[time]
            result_string += f"LLD KPI Table {time}\n"
            result_string += self.format_global_kpi_simple(timed_kpi["global"])
            fmt_title = "|{:^30}|" + "{:^10}|" * len(self.group_names) + "\n"
            fmt_result = "|{:^30}|" + "{:^10.5f}|" * len(self.group_names) + "\n"
            result_string += fmt_title.format("--", *self.group_names)
            for feature_name, kpi_name, row_name in self.main_table_kpis_to_show:
                data = [timed_kpi[group][feature_name][kpi_name] for group in self.group_names]
                result_string += fmt_result.format(row_name, *data)
            result_string += self.display_cm(timed_kpi["global"]["confusion_matrix"])
            center_lane_kpis = timed_kpi["center"]
            fmt_title = "|{:^36}|{:^5}|{:^5}|{:^5}|{:^7}|{:^7}|\n"
            fmt_result = "|{:^36}|{:^5.3f}|{:^5.3f}|{:^5.3f}|{:^7}|{:^7}|\n"
            result_string += fmt_title.format("--", "p", "r", "f1", "n_pred", "n_gt")
            for feature_name, row_name in self.center_pr_table_kpis_to_show:
                kpi_center = center_lane_kpis[feature_name]
                result_string += fmt_result.format(
                    row_name,
                    kpi_center["precision"],
                    kpi_center["recall"],
                    kpi_center["f1"],
                    kpi_center["num_pred"],
                    kpi_center["num_gt"]
                )
            result_string += self.format_f1table_kpi(center_lane_kpis["force_merge"])
            result_string += self.format_f1table_kpi(center_lane_kpis["cross2noncross"])
        result_string += "\n"
        if const.CORNERCASE_KPIS in kpi:
            result_string += self.format_corner_case_kpi(kpi[const.CORNERCASE_KPIS])
        return result_string

    def format_global_kpi_simple(self, kpi):
        result_string = "Global case KPI\n"
        result_string = "|{:^12}|{:^12}|\n".format("General_iou", "Global_acc", "n_gt")
        result_string += "|{:^12.4f}|{:^12.4f}|\n".format(kpi["general_iou"], kpi["accuracy"])
        return result_string

    def format_global_kpi(self, kpi):
        result_string = "Global case KPI\n"
        fmt_title = "|{:^7}|{:^5}|{:^5}|{:^5}|{:^6}|{:^6}|{:^6}|\n"
        fmt_result = "|{:^7}|{:^5.3f}|{:^5.3f}|{:^5.3f}|{:^6}|{:^6}|{:^6.3f}|\n"
        result_string += fmt_title.format("case","p","r","f1","n_gt","n_pred","iou")
        for global_case in self.task_config.global_info:
            kpi_class = kpi[global_case]
            result_string += fmt_result.format(
                global_case,
                kpi_class["precision"],
                kpi_class["recall"],
                kpi_class["f1"],
                kpi_class["num_gt"],
                kpi_class["num_pred"],
                kpi_class["general_iou"],
                )
        
        return result_string

    def format_f1table_kpi(self, kpis):
        result_string = ""
        fmt_title = "|{:^5}|{:^5}|{:^5}|{:^5}|{:^7}|\n"
        fmt_result = "|{:^5.2f}|{:^5.3f}|{:^5.3f}|{:^5.3f}|{:^7.5f}|\n"
        for feature_name, feature_kpi in kpis.items():
            result_string += f"{feature_name} F1 table\n"
            result_string += fmt_title.format("Thr","p", "r", "f1", "fp_rate")
            for threshold in feature_kpi:
                kpi = feature_kpi[threshold]
                result_string += fmt_result.format(
                    threshold,
                    kpi["precision"],
                    kpi["recall"],
                    kpi["f1"],
                    kpi["fp_rate"]
                )
        return result_string

    def format_corner_case_kpi(self, kpi):
        result_string = ""
        for case in kpi:
            if case in ["llde38fork", "llde38split", "llde38wide"]:
                result_string += self.format_corner_case_iou(case, kpi[case])
            elif case in ["llde38forcemergefp"]:
                result_string += self.format_corner_case_forcemergefp(case, kpi[case])
            elif case in ["llde38forcemergefn"]:
                result_string += self.format_corner_case_forcemergefn(case, kpi[case])
            elif case in ["llde38solidahead", "llde38fishbone", "llde38dashedsolid"]:
                result_string += self.format_corner_case_solidahead(case, kpi[case])
            elif case in ["llde38flickerfront"]:
                result_string += self.format_corner_case_general_iou(case, kpi[case])
        return result_string

    def format_corner_case_iou(self, case_name, kpi):
        result_string = "|" + " " * len(case_name) + "|{:^7}|{:^5}|\n".format("iou", "n_gt")
        result_string += "|" + case_name + "|{:^7.4f}|{:^5}|\n".format(kpi["iou"][0], kpi["n_gt"])
        result_string += "\n"
        return result_string

    def format_corner_case_forcemergefp(self, case_name, kpi):
        result_string = (
            "|" + " " * len(case_name)
            + "|{:^4}|{:^7}|{:^7}|{:^7}|{:^7}|\n".format(
                "thr", "leftFP", "rightFP", "n_left", "n_right"
            )
        )
        for thr, left, right in zip(kpi["threshold"], kpi["2leftFP_ratio"], kpi["2rightFP_ratio"]):
            result_string += (
                "|" + case_name
                + "|{:^4}|{:^7.4f}|{:^7.4f}|{:^7}|{:^7}|\n".format(
                    thr, left, right, kpi["n_negative_left"], kpi["n_negative_right"]
                )
            )
        result_string += "\n"
        return result_string

    def format_corner_case_forcemergefn(self, case_name, kpi):
        result_string = (
            "|" + " " * len(case_name)
            + "|{:^4}|{:^7}|{:^7}|{:^7}|{:^7}|\n".format(
                "thr", "leftTP", "rightTP", "n_left", "n_right"
            )
        )
        for thr, left, right in zip(kpi["threshold"], kpi["2leftTP_ratio"], kpi["2rightTP_ratio"]):
            result_string += (
                "|" + case_name
                + "|{:^4}|{:^7.4f}|{:^7.4f}|{:^7}|{:^7}|\n".format(
                    thr, left, right, kpi["n_positive_left"], kpi["n_positive_right"]
                )
            )
        result_string += "\n"
        return result_string

    def format_corner_case_solidahead(self, case_name, kpi):
        result_string = (
            "|" + " " * len(case_name)
            + "|{:^4}|{:^7}|{:^7}|{:^7}|{:^7}|\n".format(
                "thr", "TPrate", "FPrate", "n_pos", "n_neg"
            )
        )
        for thr, tp, fp in zip(kpi["threshold"], kpi["TP_ratio"], kpi["FP_ratio"]):
            result_string += (
                "|" + case_name
                + "|{:^4}|{:^7.4f}|{:^7.4f}|{:^7}|{:^7}|\n".format(
                    thr, tp, fp, kpi["n_positive"], kpi["n_negative"]
                )
            )
        result_string += "\n"
        return result_string

    def format_corner_case_linetype(self, case_name, kpi):
        result_string = "|" + " " * len(case_name) + "|{:^8}|{:^5}|\n".format("accuracy", "n_gt")
        result_string += "|" + case_name + "|{:^8.4f}|{:^5}|\n".format(kpi["accuracy"], kpi["n_gt"])
        result_string += "\n"
        return result_string

    def format_corner_case_general_iou(self, case_name, kpi):
        result_string = "|" + " " * len(case_name) + "|{:^7}|{:^9}|{:^5}|\n".format("iou", "global_acc", "n_gt")
        result_string += "|" + case_name + "|{:^7.4f}|{:^9.4f}|{:^5}\n".format(kpi["general_iou"][0], kpi["global_case_accuracy"][0], kpi["n_gt"])
        result_string += "\n"
        return result_string