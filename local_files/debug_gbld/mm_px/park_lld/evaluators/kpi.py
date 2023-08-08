import json
from collections import defaultdict

from xpilot_vision.tasks.ap_lld.evaluators.ap_lld_evaluator import AP_LLDEvaluator
from xpilot_lightning.machine_learning.tasks.builder import KPIS


@KPIS.register_module
class AP_LLDKpi(object):
    def __init__(self, global_config=None, task_config=None):
        self.kpi_table_list = defaultdict(list)
        self.global_config = global_config
        self.task_config = task_config
        if task_config is not None:
            self.evaluator = AP_LLDEvaluator(global_config, task_config)

    def __add__(self, other):
        if self.task_config is not None:
            return other

    def compute_xboard(self, y_hat, y, uuid, cam_id=None):
        for idx, (y_hat_dict, label_dict) in enumerate(zip(y_hat, y)):
            self.evaluator.process_once(y_hat_dict, label_dict, uuid[idx])

    def __str__(self):
        return self.str_xboard_result_dict()

    # convert xboard result dictionary to previous kpi result format
    def str_xboard_result_dict(self):
        xboard_result_dict = self.evaluator.eval_metrics()
        with open("pr_table.txt", "w") as outfile:
            json.dump(xboard_result_dict["line_point_pr_threshold"], outfile)

        self.kpi_table_list = defaultdict(list)

        precision = list(xboard_result_dict['line_exist_pr_threshold'].values())[0]['precision']
        recall = list(xboard_result_dict['line_exist_pr_threshold'].values())[0]['recall']
        if precision == 'NaN' or recall == 'NaN' or precision + recall == 0:
            line_f1 = 0.0
        else:
            line_f1 = 2 * precision * recall / (precision + recall)
        self.kpi_table_list['line_f1'].append(line_f1)

        precision = list(xboard_result_dict['line_point_pr_threshold'].values())[0]['precision']
        recall = list(xboard_result_dict['line_point_pr_threshold'].values())[0]['recall']
        if precision == 'NaN' or recall == 'NaN' or precision + recall == 0:
            grid_f1 = 0.0
        else:
            grid_f1 = 2 * precision * recall / (precision + recall)
        self.kpi_table_list['grid_f1'].append(grid_f1)

        precision = list(xboard_result_dict['line_point_pr_distance'].values())[-1]['precision']
        recall = list(xboard_result_dict['line_point_pr_distance'].values())[-1]['recall']
        if precision == 'NaN' or recall == 'NaN' or precision + recall == 0:
            near_grid_f1 = 0.0
        else:
            near_grid_f1 = 2 * precision * recall / (precision + recall)
        self.kpi_table_list['0~15m_grid_f1'].append(near_grid_f1)

        printed_result = self.benchmark()
        return printed_result

    def benchmark(self):
        printed_result = "====================AP_LLD KPI Table===================\n"
        xboard_result_dict = self.evaluator.eval_metrics()

        pr_table = xboard_result_dict["line_exist_pr_threshold"]
        printed_result += "==================Line exist PR table==================\n"
        printed_result += "|threshold|    F1    |  recall  | precision| total gt |total pred|\n"
        for thresh, values in pr_table.items():
            recall, precision, gt, pred = (
                values["recall"],
                values["precision"],
                values["gt_num"],
                values["pred_num"],
            )
            if recall == "NaN" or precision == "NaN" or precision + recall == 0:
                printed_result += "|  %.2f   |    NaN   |    NaN   |    NaN   |  %6d  |  %6d  |\n" % (
                    thresh,
                    gt,
                    pred,
                )
            else:
                f1 = 2 * precision * recall / (precision + recall)
                printed_result += "|  %.2f   |  %.4f  |  %.4f  |  %.4f  |  %6d  |  %6d  |\n" % (
                    thresh,
                    f1,
                    recall,
                    precision,
                    gt,
                    pred,
                )
        pr_table = xboard_result_dict["line_point_pr_threshold"]
        printed_result += "\n==================Line point PR table==================\n"
        printed_result += "|threshold|    F1    |  recall  | precision| total gt |total pred|\n"
        for thresh, values in pr_table.items():
            recall, precision, gt, pred = (
                values["recall"],
                values["precision"],
                values["gt_num"],
                values["pred_num"],
            )
            if recall == "NaN" or precision == "NaN" or precision + recall == 0:
                printed_result += "|  %.2f   |    NaN   |    NaN   |    NaN   |  %6d  |  %6d  |\n" % (
                    thresh,
                    gt,
                    pred,
                )
            else:
                f1 = 2 * precision * recall / (precision + recall)
                printed_result += "|  %.2f   |  %.4f  |  %.4f  |  %.4f  |  %6d  |  %6d  |\n" % (
                    thresh,
                    f1,
                    recall,
                    precision,
                    gt,
                    pred,
                )
        pr_table = xboard_result_dict["line_point_pr_distance"]
        printed_result += "\n=================Line distance PR table================\n"
        printed_result += "|threshold|    F1    |  recall  | precision| total gt |total pred|\n"
        for dist, values in pr_table.items():
            recall, precision, gt, pred = (
                values["recall"],
                values["precision"],
                values["gt_num"],
                values["pred_num"],
            )
            if recall == "NaN" or precision == "NaN" or precision + recall == 0:
                printed_result += "|  %.2f   |    NaN   |    NaN   |    NaN   |  %6d  |  %6d  |\n" % (
                    thresh,
                    gt,
                    pred,
                )
            else:
                f1 = 2 * precision * recall / (precision + recall)
                printed_result += "| %7s |  %.4f  |  %.4f  |  %.4f  |  %6d  |  %6d  |\n" % (
                    dist,
                    f1,
                    recall,
                    precision,
                    gt,
                    pred,
                )
        pr_table = xboard_result_dict["line_point_position_error"]
        printed_result += "\n====================Line point error===================\n"
        values = tuple(pr_table.values())
        printed_result += "|distance|    0    |    1    |    2    |    3    |    4    |    5    |\n"
        printed_result += "| number | %7d | %7d | %7d | %7d | %7d | %7d |\n" % values[0:6]
        printed_result += "|distance|    6    |    7    |    8    |    9    |   >=10  |\n"
        printed_result += "| number | %7d | %7d | %7d | %7d | %7d |\n\n" % values[6:]

        # Arrow
        printed_result += "=" * 20 + "\t" + "Location KPI" + "\t" + "=" * 20 + "\n"
        fmt_title = "|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|\n"
        fmt_result = "|{:^15.3}|{:^15.3}|{:^15.3}|{:^15}|{:^15}|\n\n"
        printed_result += fmt_title.format('F1', 'Precision', 'Recall', 'pred_num', 'gt_num')
        recall, precision, f1, pred, gt = xboard_result_dict['mark_location']
        printed_result += fmt_result.format(
            f1, precision, recall, pred, gt
        )
        if recall == 0:
            return printed_result

        printed_result += "=" * 20 + "\t" + "Type KPI" + "\t" + "=" * 20 + "\n"
        fmt_title = "|{:^15}|{:^15}|{:^15}|\n"
        fmt_result = "|{:^15.3}|{:^15}|{:^15}|\n\n"
        printed_result += fmt_title.format('Accuracy', 'tp_num', 'gt_num')
        accuracy, tp, gt = xboard_result_dict['mark_type']
        printed_result += fmt_result.format(
            accuracy, tp, gt
        )

        printed_result += "=" * 20 + "\t" + "Vertex KPI" + "\t" + "=" * 20 + "\n"
        fmt_title = "|{:^15}|{:^15}|{:^15}|\n"
        fmt_result = "|{:^15.3}|{:^15}|{:^15}|\n\n"
        printed_result += fmt_title.format('Accuracy', 'tp_num', 'gt_num')
        accuracy, tp, gt = xboard_result_dict['mark_vertex_type']
        printed_result += fmt_result.format(
            accuracy, tp, gt
        )

        return printed_result
