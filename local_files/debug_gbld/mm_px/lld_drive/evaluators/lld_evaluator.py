import logging
from copy import copy
import pandas as pd
import numpy as np
from xpilot_lightning.machine_learning.tasks.builder import EVALUATORS
from xpilot_lightning.machine_learning.tasks.base.evaluators.evaluator import BaseEvaluator
from xpilot_lightning.machine_learning.tasks.base.evaluators.xpilot_kpi import XpilotLightningKPI
from xpilot_vision.tasks.lld2lightning.evaluators.lld_utils import (
    cal_iou_v2,
    calculate_accuracy,
    calculate_f1,
    calculate_rmse,
    calculate_f1_with_threshold,
)
from xpilot_lightning import const
from xpilot_lightning.utilities.evaluation_helpers.compute_confusion_matrix import (
    compute_confusion_matrix,
)


@EVALUATORS.register_module
class LLD2LIGHTNINGEvaluator(BaseEvaluator):
    def __init__(self, global_config, task_config, scale=2):
        """All evaluation related vars are dict[filename] = values
        All evaluation related to positions are
        dict[filename] = {
            "center": [values, num_count],
            "left": [values, num_count],
            "right": [values, num_count]
        }
        only few of vars are still in list, will consider whether convert them or not
        use dict to store value make it easier to track the detail for each image and future develop
        Args:
            config: lld config
            print_to_terminal: if print evaluation results to terminal
        """
        self.global_config = global_config
        self.task_config = task_config
        self.scale = task_config.scale if hasattr(task_config, "scale") else scale
        self.pos2id = self.task_config.pos2id
        self.id2pos = {i: pos for pos, i in self.pos2id.items()}
        self.num_lanes = len(self.pos2id)
        self.day_time = self.task_config.validation_xboard["day_time"]
        self.iou_levels = {
            f"{level[0]}-{level[1]}": level
            for level in self.task_config.validation_xboard["iou_levels"]
        }
        self.lane_pred_gt = {pos: [] for pos in self.pos2id}
        self.global_pred_gt = []
        self.upload_to_eval2 = getattr(self.global_config, "upload_to_eval2", False)
        self.columns = self.build_column_names()
        self.corner_case_thresholds = self.task_config.corner_case_thresholds

    def build_column_names(self):
        columns = ["uuid", "tagger", "beijing_time", "time", "lane_iou"]
        for iou_level in self.iou_levels:
            columns.append("lane_iou_" + iou_level)
        for feature_name in self.task_config.bool_features_lane:
            columns.append(feature_name + "_pred")
            columns.append(feature_name + "_true")
            columns.append(feature_name + "_prob")
        for feature_name in self.task_config.reg_features_lane:
            columns.append(feature_name + "_pred")
            columns.append(feature_name + "_true")
        # flatten line level feature
        for line_name in ["left_line", "right_line"]:
            for feature_name in self.task_config.bool_features_line:
                columns.append(feature_name + "_" + line_name + "_pred")
                columns.append(feature_name + "_" + line_name + "_true")
                columns.append(feature_name + "_" + line_name + "_prob")
            for feature_name in ["line_type"] + self.task_config.other_features_line:
                columns.append(feature_name + "_" + line_name + "_pred")
                columns.append(feature_name + "_" + line_name + "_true")
        return columns

    def calculate_general_iou(self, pred, true):
        pred_global = pred["global"][0]
        true_global = true["global"][0]
        if pred_global == "no_lane" and true_global != "no_lane":
            return 0
        if true_global == "no_lane":
            return np.nan
        ego_idxs = self.task_config.ego_lanes[true_global]
        pred_idxs = self.task_config.activated_lanes[pred_global]
        best_ious = []
        for ego_idx in ego_idxs:
            ego_true = true["lanes"][ego_idx]
            all_ious = [] 
            for pred_idx in pred_idxs:
                lane_pred = pred["lanes"][pred_idx]
                _, _, iou = cal_iou_v2(self.global_config, lane_pred, ego_true, scale=self.scale)
                if iou in ["error", "invalid"]:
                    iou = 0.0
                all_ious.append(iou)
            best_ious.append(max(all_ious))
        return sum(best_ious) / len(best_ious)

    def process(self, pred: dict, true: dict, metadata: dict):
        # detach according to global case
        meta_tag = self.generate_meta_tag(metadata)
        global_pred_gt = {
            "global_pred": pred["global"][0],
            "global_true": true["global"][0],
        }
        global_pred_gt.update(meta_tag)
        for lane_id in range(self.num_lanes):
            pred_lane = pred["lanes"][lane_id]
            true_lane = true["lanes"][lane_id]
            self.lane_pred_gt[self.id2pos[lane_id]].append(
                self.flatten_lane_pred_gt(pred_lane, true_lane, meta_tag)
            )
        global_pred_gt["general_iou"] = self.calculate_general_iou(pred,true)
        self.global_pred_gt.append(global_pred_gt)

    def generate_meta_tag(self, metadata):
        hour = int(metadata["beijing_time"].split("-")[3])
        meta_tag = {
            "uuid": metadata["uuid"],
            "tagger": metadata["tagger"],
            "beijing_time": metadata["beijing_time"],
            "time": "day" if self.day_time[0] < hour < self.day_time[1] else "night",
        }
        return meta_tag

    def parse_crossability(self, line_type):
        return self.task_config.type2crossability.get(line_type, np.nan)

    def flatten_lane_pred_gt(self, pred, true, meta_tag):
        # return a dict, representing a row in df
        row_data = {}
        row_data.update(meta_tag)
        # check if lane exist first, if not, other attributes will be set to nan
        if not true["exist"][0]:
            row_data["exist_pred"] = pred["exist"][0]
            row_data["exist_true"] = true["exist"][0]
            return row_data
        # flatten lane level feature
        for feature_name in self.task_config.bool_features_lane:
            if (
                feature_name not in true
                or true[feature_name][1][0] == self.task_config.ignore_value
            ):
                continue
            row_data[feature_name + "_pred"] = pred[feature_name][0]
            row_data[feature_name + "_true"] = true[feature_name][0]
            row_data[feature_name + "_prob"] = pred[feature_name][1][0]
        for feature_name in self.task_config.reg_features_lane:
            if feature_name in true:
                row_data[feature_name + "_pred"] = pred[feature_name][1]
                row_data[feature_name + "_true"] = true[feature_name][1]
            else:
                row_data[feature_name + "_pred"] = np.nan
                row_data[feature_name + "_true"] = np.nan
        # flatten line level feature
        for line_name in ["left_line", "right_line"]:
            line_pred = pred[line_name]
            line_true = true[line_name]
            # skip other attributes if line not exist
            if not line_true["line_exist"][0]:
                row_data["line_exist" + "_" + line_name + "_pred"] = line_pred["line_exist"][0]
                row_data["line_exist" + "_" + line_name + "_true"] = line_pred["line_exist"][0]
                continue
            for feature_name in self.task_config.bool_features_line + ["line_type"]:
                # check if this attribute is labelled
                if (
                    feature_name not in line_true
                    or line_true[feature_name][1][0] == self.task_config.ignore_value
                ):
                    continue
                if feature_name != "line_type":
                    row_data[feature_name + "_" + line_name + "_prob"] = line_pred[feature_name][1][0]
                row_data[feature_name + "_" + line_name + "_pred"] = line_pred[feature_name][0]
                row_data[feature_name + "_" + line_name + "_true"] = line_true[feature_name][0]
            for feature_name in self.task_config.other_features_line:
                row_data[feature_name + "_" + line_name + "_pred"] = line_pred[feature_name][0]
                row_data[feature_name + "_" + line_name + "_true"] = line_true[feature_name][0]
        # eval lane iou, and iou by row level
        if (not true["left_line"]["line_exist"][0]) and (not true["right_line"]["line_exist"][0]):
            return row_data
        _, _, iou = cal_iou_v2(self.global_config, pred, true, scale=self.scale)
        if iou in ["error", "invalid"]:
            iou = 0.0
        row_data["lane_iou"] = iou
        for iou_level in self.iou_levels:
            _, _, iou = cal_iou_v2(
                self.global_config,
                pred,
                true,
                target_range=self.iou_levels[iou_level],
                scale=self.scale,
            )
            if iou == "error":
                iou = 0.0
            elif iou == "invalid":
                iou = np.nan
            row_data["lane_iou_" + iou_level] = iou
        return row_data

    def generate_kpi(self):
        #if hasattr(self.global_config, "load_registered_model"):
        #    kpis = XpilotLightningKPI(self.task_config.name,
        #                              self.dataset_identifier,
        #                              self.global_config.load_registered_model)
        #else:
        kpis = {}

        global_df = pd.DataFrame(self.global_pred_gt)
        lane2df = {
            lane: pd.DataFrame(data, columns=self.columns)
            for lane, data in self.lane_pred_gt.items()
        }
        for time in ["whole_day", "night", "day"]:
            kpis[time] = {}
            kpis[time]["global"] = {}
            if time != "whole_day":
                sub_global_df = global_df[global_df.time == time]
            else:
                sub_global_df = global_df
            try:
                kpis[time]["global"]["confusion_matrix"] = compute_confusion_matrix(
                    sub_global_df.global_true.tolist(),
                    sub_global_df.global_pred.tolist(),
                    labels=self.task_config.global_info,
                )
            except ValueError:
                logging.warning("Confusion matrix error", exc_info=True)
            kpis[time]["global"].update(calculate_accuracy(sub_global_df, "global"))
            for category in self.task_config.global_info:
                kpis[time]["global"][category] = {}
                kpis[time]["global"][category].update(calculate_f1(sub_global_df, "global", category))
                kpis[time]["global"][category]["general_iou"] = sub_global_df[sub_global_df["global_true"]==category]["general_iou"].dropna().mean()
            kpis[time]["global"]["general_iou"] = sub_global_df["general_iou"].dropna().mean()
            for group_name, lanes in self.task_config.agg_map.items():
                group_df = pd.concat([lane2df[lane] for lane in lanes], axis=0)
                if time != "whole_day":
                    group_df = group_df[group_df.time == time]
                kpis[time][group_name] = self.eval_lane_df(group_df)
                kpis[time][group_name]["force_merge"] = self.eval_lane_forcemerge(group_df)
                kpis[time][group_name]["cross2noncross"] = self.eval_line_cross2noncross(group_df)
        kpis[const.CORNERCASE_KPIS] = self.generate_corner_cases_kpi(global_df, lane2df)
        if self.upload_to_eval2:
            eval2_kpi = kpis["whole_day"].copy()
            del eval2_kpi["global"]
            kpis[const.EVAL2_KPI] = eval2_kpi
        return kpis

    def eval_lane_df(self, group_df):
        lane_kpis = {}
        lane_kpis["lane_iou"] = {"iou": group_df["lane_iou"].dropna().mean()}
        for iou_level in self.iou_levels:
            lane_kpis["lane_iou_" + iou_level] = {
                "iou": group_df["lane_iou_" + iou_level].dropna().mean()
            }

        for feature_name in self.task_config.bool_features_lane:
            lane_kpis[feature_name] = {}
            lane_kpis[feature_name].update(calculate_f1(group_df, feature_name, True))
            lane_kpis[feature_name].update(calculate_accuracy(group_df, feature_name))

        for feature_name in self.task_config.reg_features_lane:
            lane_kpis[feature_name] = {}
            lane_kpis[feature_name].update(calculate_accuracy(group_df, feature_name))

        for feature_name in self.task_config.bool_features_line:
            sub_df = []
            for line_name in ["left_line", "right_line"]:
                feature_key = feature_name + "_" + line_name
                lane_kpis[feature_key] = {}
                lane_kpis[feature_key].update(calculate_f1(group_df, feature_key, True))
                lane_kpis[feature_key].update(calculate_accuracy(group_df, feature_key))
                sub_df.append(
                    group_df[[feature_key + "_pred", feature_key + "_true"]]
                    .dropna()
                    .rename(
                        {
                            feature_key + "_pred": feature_name + "_pred",
                            feature_key + "_true": feature_name + "_true",
                        },
                        axis=1,
                    )
                )
            sub_df = pd.concat(sub_df, axis=0)
            lane_kpis[feature_name] = {}
            lane_kpis[feature_name].update(calculate_f1(sub_df, feature_name, True))
            lane_kpis[feature_name].update(calculate_accuracy(sub_df, feature_name))

        for feature_name in self.task_config.other_features_line:
            sub_df = []
            for line_name in ["left_line", "right_line"]:
                feature_key = feature_name + "_" + line_name
                lane_kpis[feature_key] = {}
                lane_kpis[feature_key].update(calculate_rmse(group_df, feature_key))
                sub_df.append(
                    group_df[[feature_key + "_pred", feature_key + "_true"]]
                    .dropna()
                    .rename(
                        {
                            feature_key + "_pred": feature_name + "_pred",
                            feature_key + "_true": feature_name + "_true",
                        },
                        axis=1,
                    )
                )
            sub_df = pd.concat(sub_df, axis=0)
            lane_kpis[feature_name] = {}
            lane_kpis[feature_name].update(calculate_rmse(sub_df, feature_name))

        sub_df = []
        for line_name in ["left_line", "right_line"]:
            feature_key = "line_type_" + line_name
            lane_kpis[feature_key] = {}
            lane_kpis[feature_key].update(calculate_accuracy(group_df, feature_key))
            for category in self.task_config.line_types:
                lane_kpis[category + "_" + line_name] = {}
                lane_kpis[category + "_" + line_name].update(
                    calculate_f1(group_df, feature_key, category)
                )
            sub_df.append(
                group_df[[feature_key + "_pred", feature_key + "_true"]]
                .dropna()
                .rename(
                    {
                        feature_key + "_pred": "line_type_pred",
                        feature_key + "_true": "line_type_true",
                    },
                    axis=1,
                )
            )
        sub_df = pd.concat(sub_df, axis=0)
        lane_kpis["line_type"] = {}
        lane_kpis["line_type"].update(calculate_accuracy(sub_df, "line_type"))
        for category in self.task_config.line_types:
            lane_kpis[category] = {}
            lane_kpis[category].update(calculate_f1(sub_df, "line_type", category))

        sub_df["line_crossablity_pred"] = sub_df["line_type_pred"].apply(self.parse_crossability)
        sub_df["line_crossablity_true"] = sub_df["line_type_true"].apply(self.parse_crossability)
        lane_kpis["line_crossable"] = {}
        lane_kpis["line_crossable"].update(calculate_f1(sub_df, "line_crossablity", "crossable"))
        lane_kpis["line_noncrossable"] = {}
        lane_kpis["line_noncrossable"].update(
            calculate_f1(sub_df, "line_crossablity", "non_crossable")
        )
        return lane_kpis

    def eval_lane_forcemerge(self, group_df):
        lane_kpis = {}
        for feature_name in ["merge2left", "merge2right"]:
            lane_kpis[feature_name] = {}
            for threshold in self.corner_case_thresholds:
                lane_kpis[feature_name][threshold] = calculate_f1_with_threshold(group_df, feature_name, threshold)
        return lane_kpis

    def eval_line_cross2noncross(self, group_df):
        lane_kpis = {}
        for line_name in ["left_line", "right_line"]:
            feature_name = "cross2noncross_" + line_name
            lane_kpis[feature_name] = {}
            for threshold in self.corner_case_thresholds:
                lane_kpis[feature_name][threshold] = calculate_f1_with_threshold(group_df, feature_name, threshold)
        return lane_kpis

    def generate_corner_cases_kpi(self, global_df, lane2df):
        kpis = {}
        for key in self.task_config.corner_cases:
            logging.info("Start evaluating corner case: {}".format(key))
            if key in [
                "forcemergefp",
                "fakeforcemergesevereweather",
                "mergerelatedsodcenter",
                "llde38forcemergefp",
            ]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_forcemergefp(sub_df)
            elif key in [
                "forcemergefn",
                "realforcemergesevereweather",
                "forcemergetpobstacle",
                "forcemergetpreversev",
                "llde38forcemergefn",
            ]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_forcemergefn(sub_df)
            elif key in ["fishbone", "llde38fishbone"]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_solid_ahead(sub_df, "slow")
            elif key in [
                "dashsolidsolidaheadrelatedcam0",
                "dashsolidrealtransitioncam0",
                "llde38dashedsolid",
            ]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_solid_ahead(sub_df, "dash_solid")
            elif key in [
                "ramphighway",
                "asphalt",
                "generalrealtransitioncam0",
                "cross2noncrosscam0",
                "llde38solidahead",
            ]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_solid_ahead(sub_df)
            elif key in ["llde38obstacle","llde38soliddashed"]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_line_type(sub_df)
            elif key in [
                "forkhighway",
                "forkhighwayf",
                "forkhighwayy",
                "forkhighwayk",
                "llde38fork",
            ]:
                sub_df = self.filter_corner_case(
                    key, global_df, lane2df, "fork", ["fork1", "fork2"]
                )
                kpis[key] = self.eval_iou(sub_df)
            elif key in ["lldota4poorgeowide", "ota4widecam", "llde38wide"]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "wide", ["wide"])
                kpis[key] = self.eval_iou(sub_df)
            elif key in [
                "llde38ramp", 
                "llde38wornline", 
                "llde38diversionfront",
                "llde38vrailwayfront",
                "llde38vcurbline",
                "llde38vstrangelaneline",
                "llde38vdiversionisland"
            ]:
                sub_df = self.filter_corner_case(key, global_df, lane2df, "normal", ["center"])
                kpis[key] = self.eval_iou(sub_df)
            elif key in ["llde38split"]:
                sub_df = self.filter_corner_case(
                    key, global_df, lane2df, "split", ["assist1", "assist2"]
                )
                kpis[key] = self.eval_iou(sub_df)
            elif key in ["llde38flickerfront"]:
                kpis[key] = self.eval_ego_general_iou(key, global_df)
        return kpis

    def eval_ego_general_iou(self, case_name, global_df):
        sub_df = global_df[global_df.tagger.str.contains("ccdp-" + case_name)]
        result = {
            "n_gt": len(sub_df),
            "threshold": ["iou"], 
            "general_iou": [sub_df["general_iou"].mean()],
            "global_case_accuracy": [(sub_df["global_pred"]==sub_df["global_true"]).mean()],
        }
        return result
        
    def filter_corner_case(self, case_name, global_df, lane2df, target_global, target_lanes):
        uuid_filter = global_df[
            (global_df.global_true == target_global) & (global_df.tagger.str.contains("ccdp-" + case_name))
        ].uuid
        corner_case_df = pd.concat([lane2df[lane] for lane in target_lanes], axis=0)
        corner_case_df = corner_case_df[corner_case_df.uuid.isin(uuid_filter)]
        return corner_case_df

    def eval_forcemergefn(self, df):
        merge2left_df = df[["merge2left_prob", "merge2left_true"]].dropna()
        merge2right_df = df[["merge2right_prob", "merge2right_true"]].dropna()
        left_gt = merge2left_df["merge2left_true"]
        right_gt = merge2right_df["merge2right_true"]
        n_positive_left = left_gt.sum()
        n_positive_right = right_gt.sum()
        result = {
            "threshold": copy(self.corner_case_thresholds),
            "n_positive_left": n_positive_left,
            "n_positive_right": n_positive_right,
        }
        result["2leftTP_ratio"] = []
        result["2rightTP_ratio"] = []
        for thr in self.corner_case_thresholds:
            left_pred = merge2left_df["merge2left_prob"] > thr
            right_pred = merge2right_df["merge2right_prob"] > thr
            result["2leftTP_ratio"].append((left_pred & left_gt).sum() / n_positive_left)
            result["2rightTP_ratio"].append((right_pred & right_gt).sum() / n_positive_right)
        return result

    def eval_forcemergefp(self, df):
        merge2left_df = df[["merge2left_prob", "merge2left_true"]].dropna()
        merge2right_df = df[["merge2right_prob", "merge2right_true"]].dropna()
        left_gt = merge2left_df["merge2left_true"]
        right_gt = merge2right_df["merge2right_true"]
        n_negative_left = len(left_gt) - left_gt.sum()
        n_negative_right = len(right_gt) - right_gt.sum()
        result = {
            "threshold": copy(self.corner_case_thresholds),
            "n_negative_left": n_negative_left,
            "n_negative_right": n_negative_right,
        }
        result["2leftFP_ratio"] = []
        result["2rightFP_ratio"] = []
        for thr in self.corner_case_thresholds:
            left_pred = merge2left_df["merge2left_prob"] > thr
            right_pred = merge2right_df["merge2right_prob"] > thr
            result["2leftFP_ratio"].append(
                (left_pred.sum() - (left_pred & left_gt).sum()) / n_negative_left
            )
            result["2rightFP_ratio"].append(
                (right_pred.sum() - (right_pred & right_gt).sum()) / n_negative_right
            )
        return result

    def eval_solid_ahead(self, df, linetype=None):
        df_lines = []
        for line_name in ["left_line", "right_line"]:
            rename_dict = {
                f"cross2noncross_{line_name}_prob": "cross2noncross_prob",
                f"cross2noncross_{line_name}_true": "cross2noncross_true",
            }
            if linetype:
                sub_df = df[df[f"line_type_{line_name}_true"].str.contains(linetype, na=False)]
            else:
                sub_df = df
            df_lines.append(
                sub_df[[f"cross2noncross_{line_name}_prob", f"cross2noncross_{line_name}_true"]]
                .dropna()
                .rename(rename_dict, axis=1)
            )
        df = pd.concat(df_lines, axis=0)

        n_positive = df["cross2noncross_true"].sum()
        n_negative = len(df) - n_positive
        result = {
            "threshold": copy(self.corner_case_thresholds),
            "n_positive": n_positive,
            "n_negative": n_negative,
        }
        result["TP_ratio"] = []
        result["FP_ratio"] = []
        true = df["cross2noncross_true"]
        for thr in self.corner_case_thresholds:
            pred = df["cross2noncross_prob"] > thr
            result["TP_ratio"].append((pred & true).sum() / n_positive)
            result["FP_ratio"].append((pred.sum() - (pred & true).sum()) / n_negative)
        return result

    def eval_line_type(self, df):
        df_lines = []
        for line_name in ["left_line", "right_line"]:
            rename_dict = {
                f"line_type_{line_name}_pred": "line_type_pred",
                f"line_type_{line_name}_true": "line_type_true",
            }
            df_lines.append(
                df[[f"line_type_{line_name}_pred", f"line_type_{line_name}_true"]]
                .dropna()
                .rename(rename_dict, axis=1)
            )
        df = pd.concat(df_lines, axis=0)
        result = {
            "threshold": ["accuracy"],
            "accuracy": (df["line_type_pred"] == df["line_type_true"]).mean(),
            "n_gt": len(df),
        }
        return result

    def eval_iou(self, df):
        ious = df["lane_iou"].dropna()
        result = {"threshold": ["iou"], "iou": [ious.mean()], "n_gt": len(ious)}
        return result

    def evaluate(self, gt_dict, pred_dict, **kwargs) -> dict:
        corner_case_results = {}
        # iou score
        iou_thresholds = {
            "normal": ("center", 0.94),
            "split": ("assist", 0.83),
            "fork": ("fork", 0.78),
            "wide": ("wide", 0.80),
        }
        global_case = gt_dict["global"][0]
        agg_name = iou_thresholds[global_case][0]
        threshold = iou_thresholds[global_case][1]
        valid_ious = []
        for lane_id in [self.pos2id[lane_position] for lane_position in self.task_config.agg_map[agg_name]]:
            lane_pred = pred_dict["lanes"][lane_id]
            lane_gt = gt_dict["lanes"][lane_id]
            if not lane_gt["exist"][0]:
                continue
            if (not lane_gt["left_line"]["line_exist"][0]) and (not lane_gt["right_line"]["line_exist"][0]):
                continue
            _, _, iou = cal_iou_v2(self.global_config, lane_pred, lane_gt, scale=self.scale)
            if iou in ["error", "invalid"]:
                iou = 0.0
            valid_ious.append(iou)
        if len(valid_ious) > 0:
            iou_score = sum(valid_ious) / len(valid_ious)
        else:
            iou_score = 0.0
        corner_case_results['lld_lane_iou'] = {"result": iou_score > threshold}
        # forcemerge
        threshold = 0.8
        force_merge_result = True
        pred_lane = pred_dict["lanes"][self.pos2id["center"]]
        gt_lane = gt_dict["lanes"][self.pos2id["center"]]
        for key in ["merge2left", "merge2right"]:
            force_merge_result = force_merge_result and gt_lane[key][0] == pred_lane[key][0]
        corner_case_results["merge_in"] = {"result": force_merge_result}
        return corner_case_results
