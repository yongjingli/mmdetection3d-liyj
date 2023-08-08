from xpilot_lightning.machine_learning.tasks.base_linear.config_parsers.config_parser import (
    LinearBaseConfigParser,
)
from xpilot_lightning.machine_learning.tasks.builder import CONFIGPARSERS


@CONFIGPARSERS.register_module()
class LLD2LIGHTNINGConfigParser(LinearBaseConfigParser):
    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)
        self._process_global()
        self._load_extra_corner_case()

    def _load_extra_corner_case(self):
        self.corner_cases.extend(
            [
                "llde38fork",
                "llde38split",
                "llde38wide",
                "llde38forcemergefp",
                "llde38forcemergefn",
                "llde38solidahead",
                "llde38fishbone",
                "llde38dashedsolid",
                "llde38flickerfront",
            ]
        )
        self.corner_cases = list(set(self.corner_cases))

    def _load_constants(self):
        super()._load_constants()
        self.predict_transition_branch = False

        self.reanno_e38 = False
        self.reanno_config = [250, 20]
        self.force_merge_pos_weight = 2.0
        self.softwide_out_of_image = False
        # Split reassign
        self.split_reassign = True
        self.split_hard_margin = 40
        self.split_soft_margin = 80
        self.shared_dist_max = 10
        # Wide reassign
        self.wide_reassign = True
        self.wide_hard_margin = 342.75  # 0.75*image_width
        self.wide_soft_margin = 342.75  # 0.75*image_width
        # label related
        self.need_validate = True
        self.need_normalize = True

        self.global_info = ["split", "fork", "no_lane", "normal", "wide"]
        self.ignore_feature_list = []
        self.ignore_value = -999
        # lane features
        self.bool_features_lane = [
            "exist",
            "opposite_direction",
            "primary",
            "drivable",
            "merge2left",
            "merge2right",
        ]
        self.reg_features_lane = ["num_left_lanes", "num_right_lanes"]
        # line features
        self.line_names = ["left_line", "right_line"]
        self.bool_features_line = ["line_exist", "entirely_imagined", "cross2noncross"]
        self.other_features_line = ["y_start", "y_end"]
        # line types
        self.line_types = [
            "double_yellow",
            "double_white",
            "double_dashed",
            "dash_solid",
            "solid_dash",
            "single_solid",
            "single_dashed",
            "road_curb_edge",
            "bolt",
            "inferenced_line",
            "dashed_slow",
            "solid_slow",
            "obstacle",
            "wide_dense_dashed",
        ]
        self.cross_non_cross_type = {
            "crossable": [
                "double_dashed",
                "dash_solid",
                "single_dashed",
                "bolt",
                "inferenced_line",
                "dashed_slow",
                "wide_dense_dashed",
            ],
            "non_crossable": [
                "double_yellow",
                "double_white",
                "solid_dash",
                "single_solid",
                "road_curb_edge",
                "solid_slow",
                "obstacle",
            ],
            "others": ["unknow"],
        }

        self.number_of_lanes = 10
        self.pos2id = {
            "center": 0,
            "left": 1,
            "right": 2,
            "assist1": 3,
            "assist2": 4,
            "fork1": 5,
            "fork2": 6,
            "wide": 7,
            "wide_left": 8,
            "wide_right": 9,
        }

        self.step_y = 2.5
        self.start_y = 235
        self.num_points_each_line = 64
        self.fine_tune_end_y = True
        # for points extension
        self.need_extend_boundary = True
        self.extend_points_weight = 0.2
        self.skyhigh_threshold = 30.0

        # Validation Use
        self.validation_xboard = {
            "iou_threshold": 0.85,
            "day_time": [5, 19],
            "iou_levels": [[0, 300], [300, 380], [380, 10000]],
        }
        self.agg_map = {
            "center": ["center"],
            "adj": ["left", "right"],
            "assist": ["assist1", "assist2"],
            "fork": ["fork1", "fork2"],
            "wide": ["wide"],
            "adj_wide": ["wide_left", "wide_right"],
        }
        
        self.ego_lanes = {
            "normal": [0],
            "split": [3, 4],
            "wide": [7],
            "fork": [5, 6],
            "no_lane": []
        }
        self.activated_lanes = {
            "normal": [0, 1, 2],
            "split": [3, 4],
            "wide": [7, 8, 9],
            "fork": [5, 6, 1, 2],
            "no_lane": []
        }

        self.corner_case_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        self.bit_to_consider_post_dict = {
            "normal": [0, 1, 2],
            "split": [3, 4],
            "fork": [1, 2, 5, 6, 8, 9],
            "wide": [7, 8, 9],
        }
        self.post_position_dict = {
            0: "center",
            1: "left",
            2: "right",
            3: "center",
            4: "center",
            5: "center",
            6: "center",
            7: "center",
            8: "left",
            9: "right",
        }
        self.fork_bit = [5, 6]
        # Drawing
        self.colors = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 1),
            (1, 0, 0),
            (0, 1, 1),
            (1, 1, 0),
            (0, 0, 0.5),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0.5, 0.5, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.25, 0, 0),
            (0, 0.25, 0),
            (0, 0, 0.25),
            (0.25, 0.25, 0),
            (0, 0.25, 0.25),
            (0.25, 0, 0.25),
            (1, 0.5, 0),
            (0.5, 1, 0),
            (0, 0.5, 1),
        ]

    def _process_global(self):
        super()._process_global()
        self.type2crossability = {}
        for crossability in self.cross_non_cross_type:
            self.type2crossability.update(
                {x: crossability for x in self.cross_non_cross_type[crossability]}
            )
