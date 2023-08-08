import cv2
import math
import numpy as np
import copy
from xpilot_lightning.machine_learning.tasks.builder import PREPROCESSES
from xpilot_vision.tasks.base.preprocess.preprocess import BasePreProcess
from xpilot_lightning.data.dataloader_helpers import LabelProcessor

@PREPROCESSES.register_module
class AP_LLDPreProcessing(BasePreProcess):
    """
    Generate labels and masks of lane lines in underground parkings.

    Now support the following lane annotations:
        Left
        Center
        Right

    Generate labels in the scale with (scale_y, scale_x) downscale factor as compared to
        original resolution (948 * 1828). Both scale_y and scale_x are < 1.
    """

    def __init__(self, global_config, task_config, **kwargs):
        super().__init__(global_config, task_config, **kwargs)
        self.need_validate = task_config.need_validate
        self.map_size = task_config.map_size
        self.lane_image_range = task_config.lane_image_range
        self.lane_pixel_threshold = task_config.lane_pixel_threshold
        self.map_pad = task_config.map_pad
        self.grid_size = task_config.grid_size
        self.ignore_margin = task_config.ignore_margin
        self.lane_map_range = task_config.lane_map_range
        self.mark_image_range = task_config.mark_image_range
        self.arrow2id = task_config.arrow2id
        self.mark_map_range = task_config.mark_map_range
        self.crop_size = task_config.crop_size
        self.start_index = {"lane": 1, "abnormal": 10, "arrow": 1}
        self.scale = (0.5, 0.5)  # (y, x) # todo:e38
        self.shift = (0, 0)  # (y, x) # todo:e38
        self.cam_id = "cam1"
        self.line_index_table = {}
        self.label = None
        self.mask = None
        if hasattr(task_config, 'expect_min_point'):
            self.expect_min_point = 1
        else:
            self.expect_min_point = 2

    def reset_params(self):
        if self.cam_id in self.global_config.e38_crop_method:
            self.crop_range_x_left = self.global_config.e38_crop_method[self.cam_id]['left']
            self.crop_range_y_top = self.global_config.e38_crop_method[self.cam_id]['top']
            self.crop_range_y_bottom = self.global_config.e38_crop_method[self.cam_id]['bottom']
            self.original_e38_x = self.global_config.original_image_shapes['e38'][self.cam_id][1]
            self.original_e38_y = self.global_config.original_image_shapes['e38'][self.cam_id][0]
            self.resize_scale = self.global_config.e38_crop_method[self.cam_id]['resize']

    def set_scale(self, scale, shift):
        self.scale = scale
        self.shift = shift

    def update_anno(self, label_json, cam_id):
        """
        label process has migrated to func remap_data
        """
        return label_json

    def process(self, lanes, data_augments, cam_id=None, uuid='invalid', **kwargs):
        """
        data_augments == {} for lightning dataset version
        """
        label_task = kwargs['metadata']['task'] if 'metadata' in kwargs else 'label_' + lanes['metadata']['task']
        uuid = kwargs['metadata']['uuid'] if 'metadata' in kwargs else 'None'
        self.cam_id = cam_id

        if self.cam_id in self.global_config.e38_crop_method:
            self.crop_range_x_left = self.global_config.e38_crop_method[self.cam_id]['left']
            self.crop_range_y_top = self.global_config.e38_crop_method[self.cam_id]['top']
            self.crop_range_y_bottom = self.global_config.e38_crop_method[self.cam_id]['bottom']
            self.original_e38_x = self.global_config.original_image_shapes['e38'][self.cam_id][1]
            self.original_e38_y = self.global_config.original_image_shapes['e38'][self.cam_id][0]
            self.resize_scale = self.global_config.e38_crop_method[self.cam_id]['resize']

        if lanes:
            ignore_arrowPoints = []
            # Check json annotation
            if self.need_validate:
                if label_task in ['label_ap_lld_3p0']:
                    lanes = self.jsondecoder(lanes, uuid)
                    self.validate_json_3p0(lanes, ignore_arrowPoints)
                elif label_task in ['label_ap_lld', 'label_ap_lld_1p5']:
                    self.validate_json(lanes)
                else:
                    raise Exception('Unknown name of label task for AP_LLD on validate_json')
            # Check data augmentation
            is_flip = data_augments.get("flip", False)
            self.reset(is_flip)

            if label_task in ['label_ap_lld']:
                self.set_lane_lines(lanes, data_augments, label_task)
                self.set_mark_arrows_1p0()
            elif label_task in ['label_ap_lld_1p5']:
                self.set_lane_lines(lanes, data_augments, label_task)
                # self.set_lane_lines_1p5()
                self.set_mark_arrows(lanes, data_augments, label_task, ignore_arrowPoints)
            elif label_task in ['label_ap_lld_3p0']: # todo: e38
                if self.cam_id == 'cam2':
                    self.set_lane_lines(lanes, data_augments, label_task)
                    # self.set_mark_arrows(lanes, data_augments, meta, label_task)
                    self.set_mark_arrows(lanes, data_augments, label_task, ignore_arrowPoints)
                else:
                    self.set_lane_lines(lanes, data_augments, label_task)
                    self.set_mark_arrows_1p0()

            else:
                raise Exception('Unknown name of label task for AP_LLD on set lines and arrows')
            self.crop_gt_maps()

            if self.need_validate:
                self.validate_label()

            return list(self.label), list(self.mask), {}
        else:
            raise Exception('No label task for AP_LLD on uuid: {}'.format(uuid))

    def find_bottom_y(self, points):
        point_y_list = []
        for point in points:
            point_y_list.append(point['y'])
        return max(point_y_list)

    def jsondecoder(self, lanes, uuid):
        # 3p0 lld lines json type go back to 1p5 and 1p0
        new_lanes = {}
        new_lanes['_id'] = uuid
        new_lanes['file_name'] = uuid + '.jpg'
        new_lanes['no_parking_areas'] = []
        new_lanes['task'] = lanes['metadata']['task']
        #lanes
        if lanes.__contains__('lane_list') and len(lanes['lane_list']) > 0:
            new_lanes['lanes'] = []
            # for each lane
            for i, lane in enumerate(lanes['lane_list']):
                cur_lane = {}
                cur_lane['drivable'] = lanes['lane_list'][i]['properties']['drivable']
                cur_lane['opposite_direction'] = lanes['lane_list'][i]['properties']['opposite']
                cur_lane['position'] = lanes['lane_list'][i]['properties']['position']
                cur_lane['primary'] = lanes['lane_list'][i]['properties']['primary']

                # right_line_groups
                if lanes['lane_list'][i].__contains__('right_line'):
                    #each sub_right_line
                    for ii, sub_lane in enumerate(lanes['lane_list'][i]['right_line']):
                        if lanes['lane_list'][i]['right_line'][ii]['properties']['bottom_line']:
                        #only add bottomline for each line_group in order to align to ap_lld and ap_lld_1p5
                            if cur_lane.__contains__('right_line'):
                                if self.find_bottom_y(lanes['lane_list'][i]['right_line'][ii]['points']) > self.find_bottom_y(
                                        lanes['lane_list'][i]['right_line'][ii - 1]['points']):
                                    cur_lane['right_line'] = {}
                                    cur_lane['right_line']['line_type'] = \
                                        lanes['lane_list'][i]['right_line'][ii]['properties']['type']
                                    cur_lane['right_line']['points'] = \
                                        lanes['lane_list'][i]['right_line'][ii]['points']
                                    cur_lane['right_line']['segments'] = []  # for ap_lld, ap_lld_1p5 has no segements
                                # raise Exception("Error label which has two bottomline in one line_group in uuid: {}".format(meta['uuid']))
                            else:
                                cur_lane['right_line'] = {}
                                cur_lane['right_line']['line_type'] = \
                                    lanes['lane_list'][i]['right_line'][ii]['properties']['type']
                                cur_lane['right_line']['points'] = \
                                    lanes['lane_list'][i]['right_line'][ii]['points']
                                cur_lane['right_line']['segments'] = [] # for ap_lld, ap_lld_1p5 has no segements
                    # if not cur_lane.__contains__('right_line'):
                    #     raise Exception("Error label which has no bottomline in one line_group in uuid: {}".format(meta['uuid']))
                # left_line_groups
                if lanes['lane_list'][i].__contains__('left_line'):
                    # each sub_left_line
                    for ii, sub_lane in enumerate(lanes['lane_list'][i]['left_line']):
                        if lanes['lane_list'][i]['left_line'][ii]['properties']['bottom_line']:
                            # only add bottomline for each line_group in order to align to ap_lld and ap_lld_1p5
                            if cur_lane.__contains__('left_line'):
                                if self.find_bottom_y(lanes['lane_list'][i]['left_line'][ii]['points']) > self.find_bottom_y(lanes['lane_list'][i]['left_line'][ii-1]['points']):
                                    cur_lane['left_line'] = {}
                                    cur_lane['left_line']['line_type'] = \
                                        lanes['lane_list'][i]['left_line'][ii]['properties']['type']
                                    cur_lane['left_line']['points'] = \
                                        lanes['lane_list'][i]['left_line'][ii]['points']
                                    cur_lane['left_line']['segments'] = []  # for ap_lld, ap_lld_1p5 has no segements
                                # raise Exception("Error label which has two bottomline in one line_group in uuid: {}".format(meta['uuid']))
                            else:
                                cur_lane['left_line'] = {}
                                cur_lane['left_line']['line_type'] = \
                                    lanes['lane_list'][i]['left_line'][ii]['properties']['type']
                                cur_lane['left_line']['points'] = \
                                    lanes['lane_list'][i]['left_line'][ii]['points']
                                cur_lane['left_line']['segments'] = []  # for ap_lld, ap_lld_1p5 has no segements
                    # if not cur_lane.__contains__('left_line'):
                    #     raise Exception("Error label which has no bottomline in one line_group in uuid: {}".format(meta['uuid']))

                if cur_lane.__contains__('left_line') or cur_lane.__contains__('right_line'):
                    new_lanes['lanes'].append(cur_lane)
                # else:
                #     raise Exception("No bottom lines in existed line in uuid:{}".format(meta['uuid']))
        else:
            new_lanes['lanes'] = []
            # raise Exception("No lanes in uuid:{}".format(meta['uuid']))

        #abnormal lines
        if lanes.__contains__('special_line_list') and len(lanes['special_line_list']) > 0:
            new_lanes['lines'] = []
            for i, spe_lines in enumerate(lanes['special_line_list']):
                for ii, spe_line in enumerate(spe_lines['special_line']):
                    cur_spe_line = {}
                    # if not spe_line.__contains__('properties'):
                    #     print('c3 debug properties')
                    # if not spe_line['properties'].__contains__('type'):
                    #     print('c3 debug type')
                    cur_spe_line['type'] = spe_line['properties']['type']
                    cur_spe_line['points'] = spe_line['points']
                    cur_spe_line['segments'] = []  # for ap_lld, ap_lld_1p5 has no segements
                    new_lanes['lines'].append(cur_spe_line)
        else:
            new_lanes['lines'] = []

        #marks
        if lanes.__contains__('mark_list') and len(lanes['mark_list']) > 0:
            new_lanes['marks'] = []
            for i, marks in enumerate(lanes['mark_list']):
                for ii, mark in enumerate(marks['mark_points']):
                    cur_mark = {}
                    cur_mark['is_visible'] = {}
                    cur_mark['is_visible']['cropped'] = False
                    cur_mark['is_visible']['occluded'] = False
                    cur_mark['is_visible']['worn'] = False
                    cur_mark['is_visible']['blur'] = False
                    if not mark.__contains__('properties'):
                        raise Exception("RSM has no properties in uuid: {}".format(uuid))
                    if mark['properties'].__contains__('is_visible'):
                        for mark_vis in mark['properties']['is_visible']:
                            cur_mark['is_visible'][mark_vis] = True
                    cur_mark['type'] = mark['properties']['type']
                    cur_mark['segments'] = []  # for ap_lld_1p5, ap_lld_3p0 has no segments
                    cur_mark['points'] = mark['points']
                    new_lanes['marks'].append(cur_mark)
        else:
            new_lanes['marks'] = []

        return new_lanes

    def validate_json(self, lanes):
        # Check lane order
        lane_order = {"left": 0, "center": 1, "right": 2}
        lanes["lanes"] = sorted(
            lanes["lanes"], key=lambda l: lane_order[l["position"]], reverse=False
        )
        # Check points of lane lines
        for each_lane in lanes["lanes"]:
            for each_line in ["left_line", "right_line"]:
                if each_line in each_lane:
                    if len(each_lane[each_line]["points"]) < self.expect_min_point:
                        each_lane[each_line]["points"].clear()
                    else:
                        self.finetune_points(each_lane[each_line]["points"])
        for each_line in lanes["lines"]:
            if len(each_line["points"]) < self.expect_min_point:
                each_line["points"].clear()
            else:
                self.finetune_points(each_line["points"])
        # Check arrow vertices
        if "marks" in lanes and len(lanes["marks"]) > 0:
            for each_mark in lanes["marks"]:
                if "type" in each_mark:
                    self.finetune_points(each_mark["points"])

    def validate_json_3p0(self, lanes, ignore_arrowPoints):
        # Check lane order
        lane_order = {"left": 0, "center": 1, "right": 2}
        lanes["lanes"] = sorted(
            lanes["lanes"], key=lambda l: lane_order[l["position"]], reverse=False
        )
        # Check points of lane lines
        del_line_index = []
        for i, each_lane in enumerate(lanes["lanes"]):
            for each_line in ["left_line", "right_line"]:
                if each_line in each_lane:
                    if len(each_lane[each_line]["points"]) < self.expect_min_point:
                        each_lane[each_line]["points"].clear()
                    else:
                        # self.finetune_points_3p0_lines(each_lane[each_line]["points"])
                        self.finetune_fix_points_3p0_lines(each_lane[each_line]["points"])
                    if len(each_lane[each_line]["points"]) == 0:
                        del each_lane[each_line]
            if not (each_lane.__contains__('left_line') or each_lane.__contains__('right_line')):
                del_line_index.append(i)
        new_lanes = []
        for i, each_lane in enumerate(lanes["lanes"]):
            if i not in del_line_index:
                new_lanes.append(each_lane)
        lanes["lanes"] = new_lanes

        del_abline_index = []
        for i, each_line in enumerate(lanes["lines"]):
            if len(each_line["points"]) < self.expect_min_point:
                each_line["points"].clear()
            else:
                # self.finetune_points_3p0_lines(each_line["points"])
                self.finetune_fix_points_3p0_lines(each_line["points"])
            if len(each_line["points"]) == 0:
                del_abline_index.append(i)
        new_ablines = []
        for i, each_abline in enumerate(lanes["lines"]):
            if i not in del_abline_index:
                new_ablines.append(each_abline)
        lanes["lines"] = new_ablines

        # Check arrow vertices
        if self.cam_id == 'cam2': # or self.cam_id == 'cam1': #c3 not init arrows in cam3 cam4
            if "marks" in lanes and len(lanes["marks"]) > 0:
                del_index = []
                for i, each_mark in enumerate(lanes["marks"]):
                    if "type" in each_mark:
                        if self.finetune_points_3p0_arrows(each_mark['points']):
                            del_index.append(i)
                        #create ignore arrow point
                        ignore_arrowPoint = self.get_ignore_arrowPoints(each_mark['points']) # todo: merge with self.finetune_points_3p0_arrows
                        if len(ignore_arrowPoint):
                            ignore_arrowPoints.append(ignore_arrowPoint)
                new_marks = []
                for i, each_mark in enumerate(lanes["marks"]):
                    if i not in del_index:
                        new_marks.append(each_mark)
                lanes["marks"] = new_marks

    def finetune_points(self, points):
        map_h, map_w = self.map_size
        scale_y, scale_x = self.scale
        label_h, label_w = int(map_h / scale_y), int(map_w / scale_x) #c3 960*1280
        for point in points:
            point["x"] = min(point["x"], label_w - 2)
            point["y"] = min(point["y"], label_h - 2)

    def finetune_points_3p0_arrows(self, points):
        for point in points:
            if point["y"] == self.original_e38_y * 2:  # 1550.0
                point["y"] = point["y"] - 2
            if self.cam_id == 'cam3' or self.cam_id == 'cam4':
                # if point["x"] < 7 + 1 or point["x"] > 1927 - 1 or point["y"] < 109 + 1 or point["y"] > 1549 - 1:
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2 + 1) - 1 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - 1) - 1 :
                    return True
            elif self.cam_id == 'cam2': #todo: e38 e38 another crop and resize method
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2) - 2 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - self.crop_range_y_bottom * 2 - 1) - 1:
                # if point["x"] < 1005 + 1 or point["x"] > 2833 - 1 or point["y"] < 999 + 1 or point["y"] > 1947 - 1:
                # if point["x"] <= 91 + 1 or point["x"] >= 3747 -1 or point["y"] <= 51  + 1 or point["y"] >= 1947 - 1: #e38 crop min method
                    return True
            else:
                raise Exception(
                    "Unsupport cam: {}".format(self.cam_id))
        return False

    def get_ignore_arrowPoints(self, points):
        ignore_points = []
        cropped_arrow = False
        for point in points:
            if self.cam_id == 'cam2':
                # if point["x"] < 1005 + 1 or point["x"] > 2833 - 1 or point["y"] < 999 + 1 or point["y"] > 1947 - 1:
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2) - 2 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - self.crop_range_y_bottom * 2 - 1) - 1:
                    cropped_arrow = True
                else:
                    ignore_points.append(point)
            elif self.cam_id == 'cam3' or self.cam_id == 'cam4':
                # if point["x"] < 8.0 or point["x"] > 1926.0 or point["y"] < 108.0 or point["y"] > 1548.0:
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2 + 1) - 1 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - 1) - 1 :
                    cropped_arrow = True
                else:
                    ignore_points.append(point)
            else:
                raise Exception("Unsupport cam: {}".format(self.cam_id))
        if cropped_arrow:
            return ignore_points
        else:
            return []

    def fix_crop_line(self, points, x_start, x_end, y_start, y_end):
        ### fit bottom point of a cropped line

        #create index_list
        points = sorted(points, key=lambda e: e['y'], reverse=True)
        index_list = [1 for i in range(len(points))]
        for i, point in enumerate(points):
            if point["x"] < x_start or point["x"] > x_end or point["y"] < y_start or point["y"] > y_end:
                index_list[i] = 0

        # find boundary point pair
        boundary_indexs = []
        if not sum(index_list) == 0:
            for i, index_value in enumerate(index_list):
                if i > 0:
                    if index_value != index_list[i-1]:
                        boundary_indexs.append([i - 1, i])

        #fit bottom point
        replace_points = [[] for i in range(len(boundary_indexs))]
        for ii, boundary_index in enumerate(boundary_indexs):
            outside_index = boundary_index[0] if index_list[boundary_index[0]] == 0 else boundary_index[1]
            fix_point = {}
            base_point = []
            point = []
            base_point.append(points[boundary_index[0]]['x'])
            base_point.append(points[boundary_index[0]]['y'])
            point.append(points[boundary_index[1]]['x'])
            point.append(points[boundary_index[1]]['y'])

            replace_points[ii].append(outside_index)
            if base_point[0] != point[0]:
                k = (base_point[1] - point[1]) / (base_point[0] - point[0])
                b = base_point[1] - base_point[0] * k
                if k == 0:
                    fix_point['y'] = point[1]
                    if points[outside_index]['x'] < x_start:
                        fix_point['x'] = x_start
                    else:
                        fix_point['x'] = x_end
                else:
                    if points[outside_index]['x'] > x_end:
                        y1 = y_start
                        x1 = (y1 - b) / k
                        if x1 <= x_end and x1 >= x_start:
                            fix_point['x'] = x1
                            fix_point['y'] = y1
                        y2 = y_end
                        x2 = (y2 - b) / k
                        if x2 <= x_end and x2 >= x_start:
                            fix_point['x'] = x2
                            fix_point['y'] = y2
                        x3 = x_end
                        y3 = k * x3 + b
                        if y3 <= y_end and y3 >= y_start:
                            fix_point['x'] = x3
                            fix_point['y'] = y3
                    elif points[outside_index]['x'] < x_start:
                        y1 = y_start
                        x1 = (y1 - b) / k
                        if x1 <= x_end and x1 >= x_start:
                            fix_point['x'] = x1
                            fix_point['y'] = y1
                        y2 = y_end
                        x2 = (y2 - b) / k
                        if x2 <= x_end and x2 >= x_start:
                            fix_point['x'] = x2
                            fix_point['y'] = y2
                        x3 = x_start
                        y3 = k * x3 + b
                        if y3 <= y_end and y3 >= y_start:
                            fix_point['x'] = x3
                            fix_point['y'] = y3
                    else:
                        if points[outside_index]['y'] < y_start:
                            fix_point['y'] = y_start
                            fix_point['x'] = (y_start - b) / k
                        else:
                            fix_point['y'] = y_end
                            fix_point['x'] = (y_end - b) / k
            else:
                fix_point['x'] = point[0]
                if points[outside_index]['y'] < y_start:
                    fix_point['y'] = y_start
                else:
                    fix_point['y'] = y_end
            replace_points[ii].append(fix_point)

        #replace point
        ###replace_points :[[outside_index1, {'x':222.0,'y':.111.0}][outside_index2, {'x':223.0,'y':.113.0}]..]
        for replace_point in replace_points:
            points[replace_point[0]] = replace_point[1]

        return points

    def finetune_fix_points_3p0_lines(self, points):  #
        for point in points:
            if point["y"] == self.original_e38_y * 2:  # 1550.0
                point["y"] = point["y"] - 2

        if self.cam_id == 'cam3' or self.cam_id == 'cam4':
            # points = self.fix_crop_line(points, 8.0, 1926.0, 108.0, 1548.0)
            x_start = self.crop_range_x_left*2
            x_end = self.original_e38_x * 2 - self.crop_range_x_left * 2
            y_start = self.crop_range_y_top * 2 - 2
            y_end = self.original_e38_y * 2 - 2
            points = self.fix_crop_line(points, x_start, x_end, y_start, y_end)
        elif self.cam_id == 'cam2':
            # points = self.fix_crop_line(points, 1006.0, 2832.0, 1000.0, 1946.0)
            x_start = self.crop_range_x_left*2
            x_end = self.original_e38_x * 2 - self.crop_range_x_left * 2
            y_start = self.crop_range_y_top * 2 - 2
            y_end = self.original_e38_y * 2 - 2
            points = self.fix_crop_line(points, x_start, x_end, y_start, y_end)
        else:
            raise Exception("Unsupport cam: {}".format(self.cam_id))

        tmp_points = copy.deepcopy(points)
        for point in tmp_points:
            if self.cam_id == 'cam3' or self.cam_id == 'cam4':
                # if point["x"] < 7 + 1 or point["x"] > 1927 - 1 or point["y"] < 109 + 1 or point["y"] > 1549 - 1:
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2 + 1) - 1 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - 1) - 1 :
                    points.remove(point)
            elif self.cam_id == 'cam2':  # todo: e38 another crop and resize method
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2) - 2 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - self.crop_range_y_bottom * 2 - 1) - 1:
                # if point["x"] < 1005 + 1 or point["x"] > 2833 - 1 or point["y"] < 999 + 1 or point["y"] > 1947 - 1:
                    # if point["x"] <= 45 + 1 or point["x"] >= 1873-1 or point["y"] <= 25 + 1 or point["y"] >= 973 - 1: #e38 crop min method
                    points.remove(point)
            else:
                raise Exception("Unsupport cam: {}".format(self.cam_id))
        if len(points) < self.expect_min_point:
            points.clear()

    def finetune_points_3p0_lines(self, points):  #
        # map_h, map_w = self.map_size
        # scale_y, scale_x = self.scale
        # label_h, label_w = int(map_h / scale_y), int(map_w / scale_x)
        for point in points:
            if point["y"] == self.original_e38_y * 2:  # 1550.0
                point["y"] = point["y"] - 2
        tmp_points = copy.deepcopy(points)
        for point in tmp_points:
            if self.cam_id == 'cam3' or self.cam_id == 'cam4':
                            # left:4*2                                        # 968 *2 - 8    # 55 * 2        # 775 * 2
                if point["x"] < (self.crop_range_x_left * 2 - 1) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2 + 1) - 1 \
                        or point["y"] < (self.crop_range_y_top * 2 - 1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - 1) - 1 :
                # if point["x"] < 7 + 1 or point["x"] > 1927 - 1 or point["y"] < 109 + 1 or point["y"] > 1549 - 1:
                    points.remove(point)
            elif self.cam_id == 'cam2': #todo: e38 another crop and resize method
                # left: 503 * 2         # left: 1920 * 2 - 503 * 2   # top: 500 * 2          # 1080*2 - 106 * 2
                if point["x"] < (self.crop_range_x_left * 2 - 1 ) + 1 \
                        or point["x"] > (self.original_e38_x * 2 - self.crop_range_x_left * 2) - 2 \
                        or point["y"] < (self.crop_range_y_top * 2 -1) + 1 \
                        or point["y"] > (self.original_e38_y * 2 - self.crop_range_y_bottom * 2 -1) -1:
                # if point["x"] < 1005 + 1 or point["x"] > 2833 - 1 or point["y"] < 999 + 1 or point["y"] > 1947 - 1:
                # if point["x"] <= 45 + 1 or point["x"] >= 1873-1 or point["y"] <= 25 + 1 or point["y"] >= 973 - 1: #e38 crop min method
                    points.remove(point)
            else:
                raise Exception(
                    "Unsupport cam: {}".format(self.cam_id))
        if len(points) < self.expect_min_point:
            points.clear()

    def reset(self, is_flip):
        assert is_flip == False, "Only support is_flip==False currently."
        self.line_index_table = {}
        self.label = None
        self.mask = None

    def set_lane_lines(self, lanes, data_augments, label_task):
        line_map = np.zeros(self.map_size, dtype=np.uint8)
        line_points = {}
        # Lane lines
        line_index = self.start_index["lane"]
        for each_lane in lanes["lanes"]:
            for each_line in ["left_line", "right_line"]:
                if each_line in each_lane:
                # if each_line in each_lane and len(each_lane[each_line]["points"]) > 0: # c3 add for e38 crop null line
                    self.remap_data(each_lane[each_line], "points", data_augments, label_task)  # Discard those outliers
                    self.set_center_lines(line_map, line_index, each_lane[each_line]["points"])
                    self.get_line_points(line_points, line_map, line_index)
                line_index += 1
        # Abnormal lines
        line_index = self.start_index["abnormal"]
        for each_line in lanes["lines"]:
            # if len(each_line["points"]) > 0: # c3 add for e38 crop null line
            self.remap_data(each_line, "points", data_augments, label_task)
            self.set_center_lines(line_map, line_index, each_line["points"])
            self.get_line_points(line_points, line_map, line_index)
            line_index += 1
        if self.cam_id in {"cam3", "cam4", "cam5", "cam6"}:
            self.set_center_lines_side(line_map)
            self.get_line_points_side(line_points, line_map)
        # Pad sides
        if len(self.map_pad) == 4:
            top, bottom, left, right = self.map_pad
            line_map = np.pad(
                line_map, ((top, bottom), (left, right)), mode="constant", constant_values=(0, 0)
            )
            for points in line_points.values():
                for point in points:
                    point[0], point[1] = point[0] + left, point[1] + top
        # Set mapping table of line index
        if self.cam_id in {"cam0", "cam1", "cam2"}:
            self.set_line_index_table(line_points)
        elif self.cam_id in {"cam3", "cam4", "cam5", "cam6"}:
            self.set_line_index_table_side(line_points)
        else:
            raise KeyError("Cam_ID is invalid in AP_LLD.")
        gt_line_maps, gt_line_masks = self.set_gt_line_maps(line_map)
        self.label, self.mask = gt_line_maps, gt_line_masks

    def remap_data(self, label_dict, point_key, data_augments, label_task):
        """After scaling and shifting, some of the lane data may fall out of the image.
        Discard those outliers and if the entire laneline is out of the image, change label according to it.
        """
        points = []
        for point in label_dict[point_key]:
            # Scaling and shifting

            if 'label_ap_lld_3p0' in label_task:
                if self.cam_id == 'cam2':
                    x = point["x"] * 0.5
                    y = point["y"] * 0.5
                    x = x - self.crop_range_x_left   # 503
                    y = y - self.crop_range_y_top    # 500  # self.crop_range_y_top
                    x = x
                    y = y
                    #c3 anthor shift mathod for cam2 todo: e38 not confirm
                    # x = point["x"] * 0.5
                    # y = point["y"] * 0.5
                    # x = x - 26
                    # y = y - 46
                    # x = x * 0.5
                    # y = y * 0.5
                elif self.cam_id == 'cam3' or self.cam_id == 'cam4':
                    x = point["x"] * 0.5
                    y = point["y"] * 0.5
                    x = x - self.crop_range_x_left # 4    # self.crop_range_x_left
                    y = y - self.crop_range_y_top #55   # self.crop_range_y_top
                    x = x / self.resize_scale  #  x * 2.0 / 3.0
                    y = y / self.resize_scale  #  y * 2.0 / 3.0
                else:
                    raise Exception("Unkown cam shift method")
            else:
                x = point["x"] * self.scale[1]
                y = point["y"] * self.scale[0]
                x = x - self.shift[1]
                y = y - self.shift[0]

            # Augmentation1
            assert "scale" not in data_augments, "AP_LLD doesn't support 'scale'."
            assert "shift" not in data_augments, "AP_LLD doesn't support 'shift'."
            if "position" in point:
                points.append({"position": int(point["position"]), "x": x, "y": y})
            else:
                points.append({"x": x, "y": y})
        label_dict[point_key] = points

    def set_center_lines(self, line_map, line_index, line_points):
        test_map = np.zeros_like(line_map)
        self.draw_line(test_map, 1, line_points)
        top, bottom = self.lane_image_range
        if test_map[top:bottom, :].sum() > self.lane_pixel_threshold:
            self.draw_line(line_map, line_index, line_points)

    def draw_line(self, line_map, line_index, line_points):
        if len(line_points) == 0:
            print('c3 debug len(line_points) == 0')
        pre_point = line_points[0]
        for cur_point in line_points[1:]:
            x1, y1 = round(pre_point["x"]), round(pre_point["y"])
            x2, y2 = round(cur_point["x"]), round(cur_point["y"])
            cv2.line(line_map, (x1, y1), (x2, y2), (line_index,))
            pre_point = cur_point

    def set_center_lines_side(self, line_map):
        top, bottom = self.lane_image_range
        line_map[0:top, :] = 0
        index_num, index_map = cv2.connectedComponents(line_map, connectivity=8)
        line_map[...] = index_map.astype(np.uint8)[...]
        for line_index in range(1, index_num):
            if (line_map == line_index).sum() <= self.lane_pixel_threshold:
                line_map[line_map == line_index] = 0

    def get_line_points(self, line_points, line_map, line_index):
        points = np.nonzero(line_map == line_index)
        points = [[x, y] for x, y in zip(*points)]
        if len(points) > 0:
            line_points[line_index] = points

    def get_line_points_side(self, line_points, line_map):
        line_points.clear()
        index_num = line_map.max() + 1
        for line_index in range(1, index_num):
            self.get_line_points(line_points, line_map, line_index)

    def set_line_index_table(self, line_points):
        line_grids = {}
        for index, points in line_points.items():
            grids = set((x // self.grid_size, y // self.grid_size) for x, y in points)
            line_grids[index] = grids
        # Lane line
        start_id, end_id = self.start_index["lane"], self.start_index["abnormal"]
        for id in range(start_id, end_id):
            if (id in line_grids) and (id % 2 == 1) and (id - 1 in line_grids):
                pre_right_line = line_grids[id - 1]
                cur_left_line = line_grids[id]
                shared_grids = cur_left_line.intersection(pre_right_line)
                if len(shared_grids) > 0:
                    self.line_index_table[id] = id - 1
                else:
                    self.line_index_table[id] = id
            elif id in line_grids:
                self.line_index_table[id] = id
        # Abnormal line
        start_id, end_id = self.start_index["abnormal"], self.start_index["abnormal"] + 10
        for id in range(start_id, end_id):
            if id in line_grids:
                self.line_index_table[id] = id

    def set_line_index_table_side(self, line_points):
        line_grids = {}
        for index, points in line_points.items():
            grids = set((x // self.grid_size, y // self.grid_size) for x, y in points)
            line_grids[index] = grids
        start_id, end_id = self.start_index["lane"], self.start_index["abnormal"] + 10
        for id in range(start_id, end_id):
            if id in line_grids:
                self.line_index_table[id] = id

    def set_gt_line_maps(self, line_map):
        line_map_h, line_map_w = line_map.shape
        gt_map_h, gt_map_w = math.ceil(line_map_h / self.grid_size), math.ceil(
            line_map_w / self.grid_size
        )
        gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_line_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        for y in range(0, gt_map_h):
            for x in range(0, gt_map_w):
                start_x, end_x = x * self.grid_size, (x + 1) * self.grid_size
                end_x = end_x if end_x < line_map_w else line_map_w
                start_y, end_y = y * self.grid_size, (y + 1) * self.grid_size
                end_y = end_y if end_y < line_map_h else line_map_h
                grid = line_map[start_y:end_y, start_x:end_x]
                confidence = 1 if np.any(grid) else 0
                gt_confidence[0, y, x] = confidence
                if confidence == 1:
                    ys, xs = np.nonzero(grid)
                    offset_y, offset_x = sorted(
                        zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                    )[0]
                    gt_offset_x[0, y, x] = offset_x / (self.grid_size - 1)
                    gt_offset_y[0, y, x] = offset_y / (self.grid_size - 1)
                    gt_line_index[0, y, x] = self.line_index_table[grid[offset_y, offset_x]]
        gt_line_maps = np.concatenate(
            (gt_confidence, gt_offset_x, gt_offset_y, gt_line_index), axis=0
        )
        foreground_mask = gt_confidence.astype(np.uint8)
        if self.ignore_margin > 0:
            ignore_mask = 1 - cv2.dilate(foreground_mask[0], None, iterations=self.ignore_margin)
            ignore_mask = ignore_mask[None, ...] + foreground_mask
            top, bottom = self.lane_map_range
            ignore_mask[0, :top, :] = 0
        else:
            ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
            top, bottom = self.lane_map_range
            ignore_mask[0, top:bottom, :] = 1
        gt_line_masks = np.concatenate((ignore_mask, foreground_mask), axis=0)
        return gt_line_maps, gt_line_masks

    def set_lane_lines_1p5(self):
        line_map_h, line_map_w = self.map_size
        if len(self.map_pad) == 4:
            top, bottom, left, right = self.map_pad
            line_map_h += top + bottom
            line_map_w += left + right
        gt_map_h, gt_map_w = math.ceil(line_map_h / self.grid_size), math.ceil(line_map_w / self.grid_size)
        self.label = np.zeros((4, gt_map_h, gt_map_w), dtype=np.float)
        self.mask = np.zeros((2, gt_map_h, gt_map_w), dtype=np.uint8)

    def set_mark_arrows(self, lanes, data_augments, label_task, ignore_arrowPoints):
        if "marks" not in lanes or len(lanes["marks"]) == 0:
            arrow_map_h, arrow_map_w = self.map_size
            if len(self.map_pad) == 4:
                top, bottom, left, right = self.map_pad
                arrow_map_h += top + bottom
                arrow_map_w += left + right
            gt_map_h, gt_map_w = math.ceil(arrow_map_h / self.grid_size), math.ceil(
                arrow_map_w / self.grid_size
            )
            gt_arrow_maps = np.zeros((6, gt_map_h, gt_map_w), dtype=np.float)
            gt_arrow_masks = np.zeros((2, gt_map_h, gt_map_w), dtype=np.uint8)
            top, bottom = self.mark_map_range
            gt_arrow_masks[0, top:bottom, :] = 1
        else:
            arrow_vertex = np.zeros(self.map_size, dtype=np.uint8)
            arrow_ignore = np.zeros(self.map_size, dtype=np.bool)
            arrow_type = np.zeros(self.map_size, dtype=np.uint8)
            vertex_type = np.zeros(self.map_size, dtype=np.uint8)
            arrow_index = self.start_index["arrow"]
            self.set_arrow_ignoreMaps(arrow_ignore, ignore_arrowPoints)
            for each_mark in lanes["marks"]:
                if "type" not in each_mark:
                    continue
                self.remap_data(each_mark, "points", data_augments, label_task)
                self.set_arrow_maps(
                    arrow_vertex,
                    arrow_ignore,
                    arrow_type,
                    vertex_type,
                    arrow_index,
                    each_mark
                )
                arrow_index += 1
            if len(self.map_pad) == 4:
                top, bottom, left, right = self.map_pad
                arrow_vertex = np.pad(
                    arrow_vertex,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                arrow_ignore = np.pad(
                    arrow_ignore,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=(False, False),
                )
                arrow_type = np.pad(
                    arrow_type,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=(0, 0),
                )
                vertex_type = np.pad(
                    vertex_type,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=(0, 0),
                )
            gt_arrow_maps, gt_arrow_masks = self.set_gt_arrow_maps(
                arrow_vertex, arrow_type, vertex_type, arrow_ignore
            )
        self.label = np.concatenate((self.label, gt_arrow_maps), axis=0)
        self.mask = np.concatenate((self.mask, gt_arrow_masks), axis=0)

    def set_arrow_ignoreMaps(self, arrow_ignore_map, ignore_arrowPoints):
        # c3 ignore crop arrow points
        if len(ignore_arrowPoints):
            scaled_ignore_arrowPoints = []
            for each_igore_arrowPoints in ignore_arrowPoints:
                scaled_each_ignore_arrowPoints = {}
                for each_igore_arrowPoint in each_igore_arrowPoints:
                    # same as remap_data
                    ignore_x = each_igore_arrowPoint["x"] * 0.5
                    ignore_y = each_igore_arrowPoint["y"] * 0.5
                    ignore_x = ignore_x - 503
                    ignore_y = ignore_y - 500
                    if "position" in each_igore_arrowPoint:
                        scaled_each_ignore_arrowPoints[int(each_igore_arrowPoint["position"])] = [ignore_x, ignore_y]
                    else:
                        raise Exception('mark without propert: "position"')
                scaled_ignore_arrowPoints.append(scaled_each_ignore_arrowPoints)

            for each_igore_arrowPoints in scaled_ignore_arrowPoints:
                if 0 in each_igore_arrowPoints:
                    x, y = round(each_igore_arrowPoints[0][0]), round(each_igore_arrowPoints[0][1])
                    arrow_ignore_map[y, x] = True
                if 3 in each_igore_arrowPoints and 4 in each_igore_arrowPoints:
                    x = 0.5 * (each_igore_arrowPoints[3][0] + each_igore_arrowPoints[4][0])
                    y = 0.5 * (each_igore_arrowPoints[3][1] + each_igore_arrowPoints[4][1])
                    x, y = round(x), round(y)
                    arrow_ignore_map[y, x] = True

    def set_arrow_maps(self, arrow_vertex_map, arrow_ignore_map, arrow_type_map,
                       vertex_type_map, arrow_index, arrow):
        arrow_vertices = arrow["points"]
        arrow_type = arrow["type"]

        three_vertices = {}
        for vertex in arrow_vertices:
            if "position" in vertex and vertex["position"] in [0, 3, 4]:
                three_vertices[vertex["position"]] = [vertex["x"], vertex["y"]]
        two_vertices = [None, None]
        if 0 in three_vertices:
            two_vertices[0] = three_vertices[0]
        if 3 in three_vertices and 4 in three_vertices:
            x = 0.5 * (three_vertices[3][0] + three_vertices[4][0])
            y = 0.5 * (three_vertices[3][1] + three_vertices[4][1])
            two_vertices[1] = [x, y]

        top, bottom = self.mark_image_range
        ys = [v["y"] for v in arrow_vertices]

        if max(ys) < top:
            for vertex in two_vertices:
                if not vertex:
                    continue
                x, y = round(vertex[0]), round(vertex[1])
                arrow_ignore_map[y, x] = True
        else:
            for vertex_type, vertex in enumerate(two_vertices):
                if not vertex:
                    continue
                x, y = round(vertex[0]), round(vertex[1])
                # if x >= 640 or y >= 480:
                #     print('c3 debug 640480')
                arrow_vertex_map[y, x] = arrow_index
                arrow_type_map[y, x] = self.arrow2id[arrow_type]
                vertex_type_map[y, x] = vertex_type

    def set_gt_arrow_maps(self, arrow_vertex, arrow_type, vertex_type, arrow_ignore):
        arrow_map_h, arrow_map_w = arrow_vertex.shape
        gt_map_h, gt_map_w = math.ceil(arrow_map_h / self.grid_size), math.ceil(
            arrow_map_w / self.grid_size
        )
        gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_arrow_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_arrow_type = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        gt_vertex_type = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float)
        ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.bool)
        for y in range(0, gt_map_h):
            for x in range(0, gt_map_w):
                start_x, end_x = x * self.grid_size, (x + 1) * self.grid_size
                end_x = end_x if end_x < arrow_map_w else arrow_map_w
                start_y, end_y = y * self.grid_size, (y + 1) * self.grid_size
                end_y = end_y if end_y < arrow_map_h else arrow_map_h
                grid = arrow_vertex[start_y:end_y, start_x:end_x]
                confidence = 1 if np.any(grid) else 0
                gt_confidence[0, y, x] = confidence
                if confidence == 1:
                    ys, xs = np.nonzero(grid)
                    offset_y, offset_x = sorted(
                        zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                    )[0]
                    gt_offset_x[0, y, x] = offset_x / (self.grid_size - 1)
                    gt_offset_y[0, y, x] = offset_y / (self.grid_size - 1)
                    gt_arrow_index[0, y, x] = grid[offset_y, offset_x]
                    gt_arrow_type[0, y, x] = arrow_type[start_y:end_y, start_x:end_x][
                        offset_y, offset_x
                    ]
                    gt_vertex_type[0, y, x] = vertex_type[start_y:end_y, start_x:end_x][
                        offset_y, offset_x
                    ]
                grid = arrow_ignore[start_y:end_y, start_x:end_x]
                if np.any(grid):
                    ignore_mask[0, y, x] = True
        gt_arrow_maps = np.concatenate(
            (
                gt_confidence,
                gt_offset_x,
                gt_offset_y,
                gt_arrow_index,
                gt_arrow_type,
                gt_vertex_type,
            ),
            axis=0,
        )
        gt_ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
        top, bottom = self.mark_map_range
        gt_ignore_mask[0, top:bottom, :] = 1
        gt_foreground_mask = gt_confidence.astype(np.uint8)
        if self.ignore_margin > 0:
            ignore_mask = cv2.dilate(ignore_mask[0], None, iterations=self.ignore_margin)[None, ...]
            gt_ignore_mask[ignore_mask] = 0
            ignore_mask = cv2.dilate(gt_foreground_mask[0], None, iterations=self.ignore_margin)
            ignore_mask = ignore_mask[None, ...] - gt_foreground_mask
            gt_ignore_mask[ignore_mask] = 0
        else:
            gt_ignore_mask[ignore_mask] = 0
        gt_arrow_masks = np.concatenate((gt_ignore_mask, gt_foreground_mask), axis=0)
        return gt_arrow_maps, gt_arrow_masks

    def set_mark_arrows_1p0(self):
        arrow_map_h, arrow_map_w = self.map_size
        if len(self.map_pad) == 4:
            top, bottom, left, right = self.map_pad
            arrow_map_h += top + bottom
            arrow_map_w += left + right
        gt_map_h, gt_map_w = math.ceil(arrow_map_h / self.grid_size), math.ceil(arrow_map_w / self.grid_size)
        gt_arrow_maps = np.zeros((6, gt_map_h, gt_map_w), dtype=np.float)
        gt_arrow_masks = np.zeros((2, gt_map_h, gt_map_w), dtype=np.uint8)
        self.label = np.concatenate((self.label, gt_arrow_maps), axis=0)
        self.mask = np.concatenate((self.mask, gt_arrow_masks), axis=0)

    def crop_gt_maps(self):
        if len(self.crop_size) == 4:
            top, bottom, left, right = self.crop_size
            self.label = self.label[:, top:bottom, left:right]
            self.mask = self.mask[:, top:bottom, left:right]

    def validate_label(self):
        pass
