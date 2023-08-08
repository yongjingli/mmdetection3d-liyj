import cv2
import numpy as np

from xpilot_vision.utils.decorators.tensor2numpy import tensor2numpy
from xpilot_vision.tasks.base.task import BaseTask
from xpilot_lightning.machine_learning.tasks.builder import TASKS
from xpilot_vision.tasks.utils import channel2spatial


@TASKS.register_module
class AP_LLDTask(BaseTask):
    def __init__(self, global_config, task_config, name="ap_lld"):
        BaseTask.__init__(self, global_config, task_config, name)

    @tensor2numpy
    def visualize(self, image, y_hat, uuid=None, color=None, is_gt=False, scale=2, visu_var=False):
        """This function visualize either the groundtruth or prediciton."""
        image = channel2spatial(image, scale)
        image = np.transpose(image * 255.0, (1, 2, 0)).astype(np.uint8)
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
        _, ap_lld_dict = self.postprocess.process(y_hat, is_gt=is_gt)
        lanes = ap_lld_dict["lanes"]
        num = 0
        for each_lane in lanes:
            for _, line in each_lane.items():
                for point in line["points"]:
                    point_color = color if color else self.task_config.colors[num]
                    cv2.circle(
                        image,
                        (int(point["x"]), int(point["y"])),
                        2,
                        point_color,
                        -1,
                    )
                num += 1

        marks = ap_lld_dict["marks"]
        for each_arrow in marks:
            if 'head' in each_arrow:
                x, y = each_arrow['head'][:2]
                head = (int(x), int(y))
                cv2.circle(image, head, 3, (0, 0, 255), -1)
            else:
                head = None

            if 'tail' in each_arrow:
                x, y = each_arrow['tail'][:2]
                tail = (int(x), int(y))
                cv2.circle(image, tail, 3, (255, 255, 0), -1)
            else:
                tail = None

            if head and tail:
                cv2.line(image, head, tail, (0, 255, 255), 1)
            num += 1
        return np.transpose(image[:, :, ::-1], (2, 0, 1))  # C, H ,W

    def apply_distortion_ng(self, points, intrinsic_matrix, dist_coeffs):
        """
        This is the newer format of applying the distortion to the image points

        Parameters:
            points: the points in numpy array
            intrinsec_matrix: the intrinsic matrix for camera calibration
            dist_coeffs: the distortion coefficients list

        Returns:
            points: the points in numpy array after applying distortion
        """
        # Convert the data type to float32
        points = points.astype("float32")
        # Convert pixel values into image plane
        points_homogeneous = cv2.convertPointsToHomogeneous(points)[0]
        # Apply distortion
        distorted_points = cv2.projectPoints(
            points_homogeneous, np.eye(3), np.array([0.0, 0.0, 0.0]), intrinsic_matrix, dist_coeffs
        )
        # Getting the first element from the tuple
        distorted_points = distorted_points[0]
        # Squeeze one more time to reduce one layer
        distorted_points = np.squeeze(distorted_points)

        return distorted_points

    def convert_dds2pred(self, cp_json, uuid, cam_id, distort_dic):
        """Convert ap_lld DDS message into predicted json format"""
        cpd_dds_json = cp_json
        result = {
            "task": "ap_lld",
            "file_name": str(uuid) + '.png',
            "lanes": [],
            "marks": [],
        }

        # cpd_dds_json = cpd_dds_json('ap_det')
        # lld decode
        # intrinsic_cam2 = np.eye(3)
        # # intrinsic_cam2[0, 0] = 1905.5 #fx
        # # intrinsic_cam2[1, 1] = 1905.5 #fy
        # # intrinsic_cam2[0, 2] = 1917.22 #cx
        # # intrinsic_cam2[1, 2] = 1084.65 #cy
        # # distortion_list_cam2 = [-1.16509e-05,4.17047e-05,1.03339,0.106553,0.00384529,1.39896,0.378879,-0.00354132]
        # intrinsic_cam2[0, 0] = 1907.14  # fx
        # intrinsic_cam2[1, 1] = 1907.14  # fy
        # intrinsic_cam2[0, 2] = 1921.39  # cx
        # intrinsic_cam2[1, 2] = 1090.34  # cy
        # distortion_list_cam2 = [-3.3594e-05, 2.038e-05, 1.3513, 0.28346, 0.0043641, 1.7175, 0.67076, 0.040031]
        # P1, P2, K1, K2, K3, K4, K5, K6 = distortion_list_cam2
        # distortion_cam2 = np.array([K1, K2, P1, P2, K3, K4, K5, K6])
        #
        intrinsic_cam3 = np.eye(3)
        intrinsic_cam3[0, 0] = 1106.43  # fx
        intrinsic_cam3[1, 1] = 1106.43  # fy
        intrinsic_cam3[0, 2] = 969.57  # cx
        intrinsic_cam3[1, 2] = 770.757  # cy
        distortion_list_cam3 = [0.000430631, 0.000123, 13.4711, 8.68285, 0.428097, 13.8084, 13.2677, 2.04478]
        P1, P2, K1, K2, K3, K4, K5, K6 = distortion_list_cam3
        distortion_cam3 = np.array([K1, K2, P1, P2, K3, K4, K5, K6])
        #
        intrinsic_cam4 = np.eye(3)
        intrinsic_cam4[0, 0] = 1106.91  # fx
        intrinsic_cam4[1, 1] = 1106.91  # fy
        intrinsic_cam4[0, 2] = 968.228  # cx
        intrinsic_cam4[1, 2] = 770.683  # cy
        distortion_list_cam4 = [0.000331941, 0.000123109, 11.781, 9.18342, 0.596991, 12.1115, 13.2145, 2.49806]
        P1, P2, K1, K2, K3, K4, K5, K6 = distortion_list_cam4
        distortion_cam4 = np.array([K1, K2, P1, P2, K3, K4, K5, K6])

        distort_dic_single = distort_dic[cam_id]
        if distort_dic_single["apply_distort"]:
            intrinsic_cam = np.eye(3)
            intrinsic_cam[0, 0] = distort_dic_single["intrinsic"]["fx"] # 1907.14  # fx
            intrinsic_cam[1, 1] = distort_dic_single["intrinsic"]["fy"] # 1907.14  # fy
            intrinsic_cam[0, 2] = distort_dic_single["intrinsic"]["cx"] # 1921.39  # cx
            intrinsic_cam[1, 2] = distort_dic_single["intrinsic"]["cy"] # 1090.34  # cy
            distortion_list_cam = distort_dic_single["distort_params"] #[-3.3594e-05, 2.038e-05, 1.3513, 0.28346, 0.0043641, 1.7175, 0.67076, 0.040031]
            P1, P2, K1, K2, K3, K4, K5, K6 = distortion_list_cam
            distortion_cam = np.array([K1, K2, P1, P2, K3, K4, K5, K6])
            x_offset, y_offset = [distort_dic_single["crop_offset"][k] for k in ["x", "y"]]
            x_ratio, y_ratio = [distort_dic_single["ratio"][k] for k in ["x", "y"]]

            lld_obj_list = cpd_dds_json['lanes']['pred_lanes']
            for lld_obj in lld_obj_list:
                if lld_obj.__contains__('left') or lld_obj.__contains__('right'):
                    lane_obj = {}
                    if lld_obj.__contains__('left') and len(lld_obj['left']['normalized_image_points']):
                        lane_obj['left_line'] = {"points": []}
                        for point in lld_obj['left']['normalized_image_points']:
                            undistort_point = np.array([[point["pt"]["x"], point["pt"]["y"]]])
                            # distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam, distortion_cam)
                            # x = (distorted_point[0]) / 2.0 - x_offset  # / 4.0
                            # y = (distorted_point[1]) / 2.0 - y_offset  # / 4.0
                            if cam_id == 'cam2':
                                distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam, distortion_cam)
                                x = (distorted_point[0]) / 2.0 - x_offset  # / 4.0
                                y = (distorted_point[1]) / 2.0 - y_offset  # / 4.0
                            elif cam_id == 'cam3':
                                distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam3, distortion_cam3)
                                x = (distorted_point[0] / 2.0 - 4) / 1.5 #/ 4.0
                                y = (distorted_point[1] / 2.0 - 55) / 1.5 #/ 4.0
                            elif cam_id == 'cam4':
                                distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam3, distortion_cam4)
                                x = (distorted_point[0] / 2.0 - 4) / 1.5 #/ 4.0
                                y = (distorted_point[1] / 2.0 - 55) / 1.5 #/ 4.0

                            conf = point['var']
                            # conf = 1.0 # dds have no confidence, let conf=1
                            lane_obj['left_line']["points"].append({"x": float(x), "y": float(y), "score": conf})

                    if lld_obj.__contains__('right') and len(lld_obj['right']['normalized_image_points']):
                        lane_obj['right_line'] = {"points": []}
                        for point in lld_obj['right']['normalized_image_points']:
                            undistort_point = np.array([[point["pt"]["x"], point["pt"]["y"]]])
                            # distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam, distortion_cam)
                            # x = (distorted_point[0]) / 2.0 - x_offset  # / 4.0
                            # y = (distorted_point[1]) / 2.0 - y_offset  # / 4.0

                            if cam_id == 'cam2':
                                distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam, distortion_cam)
                                x = (distorted_point[0]) / 2.0 - x_offset  # / 4.0
                                y = (distorted_point[1]) / 2.0 - y_offset  # / 4.0
                            elif cam_id == 'cam3':
                                distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam3, distortion_cam3)
                                x = (distorted_point[0] / 2.0 - 4) / 1.5 #/ 4.0
                                y = (distorted_point[1] / 2.0 - 55) / 1.5 #/ 4.0
                            elif cam_id == 'cam4':
                                distorted_point = self.apply_distortion_ng(undistort_point, intrinsic_cam4, distortion_cam4)
                                x = (distorted_point[0] / 2.0 - 4) / 1.5 #/ 4.0
                                y = (distorted_point[1] / 2.0 - 55) / 1.5 #/ 4.0
                            conf = point['var']
                            lane_obj['right_line']["points"].append({"x": float(x), "y": float(y), "score": conf})

                    if lane_obj:
                        result['lanes'].append(lane_obj)

            # rsm decode
            arrow2id = {'xpilot::msg::camera_perception::APArrowType::STRAIGHT': 0,
                        'xpilot::msg::camera_perception::APArrowType::LEFT_TURN': 1,
                        'xpilot::msg::camera_perception::APArrowType::RIGHT_TURN': 2,
                        'xpilot::msg::camera_perception::APArrowType::STRAIGHT_LEFT_TURN': 3,
                        'xpilot::msg::camera_perception::APArrowType::STRAIGHT_RIGHT_TURN': 4,
                        'xpilot::msg::camera_perception::APArrowType::STRAIGHT_LEFT_RIGHT_TURN': 5,
                        'xpilot::msg::camera_perception::APArrowType::LEFT_RIGHT_TURN': 6,
                        'xpilot::msg::camera_perception::APArrowType::LEFT_U_TURN': 7,
                        'xpilot::msg::camera_perception::APArrowType::RIGHT_U_TURN': 8,
                        'xpilot::msg::camera_perception::APArrowType::LEFT_TURN_U_TURN': 9,
                        'xpilot::msg::camera_perception::APArrowType::RIGHT_TURN_U_TURN': 10,
                        'xpilot::msg::camera_perception::APArrowType::UNKNOWN': 11}

            rsm_obj_list = cpd_dds_json['rsm_obj_list']
            for rsm_obj in rsm_obj_list:
                if rsm_obj.__contains__('arrow_in_lane'):
                    # if rsm_obj.__contains__('rsm_head_point_raw') and len(rsm_obj['rsm_head_point_raw']):
                    #     if rsm_obj['rsm_head_point_raw'][0]['x'] < 0 or rsm_obj['rsm_head_point_raw'][0]['y'] < 0:
                    #         continue
                    #     head = [(rsm_obj['rsm_head_point_raw'][0]['x']) / 2.0 - 503,
                    #             (rsm_obj['rsm_head_point_raw'][0]['y'])/ 2.0 - 426]  # dds have no confidence of head nor tail
                    # if rsm_obj.__contains__('rsm_tail_point_raw') and rsm_obj['rsm_tail_point_raw']:
                    #     if rsm_obj['rsm_tail_point_raw'][0]['x'] < 0 or rsm_obj['rsm_tail_point_raw'][0]['y'] < 0:
                    #         continue
                    #     tail = [(rsm_obj['rsm_tail_point_raw'][0]['x']) / 2.0 - 503,
                    #             (rsm_obj['rsm_tail_point_raw'][0]['y']) / 2.0 - 426] # dds have no confidence of head nor tail
                    # if rsm_obj.__contains__('rsm_head_point_raw') and len(rsm_obj['rsm_head_point_raw']) and rsm_obj.__contains__('rsm_tail_point_raw') and len(rsm_obj['rsm_tail_point_raw']):
                    #     lane_obj = {"head": head, "tail": tail, "type": float(arrow2id[rsm_obj['arrow_in_lane']])}
                    # elif rsm_obj.__contains__('rsm_head_point_raw') and len(rsm_obj['rsm_head_point_raw']):
                    #     lane_obj = {"head": head, "type": float(arrow2id[rsm_obj['arrow_in_lane']])}
                    # elif rsm_obj.__contains__('rsm_tail_point_raw') and len(rsm_obj['rsm_tail_point_raw']):
                    #     lane_obj = {"tail": tail, "type": float(arrow2id[rsm_obj['arrow_in_lane']])}
                    # else:
                    #     raise ('AP LLD RSM JOSN CONVERT FAILED: ', uuid)
                    # if len(lane_obj):
                    #     result['marks'].append(lane_obj)
                    head = []
                    tail = []
                    if rsm_obj.__contains__('rsm_head_point_raw') and len(rsm_obj['rsm_head_point_raw']):
                        if rsm_obj['rsm_head_point_raw'][0]['x'] >= 0 or rsm_obj['rsm_head_point_raw'][0]['y'] >= 0:
                            head = [(rsm_obj['rsm_head_point_raw'][0]['x']) / 2.0 - x_offset,
                                    (rsm_obj['rsm_head_point_raw'][0]['y']) / 2.0 - y_offset]
                    if rsm_obj.__contains__('rsm_tail_point_raw') and rsm_obj['rsm_tail_point_raw']:
                        if rsm_obj['rsm_tail_point_raw'][0]['x'] >= 0 or rsm_obj['rsm_tail_point_raw'][0]['y'] >= 0:
                            tail = [(rsm_obj['rsm_tail_point_raw'][0]['x']) / 2.0 - x_offset,
                                    (rsm_obj['rsm_tail_point_raw'][0]['y']) / 2.0 - y_offset]
                    if len(head) and len(tail):
                        lane_obj = {"head": head, "tail": tail, "type": float(arrow2id[rsm_obj['arrow_in_lane']])}
                    elif len(head) and (not len(tail)):
                        lane_obj = {"head": head, "type": float(arrow2id[rsm_obj['arrow_in_lane']])}
                    elif (not len(head)) and len(tail):
                        lane_obj = {"tail": tail, "type": float(arrow2id[rsm_obj['arrow_in_lane']])}
                    else:
                        pass

                    if len(lane_obj):
                        result['marks'].append(lane_obj)

            return result
