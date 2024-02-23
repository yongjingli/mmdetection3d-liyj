import cv2

color_list = [(255, 0, 0),      #    - 红色
              (0, 255, 0),      # - 绿色
              (0, 0, 255),      # - 蓝色
              (255, 255, 0),    # - 黄色
              (255, 0, 255),    # - 紫色
              (0, 255, 255),    # - 青色
              (128, 0, 0),      # - 深红色
              (0, 128, 0),      # - 深绿色
              (0, 0, 128),      # - 深蓝色
              (128, 128, 0),    # - 深黄色
              (128, 0, 128),    # - 深紫色
              (0, 128, 128),    # - 深青色
              (255, 128, 0),    # - 橙色
              (128, 255, 0),    # - 青绿色
              (0, 128, 255),    # - 亮蓝色
              (255, 0, 128),    # - 粉红色
              (128, 0, 255),    # - 紫红色
              (0, 255, 128),    # - 青黄色
              (255, 128, 128),  # - 浅红色
              (128, 255, 128),  # - 浅绿色
              (128, 128, 255),  # - 浅蓝色
              (255, 255, 128),  # - 浅黄色
              (255, 128, 255),  # - 浅紫色
              (128, 255, 255),  # - 浅青色
              (192, 0, 0),      # - 深橙色
              (0, 192, 0),      # - 深青绿色
              (0, 0, 192),      # - 深天蓝色
              (192, 192, 0),    # - 深橙黄色
              (192, 0, 192),    # - 深洋红色
              (0, 192, 192),    # - 深孔雀蓝色
              ]


def draw_line(img, line, color, thickness=1):
    pre_point = line[0]
    for i, cur_point in enumerate(line[1:]):
        x1, y1 = int(pre_point[0]), int(pre_point[1])
        x2, y2 = int(cur_point[0]), int(cur_point[1])
        # cv2.line是具有一定宽度的直线,如何获取密集的orients?应该也构建自身的mask?
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, 8)
        cv2.circle(img, (x1, y1), thickness * 2, color, -1)

        pre_point = cur_point
    return img


def draw_bbox(img, bbox, color, thickness=1):
    x1, y1, x2, y2 = [int(cord) for cord in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    return img