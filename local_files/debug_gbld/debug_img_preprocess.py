import cv2
import mmcv
import numpy as np
import mmengine.fileio as fileio


def mmcv_img_load():
    filename = "/home/dell/liyongjing/test_data/debug_model_test.jpg"
    file_client_args = dict(backend='disk')
    color_type = 'color'

    # if self.file_client_args is not None:
    file_client = fileio.FileClient.infer_client(file_client_args, filename)
    img_bytes = file_client.get(filename)

    img = mmcv.imfrombytes(img_bytes, flag=color_type)
    print(img.shape)


def img_pre_process(self, cv_img):
    # 采用numpy实现图像预处理的过程(yolo系列, 待处理)
    assert cv_img is not None
    h_scale = self.net_h / cv_img.shape[0]
    w_scale = self.net_w / cv_img.shape[1]

    scale = min(h_scale, w_scale)
    img_temp = cv2.resize(cv_img, (int(cv_img.shape[1] * scale), int(cv_img.shape[0] * scale)),
                          interpolation=cv2.INTER_LINEAR)

    # cal pad_w, and pad_h
    pad_h = (self.net_h - img_temp.shape[0]) // 2
    pad_w = (self.net_w - img_temp.shape[1]) // 2
    pad_top, pad_bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    pad_left, pad_right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    img_input = np.ones((self.net_h, self.net_w, 3), dtype=np.uint8) * 114
    # img_input[self.pad_h:img_temp.shape[0] + self.pad_h, self.pad_w:img_temp.shape[1] + self.pad_w, :] = img_temp
    img_input[pad_top:img_temp.shape[0] + pad_bottom, pad_left:img_temp.shape[1] + pad_right, :] = img_temp

    # Convert
    img_input = img_input.astype(np.float32)
    img_input = img_input[:, :, ::-1]
    img_input /= 255.0
    img_input = img_input.transpose(2, 0, 1)  # to C, H, W
    img_input = np.ascontiguousarray(img_input)
    img_input = np.expand_dims(img_input, 0)


if __name__ == "__main__":
    print("Start")
    mmcv_img_load()
    print("End")
