#ifndef CONFIG_H_
#define CONFIG_H_
#define USE_P7_MODEL 1

const static int g_image_w = 640;  // original image size
const static int g_image_h = 480;  // original image size
#ifdef USE_G3_MODEL
/*face detecton model parameter (face detecton)*/
const static int g_fd_num_calss = 2;  // just two class(have face or not)
const static int g_fd_out_layer_num = 3;
const static int g_fd_output_size[3] = {24, 12, 6};
/*filter and nms parameter (face detecton)*/
const static float g_fd_conf_thresh = 0.7;
const static float g_fd_nms_thresh = 0.3;
const static int g_fd_top_k = 20;      // number of bboxs before NMS . the g_conf_thresh will filter out the most of the
                                       // bbox,and the experiment showed the number of left is smaller than 13.
const static int g_fd_keep_top_k = 3;  // number of bboxs after NMS
/* priorbox parameter (face detecton)*/
const static int g_fd_min_size[3] = {12, 24, 48};
const static int g_fd_steps[3] = {4, 8, 16};
const static int g_fd_input_w = 96;  // face detecton model input size
const static int g_fd_input_h = 96;  // face detecton model input size
/* decode bbox parameter (face detecton)*/
const static float g_fd_prior_variance[2] = {0.1, 0.2};
/*landmark model parameter(landmark)*/
const static int g_lm_input_w = 64;  // landmark model input size
const static int g_lm_input_h = 64;  // landmark model input size
const static int g_lm_number = 106;  // the number of landmarks in the landmark model output
#endif

#ifdef USE_P7_MODEL
/*face detecton model parameter (face detecton)*/
const static int g_fd_num_calss = 2;  // just two class(have face or not)
const static int g_fd_out_layer_num = 1;
const static int g_fd_output_size[2] = {13, 20};
/*filter and nms parameter (face detecton)*/
const static float g_fd_conf_thresh = 0.7;
const static float g_fd_nms_thresh = 0.4;
const static int g_fd_top_k = 20;  // number of bboxs before NMS . the g_fd_conf_thresh will filter out the most of the
                                   // bbox,and the experiment showed the number of left is smaller than 13.
const static int g_fd_keep_top_k = 3;  // number of bboxs after NMS
/* priorbox parameter (face detecton)*/
const static int g_fd_min_size[2] = {64, 128};
const static int g_fd_steps[2] = {16, 16};
const static int g_fd_input_w = 320;  // face detecton model input size
const static int g_fd_input_h = 200;  // face detecton model input size
/* decode bbox parameter (face detecton)*/
const static float g_fd_prior_variance[2] = {0.1, 0.2};
/*landmark model parameter(landmark)*/
const static int g_lm_input_w = 224;  // landmark model input size
const static int g_lm_input_h = 224;  // landmark model input size
const static int g_lm_number = 106;   // the number of landmarks in the landmark model output
#endif

#endif  // CONFIG_H_