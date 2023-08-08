#include "dms_cuda_utils.h"
#include "dms_cuda_kernel.cuh"
#include <mutex>
#include <cassert>

DmsISPBase::DmsISPBase(const int img_height, const int img_width, 
                    const int input_h, const int input_w, void *pstream):
                    img_height_(img_height), img_width_(img_width), input_h_(input_h), input_w_(input_w)
{
    block_.x = block_size_;
    block_.y = 1;
    block_.z = 1;
    int max_in_size = input_h_*input_w_;
    CUDA_CHECK(cudaMalloc((void **)&resize_tmp_, max_in_size * sizeof(uint8_t)));
    //CUDA_CHECK(cudaMalloc((void **)&chw16_gpu_, max_in_size * 16 * sizeof(half)));
    //CUDA_CHECK(cudaMemset(chw16_gpu_, 0, max_in_size * 16 * sizeof(half)));
    assert(pstream);
    stream_ = reinterpret_cast<cudaStream_t>(pstream);
    //CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    // Set stream to npp lib and get NppStreamContext
   {
    static std::mutex m_mutex; // static mutex for npp stream & context
    std::lock_guard<std::mutex> lock(m_mutex); //protect "nppSetStream"
    auto old_nppstream = nppGetStream();
    NppStatus nppstatus = nppSetStream(stream_);
    nppstatus = nppGetStreamContext(&nppStream_); 
    if (nppstatus != NPP_SUCCESS) {
        printf("nppGetStreamContext failed!");
    }
    nppSetStream( old_nppstream );
   } 
}

DmsISPBase::~DmsISPBase()
{
    //cudaFree(chw16_gpu_);
    cudaFree(resize_tmp_);
}

void DmsISPBase::CHWInt8ToCHW16Half(const uint8_t *input_gpu, void *output_gpu)
{
    
    grid_.x = (input_w_ * input_h_ - 1) / block_size_ + 1;
    grid_.y = 1;
    grid_.z = 1;
    HWInt8ToCHW16HalfKernel<<<grid_,block_,0,stream_>>>(input_gpu,(half *)output_gpu, input_w_,input_h_);
    
    //CUDA_CHECK(cudaMemcpyAsync((void*)output_cpu, (void*)chw16_gpu_, out_h * out_w *16* sizeof(half), cudaMemcpyDeviceToHost, stream_));
    //cudaStreamSynchronize(stream_);
}

DmsDetectionISP::DmsDetectionISP(const int img_height, const int img_width, const int fd_input_h, const int fd_input_w, void *pstream):
                                DmsISPBase(img_height, img_width, fd_input_h, fd_input_w, pstream){}
                             
void DmsDetectionISP::Resize(const uint8_t* input_gpu, int in_h, int in_w,int out_h, int out_w) {


    NppiRect in_roi{0, 0, in_w, in_h};
    NppiRect out_roi{0, 0, out_w, out_h};
    NppiSize in_size{in_w, in_h};
    NppiSize out_size{out_w, out_h};

    NppStatus result = nppiResize_8u_C1R_Ctx(input_gpu, in_w, in_size, in_roi, resize_tmp_,
        out_w, out_size, out_roi, NPPI_INTER_LINEAR,nppStream_);//NPPI_INTER_SUPER NPPI_INTER_LINEAR
    if (result != NPP_SUCCESS) {
        std::cout << "Unable to run nppiResize_8u_C1R in Resize, error " << result << std::endl;
    }
}

void DmsDetectionISP::resizeAndToCHW16Half(const uint8_t *input_gpu, int in_h, int in_w, int out_h, int out_w, void *output_gpu)
{
    Resize(input_gpu, in_h, in_w, out_h, out_w);
    CHWInt8ToCHW16Half(resize_tmp_, output_gpu);
}

DmsLandmarkISP::DmsLandmarkISP(const int img_height, const int img_width, const int lm_input_h, const int lm_input_w, 
    void *pstream):DmsISPBase(img_height, img_width, lm_input_h, lm_input_w, pstream){}

void DmsLandmarkISP::CropAndResize(const uint8_t *input_gpu, int in_h, int in_w, int out_h, int out_w, 
        int x1, int y1, int x2, int y2)
{
    NppiRect in_roi{x1, y1, x2-x1, y2-y1};
    NppiRect out_roi{0, 0, out_w, out_h};
    NppiSize in_size{in_w, in_h};
    NppiSize out_size{out_w, out_h};

    NppStatus result = nppiResize_8u_C1R_Ctx(input_gpu, in_w, in_size, in_roi, resize_tmp_,
        out_w, out_size, out_roi, NPPI_INTER_LINEAR,nppStream_);//NPPI_INTER_SUPER NPPI_INTER_LINEAR
    if (result != NPP_SUCCESS) {
        std::cout << "Unable to run nppiResize_8u_C1R in CropAndResize, error " << result << std::endl;
    }
}

void DmsLandmarkISP::CropResizeAndToCHW16Half(const uint8_t *input_gpu, void *output_cpu, int in_h, int in_w, 
    int out_h, int out_w, int x1, int y1, int x2, int y2)
{
    CropAndResize(input_gpu, in_h, in_w, out_h, out_w, x1, y1, x2, y2);
    CHWInt8ToCHW16Half(resize_tmp_, output_cpu);
}




