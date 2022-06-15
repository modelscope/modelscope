#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "utilities.h"

using namespace tensorflow;


REGISTER_OP("ClipRboxes")
    .Input("rboxes: float32")
    .Input("crop_bbox: float32")
    .Output("clipped_rboxes: float32");

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ClipRboxesOp : public OpKernel {
 public:
  explicit ClipRboxesOp(OpKernelConstruction* context)
    : OpKernel(context),
      rbox_dim_(5) {}

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& rboxes = context->input(0);
    OP_REQUIRES(context, rboxes.dims() == 2 && rboxes.dim_size(1) == rbox_dim_,
                errors::InvalidArgument("rboxes must be 2-dimensional and the second dim must be 5",
                                        rboxes.shape().DebugString()));
    const Tensor& crop_bbox = context->input(1);
    OP_REQUIRES(context, crop_bbox.dims() == 1 && crop_bbox.dim_size(0) == 4,
                errors::InvalidArgument("crop_bbox must be 1-dimensional, 4 elements",
                                        crop_bbox.shape().DebugString()));

    // allocate output
    const int n_rboxes = rboxes.dim_size(0);
    Tensor* clipped_rboxes = nullptr;
    OP_REQUIRES_OK(context,
        context->allocate_output(0, {n_rboxes, rbox_dim_}, &clipped_rboxes));

    // compute
    ClipRboxes(rboxes.tensor<T, 2>(),
               crop_bbox.tensor<T, 1>(),
               clipped_rboxes->tensor<T, 2>());
  }

 private:
  /**
   * @brief Clip rboxes by rectangular crop_bbox
   * @param rboxes, tensor [n_rboxes, 5]
   * @param crop_bbox, tensor [4]
   * @param clipped_rboxes, tensor [n_rboxes, 4]
   */
  void ClipRboxes(typename TTypes<T, 2>::ConstTensor rboxes,
                  typename TTypes<T, 1>::ConstTensor crop_bbox,
                  typename TTypes<T, 2>::Tensor clipped_rboxes) {
    const T pi = std::atan(1) * 4;
    const T zero = static_cast<T>(0);
    const T eps = 1e-6;

    const int n_rboxes = rboxes.dimension(0);
    const T* rboxes_data = rboxes.data();
    for (int i = 0; i < n_rboxes; ++i) {
      const T* rbox = rboxes_data + i * rbox_dim_;
      T* clipped_rbox = clipped_rboxes.data() + i * rbox_dim_;
      T cx = rbox[0];
      T cy = rbox[1];
      T width = rbox[2];
      T height = rbox[3];
      T theta = rbox[4];

      // calculate the new width
      T half_width = width / 2.;
      T len_rhalf = util::ray_length_in_bbox(cx, cy, theta, crop_bbox.data());
      T len_lhalf = util::ray_length_in_bbox(cx, cy, pi + theta, crop_bbox.data());
      T clipped_width = std::min(half_width, len_rhalf) + std::min(half_width, len_lhalf);

      // move center point
      T clipped_cx, clipped_cy;
      if (clipped_width < width - eps) {
        T move_rhalf = (len_rhalf < half_width) ? (len_rhalf - half_width) / 2. : zero;
        T move_lhalf = (len_lhalf < half_width) ? (half_width - len_lhalf) / 2. : zero;
        T move_len = move_rhalf + move_lhalf;
        clipped_cx = cx + std::cos(theta) * move_len;
        clipped_cy = cy + std::sin(theta) * move_len;
      } else {
        clipped_cx = cx;
        clipped_cy = cy;
      }

      clipped_rbox[0] = clipped_cx;
      clipped_rbox[1] = clipped_cy;
      clipped_rbox[2] = clipped_width;
      clipped_rbox[3] = height;
      clipped_rbox[4] = theta;
    }
  }

 const int rbox_dim_;
};

REGISTER_KERNEL_BUILDER(Name("ClipRboxes").Device(DEVICE_CPU),
                        ClipRboxesOp<CPUDevice, float>)
