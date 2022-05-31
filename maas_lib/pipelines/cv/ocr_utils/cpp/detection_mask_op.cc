#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "utilities.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DetectionMask")
  .Input("object_mask: bool")
  .Input("local_rboxes: float32")
  .Input("local_counts: int32")
  .Output("updated_mask: bool");


template <typename Device, typename T>
class DetectionMaskOp : public OpKernel {
 public:
  explicit DetectionMaskOp(OpKernelConstruction* context)
    : OpKernel(context),
      rbox_dim_(5) {}

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& object_mask = context->input(0);
    const Tensor& local_rboxes = context->input(1);
		const Tensor& local_counts = context->input(2);
    OP_REQUIRES(context, object_mask.dims() == 3,
                errors::InvalidArgument("Expected object_mask be 3-dimensional, got ",
                                        object_mask.shape().DebugString()));
    OP_REQUIRES(context, local_rboxes.dims() == 3 && local_rboxes.dim_size(2) == rbox_dim_,
                errors::InvalidArgument("Expected local_rboxes be 3-d and its last dim is 5, got ",
                                        local_rboxes.shape().DebugString()));
    OP_REQUIRES(context, local_counts.dims() == 1,
                errors::InvalidArgument("Expected local_counts be 1-dimensional, got ",
                                        local_counts.shape().DebugString()));

    // allocate output
    const int batch_size = object_mask.dim_size(0);
    const int mask_h = object_mask.dim_size(1);
    const int mask_w = object_mask.dim_size(2);
    Tensor* updated_mask = nullptr;
    OP_REQUIRES_OK(context,
        context->allocate_output(0, {batch_size, mask_h, mask_w}, &updated_mask));

    // compute
    UpdateDetectionMask(object_mask.tensor<bool, 3>(),
                        local_rboxes.tensor<T, 3>(),
												local_counts.tensor<int, 1>(),
                        updated_mask->tensor<bool, 3>());
  }

 private:
  void UpdateDetectionMask(typename TTypes<bool, 3>::ConstTensor object_mask,
                           typename TTypes<T, 3>::ConstTensor local_rboxes,
                           typename TTypes<int, 1>::ConstTensor local_counts,
                           typename TTypes<bool, 3>::Tensor updated_mask) {
    const int batch_size = object_mask.dimension(0);
    const int map_h = object_mask.dimension(1);
    const int map_w = object_mask.dimension(2);
    const int n_max_rboxes = local_rboxes.dimension(1);

    for (int i = 0; i < batch_size; ++i) {
      int local_count = local_counts(i);
      for (int p = 0; p < map_h * map_w; ++p) {
        int px = p % map_w;
        int py = p / map_w;
        bool mask = false;
        if (object_mask(i, py, px) == true) {
          mask = true;
        } else {
          T x = px + 0.5;
          T y = py + 0.5;
          for (int j = 0; j < local_count; ++j) {
            const T* local_rbox = local_rboxes.data() + (i * n_max_rboxes + j) * rbox_dim_;
            T* dx = nullptr;
            T* dy = nullptr;
            bool inside = util::point_inside_rbox(local_rbox, x, y, dx, dy);
            if (inside) {
              mask = true;
              break;
            }
          }
        }
        updated_mask(i, py, px) = mask;
      }
    }
  }

  const int rbox_dim_;
};

REGISTER_KERNEL_BUILDER(Name("DetectionMask").Device(DEVICE_CPU),
                        DetectionMaskOp<CPUDevice, float>)
