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

REGISTER_OP("DecodeLocalRboxes")
    .Attr("region_size: float")
    .Attr("cell_size: float")
    .Input("match_status: int32")
    .Input("local_pred: float32")
		.Input("image_size: int32")
    .Output("decoded_pred: float32")
    .Output("decoded_counts: int32");


template <typename Device, typename T>
class DecodeLocalRboxesOp : public OpKernel {
 public:
  explicit DecodeLocalRboxesOp(OpKernelConstruction* context)
    : OpKernel(context),
      rbox_dim_(5) {
    OP_REQUIRES_OK(context, context->GetAttr("region_size", &region_size_));
    OP_REQUIRES(context, region_size_ > 0,
                errors::InvalidArgument("Expected region_size > 0, get ", region_size_));
    OP_REQUIRES_OK(context, context->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES(context, cell_size_ > 0,
                errors::InvalidArgument("Expected cell_size > 0, get ", cell_size_));
  }

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& match_status = context->input(0);
    const Tensor& local_pred = context->input(1);
		const Tensor& image_size = context->input(2);
    OP_REQUIRES(context, match_status.dims() == 3,
                errors::InvalidArgument("Expected match_status be 3-dimensional, got ",
                                        match_status.shape().DebugString()));
    OP_REQUIRES(context, local_pred.dims() == 4 && local_pred.dim_size(3) == rbox_dim_,
                errors::InvalidArgument("Expected local_pred be 4-d and its last dim is 4, got ",
                                        local_pred.shape().DebugString()));
    OP_REQUIRES(context, image_size.dims() == 1 && image_size.dim_size(0) == 2,
                errors::InvalidArgument("Expected image_size has shape [2], got ",
                                        image_size.shape().DebugString()));

    // allocate output
    const int batch_size = local_pred.dim_size(0);
    const int map_h = local_pred.dim_size(1);
    const int map_w = local_pred.dim_size(2);
    const int n_decoded_pred_max = map_h * map_w;
    Tensor* decoded_pred = nullptr;
    OP_REQUIRES_OK(context,
        context->allocate_output(0, {batch_size, n_decoded_pred_max, rbox_dim_}, &decoded_pred));
    Tensor* decoded_counts = nullptr;
    OP_REQUIRES_OK(context,
        context->allocate_output(1, {batch_size}, &decoded_counts));

    // compute
    DecodePredictionBatch(match_status.tensor<int, 3>(),
                          local_pred.tensor<T, 4>(),
													image_size.tensor<int, 1>(),
                          decoded_pred->tensor<T, 3>(),
                          decoded_counts->tensor<int, 1>());
  }

 private:
  /**
   * @brief Encode groundtruth rboxes to local groundtruths in a batch
   * @param match_status, int tensor [batch, map_h, map_w]
   * @param local_pred, tensor [batch, map_h, map_w, rbox_dim_]
   * @param image_size, int tensor [2]
   * @param decoded_pred, tensor [batch, n_decoded_pred_max, rbox_dim_]
   * @param decoded_counts, int tensor [batch]
   */
  void DecodePredictionBatch(typename TTypes<int, 3>::ConstTensor match_status,
                             typename TTypes<T, 4>::ConstTensor local_pred,
                             typename TTypes<int, 1>::ConstTensor image_size,
                             typename TTypes<T, 3>::Tensor decoded_pred,
                             typename TTypes<int, 1>::Tensor decoded_counts) {
    const int batch_size = local_pred.dimension(0);
    const int map_h = local_pred.dimension(1);
    const int map_w = local_pred.dimension(2);
		const int image_h = image_size(0);
		const int image_w = image_size(1);
    const int n_decoded_pred_max = decoded_pred.dimension(1);

    for (int i = 0; i < batch_size; ++i) {
      const int* match_status_i_data = match_status.data() + i * map_h * map_w;
      const T* local_pred_i_data = local_pred.data() + i * map_h * map_w * rbox_dim_;
      T* decoded_pred_i_data = decoded_pred.data() + i * n_decoded_pred_max * rbox_dim_;
      int* decoded_counts_i_data = decoded_counts.data() + i;
      DecodePredictionExample(match_status_i_data, local_pred_i_data,
                              map_h, map_w, image_h, image_w,
                              decoded_pred_i_data, n_decoded_pred_max,
                              decoded_counts_i_data);
    }
  }

  /**
   * @brief Decode an example
   * @param match_status, int tensor data [map_h, map_w]
   * @param local_pred, tensor data [map_h, map_w, rbox_dim_]
   * @param map_h, map_w, int conv maps size
   * @param image_h, image_w, int image size
   * @param decoded_pred, tensor data [n_decoded_pred_max, rbox_dim_]
   * @param n_decoded_pred_max, int
   * @param decoded_counts, int tensor data [1]
   */
  void DecodePredictionExample(const int* match_status, const T* local_pred,
                               int map_h, int map_w, int image_h, int image_w,
                               T* decoded_pred, int n_decoded_pred_max,
                               int* decoded_counts) {
    // const T step_x = static_cast<T>(image_w) / map_w;
    // const T step_y = static_cast<T>(image_h) / map_h;
    const T step_x = cell_size_;
    const T step_y = cell_size_;
    int decoded_count = 0;
    for (int p = 0; p < map_h * map_w; ++p) {
      const int px = p % map_w;
      const int py = p / map_w;
      const T grid_cx = step_x * (static_cast<T>(px) + 0.5);
      const T grid_cy = step_y * (static_cast<T>(py) + 0.5);
      const int* match_status_p = match_status + p;
      const T* local_pred_p = local_pred + p * rbox_dim_;

      if (match_status_p[0] == 1) {
        LocalDecode(local_pred_p, grid_cx, grid_cy, decoded_pred + decoded_count * rbox_dim_);
        decoded_count++;
      }
    }
    decoded_counts[0] = decoded_count;
  }

  /**
   * @brief Decode a local prediction
   * @param local_pred_p, tensor data [rbox_dim_], (cx, cy, w, h, theta)
   * @param grid_cx, grid_cy, float
   * @param decoded_pred, tensor data [rbox_dim_]
   */
  void LocalDecode(const T* local_pred_p, T grid_cx, T grid_cy, T* decoded_pred) {
    const T eps = 1e-6;

    T encoded_cx = local_pred_p[0];
    T encoded_cy = local_pred_p[1];
    T encoded_width = local_pred_p[2];
    T encoded_height = local_pred_p[3];
    T encoded_theta = local_pred_p[4];

    decoded_pred[0] = encoded_cx * region_size_ + grid_cx;
    decoded_pred[1] = encoded_cy * region_size_ + grid_cy;
    decoded_pred[2] = std::exp(encoded_width) * region_size_ - eps;
    decoded_pred[3] = std::exp(encoded_height) * region_size_ - eps;
    decoded_pred[4] = encoded_theta;
  }

  const int rbox_dim_;
  T region_size_;
  T cell_size_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeLocalRboxes").Device(DEVICE_CPU),
                        DecodeLocalRboxesOp<CPUDevice, float>)
