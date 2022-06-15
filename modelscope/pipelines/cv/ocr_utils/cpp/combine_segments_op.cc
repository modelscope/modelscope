#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>
#include <map>
#include <vector>
#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "utilities.h"

using namespace tensorflow;


REGISTER_OP("CombineSegments")
    .Input("segments: float32")
    .Input("group_indices: int32")
    .Input("seg_counts: int32")
    .Output("combined_rboxes: float32")
    .Output("combined_counts: int32");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CombineSegmentsOp : public OpKernel {
  typedef std::array<T, 5> rbox_t;
  typedef std::array<T, 6> seg_t;
  typedef std::array<T, 2> point_t;

 public:
  explicit CombineSegmentsOp(OpKernelConstruction* context)
    : OpKernel(context),
      seg_dim_(6), rbox_dim_(5) { }

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& segments = context->input(0);
    const Tensor& group_indices = context->input(1);
    const Tensor& seg_counts = context->input(2);

    OP_REQUIRES(context, segments.dims() == 3 && segments.dim_size(2) == seg_dim_,
                errors::InvalidArgument("Expected segments has shape [*, *, 6], got ",
                                        segments.shape().DebugString()));
    OP_REQUIRES(context, group_indices.dims() == 2,
                errors::InvalidArgument("Expected group_indices has shape [*, *], got ",
                                        group_indices.shape().DebugString()));
    OP_REQUIRES(context, segments.dim_size(1) == group_indices.dim_size(1),
                errors::InvalidArgument("segments and group_indices must have the same 2nd dimension size"));
    OP_REQUIRES(context, seg_counts.dims() == 1,
                errors::InvalidArgument("Expected seg_counts has shape [*], got ",
                                        seg_counts.shape().DebugString()));

    std::vector<std::vector<rbox_t>> batch_combined_rboxes;
    BatchCombineSegments(segments.tensor<T, 3>(),
                         group_indices.tensor<int, 2>(),
                         seg_counts.tensor<int, 1>(),
                         &batch_combined_rboxes);

    const int batch_size = segments.dim_size(0);
    int max_count = std::max_element(
      batch_combined_rboxes.begin(), batch_combined_rboxes.end(),
      [](const std::vector<rbox_t>& v1, const std::vector<rbox_t>& v2) {
        return v1.size() < v2.size();
      })->size();

    Tensor* combined_rboxes = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, {batch_size, max_count, rbox_dim_}, &combined_rboxes));
    Tensor* combined_counts = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(1, {batch_size}, &combined_counts));
    auto combined_rboxes_tensor = combined_rboxes->tensor<T, 3>();
    auto combined_counts_tensor = combined_counts->tensor<int, 1>();
    for (int i = 0; i < batch_size; ++i) {
      const auto& rboxes = batch_combined_rboxes[i];
      combined_counts_tensor(i) = rboxes.size();
      for (int j = 0; j < rboxes.size(); ++j) {
        for (int k = 0; k < rbox_dim_; ++k) {
          combined_rboxes_tensor(i, j, k) = rboxes[j][k];
        }
      }
    }
  }

  /**
   * @brief Batch combine local rboxes.
   */
  void BatchCombineSegments(typename TTypes<T, 3>::ConstTensor segments,
                            typename TTypes<int, 2>::ConstTensor group_indices,
                            typename TTypes<int, 1>::ConstTensor seg_counts,
                            std::vector<std::vector<rbox_t>>* batch_combined_rboxes) {
    const int batch_size = segments.dimension(0);
    const int max_count = segments.dimension(1);

    for (int i = 0; i < batch_size; ++i) {
      const T* segments_i = segments.data() + i * max_count * seg_dim_;
      const int* group_indices_i = group_indices.data() + i * max_count;
      const int count = seg_counts(i);
      std::vector<rbox_t> combined_rboxes;
      CombineSegments(segments_i, group_indices_i, count, &combined_rboxes);
      batch_combined_rboxes->push_back(combined_rboxes);
    }
  }

  /**
   * @brief Combine local rboxes in one example.
   */
  void CombineSegments(const T* segments,
                       const int* group_indices,
                       const int count,
                       std::vector<rbox_t>* combined_rboxes) {
    std::map<int, std::vector<seg_t>> groups; // group idx => segments
    for (int i = 0; i < count; ++i) {
      const int group_idx = group_indices[i];
      if (groups.find(group_idx) == groups.end()) {
        groups[group_idx] = std::vector<seg_t>();
      }
      seg_t segment = {segments[i * seg_dim_ + 0],
                       segments[i * seg_dim_ + 1],
                       segments[i * seg_dim_ + 2],
                       segments[i * seg_dim_ + 3],
                       segments[i * seg_dim_ + 4],
                       segments[i * seg_dim_ + 5]};
      groups[group_idx].push_back(segment);
    }

    for (const auto& kv : groups) {
      const std::vector<seg_t>& segments = kv.second;

      // find mx + b
      T combined_theta_cos = std::accumulate(
        segments.begin(), segments.end(), static_cast<T>(0),
        [](T x, const seg_t& seg) { return x + seg[4]; }
        ) / static_cast<T>(segments.size());
      T combined_theta_sin = std::accumulate(
        segments.begin(), segments.end(), static_cast<T>(0),
        [](T x, const seg_t& seg) { return x + seg[5]; }
        ) / static_cast<T>(segments.size());
      //float combined_theta_sin;
      //float combined_theta_cos;
      //if (std::abs(combined_theta_sin_pre) > std::abs(combined_theta_cos_pre)) {
      //    combined_theta_sin =  -1.0;
      //    combined_theta_cos = 0.000001;
      //}
      //else {
      //    combined_theta_sin = 0.0;
      //    combined_theta_cos = 1.0;
      //}
      const T m = combined_theta_sin / combined_theta_cos;
      const T combined_theta = std::atan(m);

      const T b = std::accumulate(
        segments.begin(), segments.end(), static_cast<T>(0),
        [m](T x, const seg_t& r) { return x + r[1] - m * r[0]; }
        ) / static_cast<T>(segments.size());

      // project all points to mx + b
      std::vector<point_t> proj_points;
      for (const seg_t& seg : segments) {
        T proj_x = (m * seg[1] + seg[0] - m * b) / (m * m + 1);
        T proj_y = (m * m * seg[1] + m * seg[0] + b) / (m * m + 1);
        T dx = seg[2]*cos(combined_theta)/2;
        T dy = seg[3]*sin(combined_theta)/2;
        proj_points.push_back({proj_x-dx, proj_y-dy});
        proj_points.push_back({proj_x+dx, proj_y+dy});
      }

      // find end points by finding the min and max of
      // the inner products of (cx, cy) and (1, m)
      std::vector<T> inner_products(2*segments.size());
      for (int i = 0; i < (int)segments.size(); ++i) {
        T cx = segments[i][0];
        T cy = segments[i][1];
        T dx = segments[i][2]*cos(combined_theta)/2;
        T dy = segments[i][3]*sin(combined_theta)/2;
        inner_products[2*i] = cx-dx + m * (cy-dy);
        inner_products[2*i+1] = cx+dx + m * (cy+dy);
      }
      int argmin_ip = std::distance(inner_products.begin(),
        std::min_element(inner_products.begin(), inner_products.end()));
      int argmax_ip = std::distance(inner_products.begin(),
        std::max_element(inner_products.begin(), inner_products.end()));
      const point_t& endpoint_1 = proj_points[argmin_ip];
      const point_t& endpoint_2 = proj_points[argmax_ip];

      // combined rbox center
      const T combined_cx = 0.5 * (endpoint_1[0] + endpoint_2[0]);
      const T combined_cy = 0.5 * (endpoint_1[1] + endpoint_2[1]);

      // combined rbox width and height
      T combined_w = util::point_distance<T>(endpoint_1[0], endpoint_1[1],
                                             endpoint_2[0], endpoint_2[1]);

      const T combined_h = std::accumulate(
        segments.begin(), segments.end(), static_cast<T>(0),
        [](T x, const seg_t& s) { return x + s[3]; }
        ) / static_cast<T>(segments.size());

      rbox_t combined_rbox = {combined_cx, combined_cy, combined_w, combined_h, combined_theta};
      combined_rboxes->push_back(combined_rbox);
    }
  }

private:
  const int seg_dim_;
  const int rbox_dim_;
};

REGISTER_KERNEL_BUILDER(Name("CombineSegments").Device(DEVICE_CPU),
                        CombineSegmentsOp<CPUDevice, float>)
