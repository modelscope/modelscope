#include <cmath>
#include <random>
#include <chrono>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

#include "utilities.h"

using namespace tensorflow;

REGISTER_OP("SampleCropBbox")
  .Input("image_size: int32")
  .Input("bboxes: float32")
  .Output("crop_bbox: float32")
  .Output("success: bool")
  .Attr("overlap_mode: string = 'jaccard'")
  .Attr("min_overlap: float = 0.1")
  .Attr("aspect_ratio_range: list(float) = [0.5, 2.0]")
  .Attr("scale_ratio_range: list(float) = [0.3, 1.0]")
  .Attr("max_trials: int = 50")
  .Attr("seed: int = -1")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    using namespace shape_inference;

    // bboxes should be in shape [n>=0, 4]
    ShapeHandle bbox_shape = c->input(1);
    TF_RETURN_IF_ERROR(c->WithRank(bbox_shape, 2, &bbox_shape));
    DimensionHandle box_dim = c->Dim(bbox_shape, 1);
    TF_RETURN_IF_ERROR(c->WithValue(box_dim, 4, &box_dim));

    // crop_bbox shape [4]
    ShapeHandle crop_bbox_shape;
    crop_bbox_shape = c->MakeShape({4});
    c->set_output(0, crop_bbox_shape);

    // success []
    ShapeHandle success_shape;
    success_shape = c->MakeShape({});
    c->set_output(1, success_shape);

    return Status::OK();
  });


template <typename T>
class SampleCropBboxOp : public OpKernel {
public:
  enum OverlapMode {
     Jaccard,
     Coverage
  };

  explicit SampleCropBboxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    // get attribute overlap_mode
    std::string overlap_mode;
    OP_REQUIRES_OK(context, context->GetAttr("overlap_mode", &overlap_mode));
    OP_REQUIRES(context, overlap_mode == "jaccard" || overlap_mode == "coverage",
                errors::InvalidArgument("overlap_mode must be 'jaccard' or 'coverage', got ",
                                        overlap_mode));
    if (overlap_mode == "jaccard") {
      overlap_mode_ = OverlapMode::Jaccard;
    } else {
      overlap_mode_ = OverlapMode::Coverage;
    }

    // get attribute min_overlap
    OP_REQUIRES_OK(context, context->GetAttr("min_overlap", &min_overlap_));
    OP_REQUIRES(context, min_overlap_ >= 0 && min_overlap_ <= 1,
                errors::InvalidArgument("min_overlap must be in range [0, 1], input value: ",
                                        min_overlap_));

    // get attributes aspect_ratio_range
    OP_REQUIRES_OK(context, context->GetAttr("aspect_ratio_range", &aspect_ratio_range_));
    OP_REQUIRES(context, aspect_ratio_range_[0] > 0 && aspect_ratio_range_[1] > 0,
                errors::InvalidArgument("aspect_ratio_range must be positive"));
    OP_REQUIRES(context, aspect_ratio_range_[1] >= aspect_ratio_range_[0],
                errors::InvalidArgument("aspect_ratio_range max < min"));

    // get attribute scale_ratio_range
    OP_REQUIRES_OK(context, context->GetAttr("scale_ratio_range", &scale_ratio_range_));
    OP_REQUIRES(context,
                scale_ratio_range_[0] >= 0 &&
                scale_ratio_range_[0] <= 1 &&
                scale_ratio_range_[1] >= 0 &&
                scale_ratio_range_[1] <= 1,
                errors::InvalidArgument("scale_ratio_range must be in range [0, 1]"));

    // get attribute max_trials
    OP_REQUIRES_OK(context, context->GetAttr("max_trials", &max_trials_));
    OP_REQUIRES(context, max_trials_ >= 0,
                errors::InvalidArgument("max_trials must be non-negative, input value: ",
                                        max_trials_));

    // get attribute seed
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES(context, seed_ >= -1,
                errors::InvalidArgument("seed must be >= -1, got ", seed_));

    // set seed for the random generator
    if (seed_ == -1) {
      seed_ = std::chrono::system_clock::now().time_since_epoch().count();
    }
    generator_.seed(seed_);
  }

  void Compute(OpKernelContext* context) override {
    // read inputs
    const Tensor& image_size = context->input(0);
    OP_REQUIRES(context, image_size.dims() == 1 && image_size.dim_size(0) == 2,
                errors::InvalidArgument("image_size shape must be [2], got ",
                                        image_size.shape().DebugString()));
    const Tensor& bboxes = context->input(1);
    OP_REQUIRES(context, bboxes.dims() == 2 && bboxes.dim_size(1) == 4,
                errors::InvalidArgument("bboxes must be in shape [*, 4], got ",
                                        bboxes.shape().DebugString()));

    // allocate outputs
    Tensor* crop_bbox = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {4}, &crop_bbox));
    Tensor* success = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &success));

    SampleRandomCropBbox(image_size.tensor<int, 1>(),
                         bboxes.tensor<T, 2>(),
                         crop_bbox->tensor<T, 1>(),
                         success->scalar<bool>());
  }

private:
  void SampleRandomCropBbox(typename TTypes<int, 1>::ConstTensor image_size,
                            typename TTypes<T, 2>::ConstTensor bboxes,
                            typename TTypes<T, 1>::Tensor crop_bbox,
                            typename TTypes<bool>::Scalar success) {
    int image_h = image_size(0);
    int image_w = image_size(1);
    int shorter_side = std::min(image_h, image_w);
    int n_bboxes = bboxes.dimension(0);
    const T* bboxes_data = bboxes.data();
    T* crop_bbox_data = crop_bbox.data();
    crop_bbox_data[0] = static_cast<T>(0);
    crop_bbox_data[1] = static_cast<T>(0);
    crop_bbox_data[2] = static_cast<T>(0);
    crop_bbox_data[3] = static_cast<T>(0);

    T min_size = shorter_side * scale_ratio_range_[0];
    T max_size = shorter_side * scale_ratio_range_[1];

    std::vector<T> crop_bbox_vec(4, 0);
    bool found = false;
    for (int i = 0; i < max_trials_; ++i) {
      GenerateRandomCrop(static_cast<T>(image_h), static_cast<T>(image_w),
                         min_size, max_size, &crop_bbox_vec[0]);
      found = SatisfiesConstraints(&crop_bbox_vec[0], bboxes_data, n_bboxes);
      if (found) {
        crop_bbox_data[0] = crop_bbox_vec[0];
        crop_bbox_data[1] = crop_bbox_vec[1];
        crop_bbox_data[2] = crop_bbox_vec[2];
        crop_bbox_data[3] = crop_bbox_vec[3];
        break;
      }
    }

    success(0) = found;
  }

  void GenerateRandomCrop(T image_h, T image_w, T min_size, T max_size,
                          T* crop_bbox) {
    auto random_uniform = [&](T a, T b) {
        return std::uniform_real_distribution<T>(a, b)(generator_); };

    T shorter_side = std::min(image_h, image_w);
    T size = random_uniform(min_size, max_size);
    T min_ar = std::max<T>(aspect_ratio_range_[0],
                           std::pow(size / shorter_side , 2));
    T max_ar = std::min<T>(aspect_ratio_range_[1],
                           1.0 / std::pow(size / shorter_side, 2));
    T ar = random_uniform(min_ar, max_ar);
    T bbox_width = size * std::sqrt(ar);
    T bbox_height = size / std::sqrt(ar);
    T bbox_x = random_uniform(0.0, image_w - 1 - bbox_width);
    T bbox_y = random_uniform(0.0, image_h - 1 - bbox_height);

    auto restrict_x = [image_w](T x) {
      return std::max(static_cast<T>(0), std::min(image_w - 1, x)); };
    auto restrict_y = [image_h](T y) {
      return std::max(static_cast<T>(0), std::min(image_h - 1, y)); };
    crop_bbox[0] = restrict_x(bbox_x);
    crop_bbox[1] = restrict_y(bbox_y);
    crop_bbox[2] = restrict_x(bbox_x + bbox_width);
    crop_bbox[3] = restrict_y(bbox_y + bbox_height);
  }

  bool SatisfiesConstraints(const T* crop_bbox, const T* object_bboxes,
                            int n_object) {
    bool found = false;
    for (int i = 0; i < n_object; ++i) {
      const T* bboxes_i = object_bboxes + 4 * i;
      T overlap = -std::numeric_limits<T>::infinity();
      switch (overlap_mode_) {
      case OverlapMode::Jaccard:
        overlap = util::bbox_jaccard_overlap(crop_bbox, bboxes_i);
        break;
      case OverlapMode::Coverage:
        overlap = util::bbox_coverage(bboxes_i, crop_bbox);
        break;
      default:
        break;
      }
      if (overlap >= min_overlap_) {
        found = true;
        break;
      }
    }
    return found;
  }

  std::default_random_engine generator_;
  int max_trials_;
  std::vector<T> scale_ratio_range_;
  std::vector<T> aspect_ratio_range_;
  T min_overlap_;
  OverlapMode overlap_mode_;
  int seed_;
};

REGISTER_KERNEL_BUILDER(Name("SampleCropBbox").Device(DEVICE_CPU),
                        SampleCropBboxOp<float>)