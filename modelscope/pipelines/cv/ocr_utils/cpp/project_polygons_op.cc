#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

#include "utilities.h"

using namespace tensorflow;

REGISTER_OP("ProjectPolygons")
  .Input("polygons: float32")
  .Input("crop_bbox: float32")
  .Input("image_size: int32")
  .Output("proj_polygons: float32")
  .Output("valid_mask: bool")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    using namespace shape_inference;

    // polygons must have shape [n>=0, 8]
    ShapeHandle polygons_shape = c->input(0);
    TF_RETURN_IF_ERROR(c->WithRank(polygons_shape, 2, &polygons_shape));
    DimensionHandle polygon_dim = c->Dim(polygons_shape, 1);
    DimensionHandle n_polygon = c->Dim(polygons_shape, 0);
    TF_RETURN_IF_ERROR(c->WithValue(polygon_dim, 8, &polygon_dim));

    // output-0 proj_polygons has shape [n, 8]
    c->set_output(0, polygons_shape);

    // output-1 valid_mask has shape [n]
    c->set_output(1, c->MakeShape({n_polygon}));
    
    return Status::OK();
  });


template <typename T>
class ProjectPolygonsOp : public OpKernel {
public:
  explicit ProjectPolygonsOp(OpKernelConstruction* context):
    OpKernel(context), polygon_dim_(8), rbox_dim_(5), bbox_dim_(4) {}

  void Compute(OpKernelContext* context) override {
    // input-0 polygons
    const Tensor& polygons = context->input(0);
    OP_REQUIRES(context, polygons.dims() == 2,
                errors::InvalidArgument("Expected polygons be 2-dimensional, got ",
                                        polygons.shape().DebugString()));
    OP_REQUIRES(context, polygons.dim_size(1) == polygon_dim_,
                errors::InvalidArgument("Expected the last dimension of polygons be 8, got ",
                                        polygons.dim_size(1)));

    // input-1 crop_bbox
    const Tensor& crop_bbox = context->input(1);
    OP_REQUIRES(context, crop_bbox.dims() == 1,
                errors::InvalidArgument("Expected crop_bbox be 1-dimensional, got ",
                                        crop_bbox.shape().DebugString()));
    OP_REQUIRES(context, crop_bbox.dim_size(0) == bbox_dim_,
                errors::InvalidArgument("Expected the last dimension of crop_bbox be 4, got "));

    // input-2 image_size
    const Tensor& image_size = context->input(2);
    OP_REQUIRES(context, image_size.dims() == 1 && image_size.dim_size(0) == 2,
                errors::InvalidArgument("Expected image_size has shape [2], got ",
                                        image_size.shape().DebugString()));

    const int n_polygons = polygons.dim_size(0);

    // output-0 proj_polygons
    Tensor* proj_polygons = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {n_polygons, polygon_dim_}, &proj_polygons));

    // output-1 valid_mask
    Tensor* valid_mask = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {n_polygons}, &valid_mask));

    // compute
    ProjectPolygons(polygons.tensor<T, 2>(),
                    crop_bbox.tensor<T, 1>(),
                    image_size.tensor<int, 1>(),
                    proj_polygons->tensor<T, 2>(),
                    valid_mask->tensor<bool, 1>());
  }

private:
  /**
   * @brief Convert polygons to proj_polygons
   * @param polygons, tensor [n_polygons, 8]
   * @param crop_bbox, tensor [4]
   * @param proj_polygons, tensor [n_polygons, 8]
   */
  void ProjectPolygons(typename TTypes<T, 2>::ConstTensor polygons,
                       typename TTypes<T, 1>::ConstTensor crop_bbox,
                       typename TTypes<int, 1>::ConstTensor resize_size,
                       typename TTypes<T, 2>::Tensor proj_polygons,
                       typename TTypes<bool, 1>::Tensor valid_mask) {
    const T eps = 1e-6;
    const int n_polygons = polygons.dimension(0);
    const T crop_xmin = crop_bbox(0);
    const T crop_ymin = crop_bbox(1);
    const T crop_xmax = crop_bbox(2);
    const T crop_ymax = crop_bbox(3);
    const T crop_width = crop_xmax - crop_xmin;
    const T crop_height = crop_ymax - crop_ymin;
    const int resize_h = resize_size(0);
    const int resize_w = resize_size(1);

    auto project_x = [crop_xmin, crop_width, resize_w, eps](T x) {
        return (x - crop_xmin) / (crop_width + eps) * static_cast<T>(resize_w - 1); };
    auto project_y = [crop_ymin, crop_height, resize_h, eps](T y) {
        return (y - crop_ymin) / (crop_height + eps) * static_cast<T>(resize_h - 1); };
    auto proj_point_inside = [resize_h, resize_w](T x, T y) {
        return (x >= 0 && x <= resize_w - 1 && y >= 0 && y <= resize_h - 1); };

    for (int i = 0; i < n_polygons; ++i) {
      T x1 = project_x(polygons(i, 0));
      T y1 = project_y(polygons(i, 1));
      T x2 = project_x(polygons(i, 2));
      T y2 = project_y(polygons(i, 3));
      T x3 = project_x(polygons(i, 4));
      T y3 = project_y(polygons(i, 5));
      T x4 = project_x(polygons(i, 6));
      T y4 = project_y(polygons(i, 7));

      // valid if at least 1 vertices is inside
      int n_vertices_inside = static_cast<int>(proj_point_inside(x1, y1)) +
                              static_cast<int>(proj_point_inside(x2, y2)) +
                              static_cast<int>(proj_point_inside(x3, y3)) +
                              static_cast<int>(proj_point_inside(x4, y4));
      bool valid = n_vertices_inside >= 1;

      proj_polygons(i, 0) = x1;
      proj_polygons(i, 1) = y1;
      proj_polygons(i, 2) = x2;
      proj_polygons(i, 3) = y2;
      proj_polygons(i, 4) = x3;
      proj_polygons(i, 5) = y3;
      proj_polygons(i, 6) = x4;
      proj_polygons(i, 7) = y4;
      valid_mask(i) = valid;
    }
  }

  const int polygon_dim_;
  const int rbox_dim_;
  const int bbox_dim_;
};

REGISTER_KERNEL_BUILDER(Name("ProjectPolygons").Device(DEVICE_CPU),
                        ProjectPolygonsOp<float>)
