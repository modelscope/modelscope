#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "utilities.h"

using namespace tensorflow;

REGISTER_OP("PolygonsToRboxes")
    .Input("polygons: float32")
    .Output("rboxes: float32");


template <typename T>
class PolygonsToRboxesOp : public OpKernel {
 public:
  explicit PolygonsToRboxesOp(OpKernelConstruction* context)
    : OpKernel(context),
      polygon_dim_(8),
      rbox_dim_(5) {}

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& polygons = context->input(0);
    OP_REQUIRES(context, polygons.dims() == 2,
                errors::InvalidArgument("polygons must be 2-dimensional",
                                        polygons.shape().DebugString()));
    OP_REQUIRES(context, polygons.dim_size(1) == polygon_dim_,
                errors::InvalidArgument("the last dimension of polygons must be 8"));

    const int n_polygons = polygons.dim_size(0);

    // allocate output
    Tensor* rboxes = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {n_polygons, rbox_dim_}, &rboxes));

    // compute
    PolygonsToRboxes(polygons.tensor<T, 2>(), rboxes->tensor<T, 2>());
  }

 private:
  /**
   * @brief Convert polygons to rboxes
   * @param polygons, tensor [n_polygons, 8]
   * @param rboxes, tensor [n_polygons, 5]
   */
  void PolygonsToRboxes(typename TTypes<T, 2>::ConstTensor polygons,
                        typename TTypes<T, 2>::Tensor rboxes) {
    const int n_polygons = polygons.dimension(0);

    const T* polygons_data = polygons.data();
    T* rboxes_data = rboxes.data();
    for (int i = 0; i < n_polygons; ++i) {
      util::polygon_to_rbox(polygons_data + i * polygon_dim_,
                            rboxes_data + i * rbox_dim_);
    }
  }

  const int polygon_dim_;
  const int rbox_dim_;
};

REGISTER_KERNEL_BUILDER(Name("PolygonsToRboxes").Device(DEVICE_CPU),
                        PolygonsToRboxesOp<float>)
