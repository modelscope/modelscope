#include <torch/extension.h>

/* CUDA Forward Declarations */

void LutTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


void LutTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


void AiLutTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut,
    const torch::Tensor &vertices, torch::Tensor output);


void AiLutTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, const torch::Tensor &vertices,
    torch::Tensor grad_inp, torch::Tensor grad_lut, torch::Tensor grad_ver);


void lut_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    LutTransformForwardCUDAKernelLauncher(input, lut, output);
}


void lut_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    LutTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


void ailut_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output) {

    AiLutTransformForwardCUDAKernelLauncher(input, lut, vertices, output);
}


void ailut_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut,
    torch::Tensor grad_ver) {

    AiLutTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, vertices, grad_inp, grad_lut, grad_ver);
}


void lut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output);


void lut_transform_cpu_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut);


void ailut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output);


void ailut_transform_cpu_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut,
    torch::Tensor grad_ver);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void lut_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(output);

        lut_transform_cuda_forward(input, lut, output);
    } else {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(output);

        lut_transform_cpu_forward(input, lut, output);
    }
}


void lut_transform_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);

        lut_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    } else {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);

        lut_transform_cpu_backward(grad_output, input, lut, grad_inp, grad_lut);
    }
}


void ailut_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(vertices);
        CHECK_INPUT(output);

        ailut_transform_cuda_forward(input, lut, vertices, output);
    } else {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(vertices);
        CHECK_CONTIGUOUS(output);

        ailut_transform_cpu_forward(input, lut, vertices, output);
    }
}


void ailut_transform_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut,
    torch::Tensor grad_ver) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(vertices);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);
        CHECK_INPUT(grad_ver);

        ailut_transform_cuda_backward(grad_output, input, lut, vertices, grad_inp, grad_lut, grad_ver);
    } else {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(vertices);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);
        CHECK_CONTIGUOUS(grad_ver);

        ailut_transform_cpu_backward(grad_output, input, lut, vertices, grad_inp, grad_lut, grad_ver);
    }
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lut_cforward", &lut_transform_forward, "Lut-Transform forward");
  m.def("lut_cbackward", &lut_transform_backward, "Lut-Transform backward");
  m.def("ailut_cforward", &ailut_transform_forward, "AiLut-Transform forward");
  m.def("ailut_cbackward", &ailut_transform_backward, "AiLut-Transform backward");
}
