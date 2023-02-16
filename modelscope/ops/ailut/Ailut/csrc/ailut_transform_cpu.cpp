#include <torch/extension.h>

#include <ATen/ATen.h>


/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}


/* binary search on a sorted array to find and clamp the lower bound */
template <typename scalar_t>
inline int32_t lower_bound(
        const scalar_t *data_ss,
        int32_t start,
        int32_t end,
        scalar_t val) {

    const int32_t ori_start = start;
    const int32_t upper_bound = end - start - 2;
    while (start < end) {
        int64_t mid = start + ((end - start) >> 1);
        if (!(data_ss[mid] >= val)) {
            start = mid + 1;
        }
        else {
            end = mid;
        }
    }
    return clamp(start - ori_start - 1, 0, upper_bound);
}


template <typename scalar_t>
void lut_transform_3d_cpu_forward_impl(
        const int n,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ data_col) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);

    for (int index = 0; index < n; ++index) {

        /* retrieve rgb value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + height * width];
        const scalar_t b = data_inp[index + height * width * 2];

        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid = clamp((int32_t)floor(b * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility variables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;

        /* retrieve the interpolation verticess (number of 8 in case of trilinear interpolation) */
        const int id000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id100 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id010 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id101 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id011 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);
        const int id111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        const scalar_t bd = (b - size_bin * bid) / size_bin;

        const scalar_t w000 = (1 - rd) * (1 - gd) * (1 - bd);
        const scalar_t w100 = (    rd) * (1 - gd) * (1 - bd);
        const scalar_t w010 = (1 - rd) * (    gd) * (1 - bd);
        const scalar_t w110 = (    rd) * (    gd) * (1 - bd);
        const scalar_t w001 = (1 - rd) * (1 - gd) * (    bd);
        const scalar_t w101 = (    rd) * (1 - gd) * (    bd);
        const scalar_t w011 = (1 - rd) * (    gd) * (    bd);
        const scalar_t w111 = (    rd) * (    gd) * (    bd);

        /* Execute the interpolation */
        for (int i = 0; i < num_channels; ++i) {
            data_col[index + height * width * i] =
                w000 * data_lut[id000 + stride_lut_3 * i] + w100 * data_lut[id100 + stride_lut_3 * i] +
                w010 * data_lut[id010 + stride_lut_3 * i] + w110 * data_lut[id110 + stride_lut_3 * i] +
                w001 * data_lut[id001 + stride_lut_3 * i] + w101 * data_lut[id101 + stride_lut_3 * i] +
                w011 * data_lut[id011 + stride_lut_3 * i] + w111 * data_lut[id111 + stride_lut_3 * i];
        }
    }
}


template <typename scalar_t>
void lut_transform_3d_cpu_backward_impl(
        const int n,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ grad_inp,
        scalar_t* __restrict__ grad_lut) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);

    for (int index = 0; index < n; ++index) {

        /* retrieve rgb value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + height * width];
        const scalar_t b = data_inp[index + height * width * 2];

        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid = clamp((int32_t)floor(b * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility variables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;

        /* retrieve the interpolation verticess (number of 8 in case of trilinear interpolation) */
        const int id000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id100 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id010 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id101 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id011 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);
        const int id111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        const scalar_t bd = (b - size_bin * bid) / size_bin;

        const scalar_t w000 = (1 - rd) * (1 - gd) * (1 - bd);
        const scalar_t w100 = (    rd) * (1 - gd) * (1 - bd);
        const scalar_t w010 = (1 - rd) * (    gd) * (1 - bd);
        const scalar_t w110 = (    rd) * (    gd) * (1 - bd);
        const scalar_t w001 = (1 - rd) * (1 - gd) * (    bd);
        const scalar_t w101 = (    rd) * (1 - gd) * (    bd);
        const scalar_t w011 = (1 - rd) * (    gd) * (    bd);
        const scalar_t w111 = (    rd) * (    gd) * (    bd);

        /* derivatives: w to rd */
        const scalar_t w000_rd = - (1 - gd) * (1 - bd);
        const scalar_t w100_rd =   (1 - gd) * (1 - bd);
        const scalar_t w010_rd = - (    gd) * (1 - bd);
        const scalar_t w110_rd =   (    gd) * (1 - bd);
        const scalar_t w001_rd = - (1 - gd) * (    bd);
        const scalar_t w101_rd =   (1 - gd) * (    bd);
        const scalar_t w011_rd = - (    gd) * (    bd);
        const scalar_t w111_rd =   (    gd) * (    bd);

        /* derivatives: w to gd */
        const scalar_t w000_gd = - (1 - rd) * (1 - bd);
        const scalar_t w100_gd = - (    rd) * (1 - bd);
        const scalar_t w010_gd =   (1 - rd) * (1 - bd);
        const scalar_t w110_gd =   (    rd) * (1 - bd);
        const scalar_t w001_gd = - (1 - rd) * (    bd);
        const scalar_t w101_gd = - (    rd) * (    bd);
        const scalar_t w011_gd =   (1 - rd) * (    bd);
        const scalar_t w111_gd =   (    rd) * (    bd);

        /* derivatives: w to bd */
        const scalar_t w000_bd = - (1 - rd) * (1 - gd);
        const scalar_t w100_bd = - (    rd) * (1 - gd);
        const scalar_t w010_bd = - (1 - rd) * (    gd);
        const scalar_t w110_bd = - (    rd) * (    gd);
        const scalar_t w001_bd =   (1 - rd) * (1 - gd);
        const scalar_t w101_bd =   (    rd) * (1 - gd);
        const scalar_t w011_bd =   (1 - rd) * (    gd);
        const scalar_t w111_bd =   (    rd) * (    gd);

        for (int i = 0; i < num_channels; ++i) {
            scalar_t grad_o_ = grad_output[index + width * height * i];

            /* compute gradient of lut */
            grad_lut[id000 + stride_lut_3 * i] += grad_o_ * w000;
            grad_lut[id100 + stride_lut_3 * i] += grad_o_ * w100;
            grad_lut[id010 + stride_lut_3 * i] += grad_o_ * w010;
            grad_lut[id110 + stride_lut_3 * i] += grad_o_ * w110;
            grad_lut[id001 + stride_lut_3 * i] += grad_o_ * w001;
            grad_lut[id101 + stride_lut_3 * i] += grad_o_ * w101;
            grad_lut[id011 + stride_lut_3 * i] += grad_o_ * w011;
            grad_lut[id111 + stride_lut_3 * i] += grad_o_ * w111;

            /* compute gradient of vertices */
            scalar_t grad_d = 0;
            const scalar_t lut000 = data_lut[id000 + stride_lut_3 * i];
            const scalar_t lut100 = data_lut[id100 + stride_lut_3 * i];
            const scalar_t lut010 = data_lut[id010 + stride_lut_3 * i];
            const scalar_t lut110 = data_lut[id110 + stride_lut_3 * i];
            const scalar_t lut001 = data_lut[id001 + stride_lut_3 * i];
            const scalar_t lut101 = data_lut[id101 + stride_lut_3 * i];
            const scalar_t lut011 = data_lut[id011 + stride_lut_3 * i];
            const scalar_t lut111 = data_lut[id111 + stride_lut_3 * i];
            grad_d = grad_o_ *
                (w000_rd * lut000 + w100_rd * lut100 + w010_rd * lut010 + w110_rd * lut110 +
                 w001_rd * lut001 + w101_rd * lut101 + w011_rd * lut011 + w111_rd * lut111);
            // r
            grad_inp[index] += grad_d * 1 / size_bin;

            grad_d = grad_o_ *
                (w000_gd * lut000 + w100_gd * lut100 + w010_gd * lut010 + w110_gd * lut110 +
                 w001_gd * lut001 + w101_gd * lut101 + w011_gd * lut011 + w111_gd * lut111);
            // g
            grad_inp[index + height * width] += grad_d * 1 / size_bin;

            grad_d = grad_o_ *
                (w000_bd * lut000 + w100_bd * lut100 + w010_bd * lut010 + w110_bd * lut110 +
                 w001_bd * lut001 + w101_bd * lut101 + w011_bd * lut011 + w111_bd * lut111);
            // b
            grad_inp[index + height * width * 2] += grad_d * 1 / size_bin;
        }
    }
}


template <typename scalar_t>
void ailut_transform_3d_cpu_forward_impl(
        const int n,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const scalar_t* __restrict__ data_anc,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ data_col) {

    const static scalar_t eps = 1e-10;

    for (int index = 0; index < n; ++index) {

        /* retrieve rgb value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + height * width];
        const scalar_t b = data_inp[index + height * width * 2];

        /* retrieve index of the interpolation verticess */
        const int32_t rid = lower_bound(data_anc, 0, stride_lut, r);
        const int32_t gid = lower_bound(data_anc, stride_lut, stride_lut * 2, g);
        const int32_t bid = lower_bound(data_anc, stride_lut * 2, stride_lut * 3, b);

        /* utility variables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;

        /* retrieve the interpolation verticess (number of 8 in case of trilinear interpolation) */
        const int id000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id100 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id010 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id101 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id011 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);
        const int id111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);

        /* compute interpolation weights */
        const scalar_t r0 = data_anc[rid];
        const scalar_t r1 = data_anc[rid + 1];
        const scalar_t g0 = data_anc[gid + stride_lut];
        const scalar_t g1 = data_anc[gid + stride_lut + 1];
        const scalar_t b0 = data_anc[bid + stride_lut * 2];
        const scalar_t b1 = data_anc[bid + stride_lut * 2 + 1];

        const scalar_t rd = (r - r0) / (r1 - r0 + eps);
        const scalar_t gd = (g - g0) / (g1 - g0 + eps);
        const scalar_t bd = (b - b0) / (b1 - b0 + eps);

        const scalar_t w000 = (1 - rd) * (1 - gd) * (1 - bd);
        const scalar_t w100 = (    rd) * (1 - gd) * (1 - bd);
        const scalar_t w010 = (1 - rd) * (    gd) * (1 - bd);
        const scalar_t w110 = (    rd) * (    gd) * (1 - bd);
        const scalar_t w001 = (1 - rd) * (1 - gd) * (    bd);
        const scalar_t w101 = (    rd) * (1 - gd) * (    bd);
        const scalar_t w011 = (1 - rd) * (    gd) * (    bd);
        const scalar_t w111 = (    rd) * (    gd) * (    bd);

        /* Execute the interpolation */
        for (int i = 0; i < num_channels; ++i) {
            data_col[index + height * width * i] =
                w000 * data_lut[id000 + stride_lut_3 * i] + w100 * data_lut[id100 + stride_lut_3 * i] +
                w010 * data_lut[id010 + stride_lut_3 * i] + w110 * data_lut[id110 + stride_lut_3 * i] +
                w001 * data_lut[id001 + stride_lut_3 * i] + w101 * data_lut[id101 + stride_lut_3 * i] +
                w011 * data_lut[id011 + stride_lut_3 * i] + w111 * data_lut[id111 + stride_lut_3 * i];
        }
    }
}


template <typename scalar_t>
void ailut_transform_3d_cpu_backward_impl(
        const int n,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const scalar_t* __restrict__ data_anc,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ grad_inp,
        scalar_t* __restrict__ grad_lut,
        scalar_t* __restrict__ grad_ver) {

    const static scalar_t eps = 1e-10;

    for (int index = 0; index < n; ++index) {

        /* retrieve rgb value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + height * width];
        const scalar_t b = data_inp[index + height * width * 2];

        /* retrieve index of the interpolation verticess */
        const int32_t rid = lower_bound(data_anc, 0, stride_lut, r);
        const int32_t gid = lower_bound(data_anc, stride_lut, stride_lut * 2, g);
        const int32_t bid = lower_bound(data_anc, stride_lut * 2, stride_lut * 3, b);

        /* utility variables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;

        /* retrieve the interpolation verticess (number of 8 in case of trilinear interpolation) */
        const int id000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id100 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    );
        const int id010 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    );
        const int id001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id101 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1);
        const int id011 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);
        const int id111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1);

        /* compute interpolation weights */
        const scalar_t r0 = data_anc[rid];
        const scalar_t r1 = data_anc[rid + 1];
        const scalar_t g0 = data_anc[gid + stride_lut];
        const scalar_t g1 = data_anc[gid + stride_lut + 1];
        const scalar_t b0 = data_anc[bid + stride_lut * 2];
        const scalar_t b1 = data_anc[bid + stride_lut * 2 + 1];

        const scalar_t rd = (r - r0) / (r1 - r0 + eps);
        const scalar_t gd = (g - g0) / (g1 - g0 + eps);
        const scalar_t bd = (b - b0) / (b1 - b0 + eps);

        const scalar_t w000 = (1 - rd) * (1 - gd) * (1 - bd);
        const scalar_t w100 = (    rd) * (1 - gd) * (1 - bd);
        const scalar_t w010 = (1 - rd) * (    gd) * (1 - bd);
        const scalar_t w110 = (    rd) * (    gd) * (1 - bd);
        const scalar_t w001 = (1 - rd) * (1 - gd) * (    bd);
        const scalar_t w101 = (    rd) * (1 - gd) * (    bd);
        const scalar_t w011 = (1 - rd) * (    gd) * (    bd);
        const scalar_t w111 = (    rd) * (    gd) * (    bd);

        /* derivatives: rd to r/r0/r1 */
        const scalar_t rd_r  =          1 / (r1 - r0 + eps);
        const scalar_t rd_r0 = - (1 - rd) / (r1 - r0 + eps);
        const scalar_t rd_r1 = - (    rd) / (r1 - r0 + eps);
        /* derivatives: gd to g/g0/g1 */
        const scalar_t gd_g  =          1 / (g1 - g0 + eps);
        const scalar_t gd_g0 = - (1 - gd) / (g1 - g0 + eps);
        const scalar_t gd_g1 = - (    gd) / (g1 - g0 + eps);
        /* derivatives: bd to b/b0/b1 */
        const scalar_t bd_b =           1 / (b1 - b0 + eps);
        const scalar_t bd_b0 = - (1 - bd) / (b1 - b0 + eps);
        const scalar_t bd_b1 = - (    bd) / (b1 - b0 + eps);

        /* derivatives: w to rd */
        const scalar_t w000_rd = - (1 - gd) * (1 - bd);
        const scalar_t w100_rd =   (1 - gd) * (1 - bd);
        const scalar_t w010_rd = - (    gd) * (1 - bd);
        const scalar_t w110_rd =   (    gd) * (1 - bd);
        const scalar_t w001_rd = - (1 - gd) * (    bd);
        const scalar_t w101_rd =   (1 - gd) * (    bd);
        const scalar_t w011_rd = - (    gd) * (    bd);
        const scalar_t w111_rd =   (    gd) * (    bd);

        /* derivatives: w to gd */
        const scalar_t w000_gd = - (1 - rd) * (1 - bd);
        const scalar_t w100_gd = - (    rd) * (1 - bd);
        const scalar_t w010_gd =   (1 - rd) * (1 - bd);
        const scalar_t w110_gd =   (    rd) * (1 - bd);
        const scalar_t w001_gd = - (1 - rd) * (    bd);
        const scalar_t w101_gd = - (    rd) * (    bd);
        const scalar_t w011_gd =   (1 - rd) * (    bd);
        const scalar_t w111_gd =   (    rd) * (    bd);

        /* derivatives: w to bd */
        const scalar_t w000_bd = - (1 - rd) * (1 - gd);
        const scalar_t w100_bd = - (    rd) * (1 - gd);
        const scalar_t w010_bd = - (1 - rd) * (    gd);
        const scalar_t w110_bd = - (    rd) * (    gd);
        const scalar_t w001_bd =   (1 - rd) * (1 - gd);
        const scalar_t w101_bd =   (    rd) * (1 - gd);
        const scalar_t w011_bd =   (1 - rd) * (    gd);
        const scalar_t w111_bd =   (    rd) * (    gd);

        for (int i = 0; i < num_channels; ++i) {
            scalar_t grad_o_ = grad_output[index + width * height * i];

            /* compute gradient of lut */
            grad_lut[id000 + stride_lut_3 * i] += grad_o_ * w000;
            grad_lut[id100 + stride_lut_3 * i] += grad_o_ * w100;
            grad_lut[id010 + stride_lut_3 * i] += grad_o_ * w010;
            grad_lut[id110 + stride_lut_3 * i] += grad_o_ * w110;
            grad_lut[id001 + stride_lut_3 * i] += grad_o_ * w001;
            grad_lut[id101 + stride_lut_3 * i] += grad_o_ * w101;
            grad_lut[id011 + stride_lut_3 * i] += grad_o_ * w011;
            grad_lut[id111 + stride_lut_3 * i] += grad_o_ * w111;

            /* compute gradient of vertices */
            scalar_t grad_d = 0;
            const scalar_t lut000 = data_lut[id000 + stride_lut_3 * i];
            const scalar_t lut100 = data_lut[id100 + stride_lut_3 * i];
            const scalar_t lut010 = data_lut[id010 + stride_lut_3 * i];
            const scalar_t lut110 = data_lut[id110 + stride_lut_3 * i];
            const scalar_t lut001 = data_lut[id001 + stride_lut_3 * i];
            const scalar_t lut101 = data_lut[id101 + stride_lut_3 * i];
            const scalar_t lut011 = data_lut[id011 + stride_lut_3 * i];
            const scalar_t lut111 = data_lut[id111 + stride_lut_3 * i];
            grad_d = grad_o_ *
                (w000_rd * lut000 + w100_rd * lut100 + w010_rd * lut010 + w110_rd * lut110 +
                 w001_rd * lut001 + w101_rd * lut101 + w011_rd * lut011 + w111_rd * lut111);
            // r0/r1
            grad_ver[rid    ] += grad_d * rd_r0;
            grad_ver[rid + 1] += grad_d * rd_r1;
            // r
            grad_inp[index] += grad_d * rd_r;

            grad_d = grad_o_ *
                (w000_gd * lut000 + w100_gd * lut100 + w010_gd * lut010 + w110_gd * lut110 +
                 w001_gd * lut001 + w101_gd * lut101 + w011_gd * lut011 + w111_gd * lut111);
            // g0/g1
            grad_ver[stride_lut + gid    ] += grad_d * gd_g0;
            grad_ver[stride_lut + gid + 1] += grad_d * gd_g1;
            // g
            grad_inp[index + height * width] += grad_d * gd_g;

            grad_d = grad_o_ *
                (w000_bd * lut000 + w100_bd * lut100 + w010_bd * lut010 + w110_bd * lut110 +
                 w001_bd * lut001 + w101_bd * lut101 + w011_bd * lut011 + w111_bd * lut111);
            // b0/b1
            grad_ver[stride_lut * 2 + bid    ] += grad_d * bd_b0;
            grad_ver[stride_lut * 2 + bid + 1] += grad_d * bd_b1;
            // b
            grad_inp[index + height * width * 2] += grad_d * bd_b;
        }
    }
}


void lut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);

    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "lut_transform_cpu_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                lut_transform_3d_cpu_forward_impl(
                    num_kernels, data_inp, data_lut,
                    height, width, stride_lut, num_channels,
                    data_col);
            }));
    }
}


void lut_transform_cpu_backward(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);

    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "lut_transform_cpu_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_inp_  = grad_inp[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();

                lut_transform_3d_cpu_backward_impl(
                    num_kernels, grad_out, data_inp, data_lut,
                    height, width, stride_lut, num_channels,
                    grad_inp_, grad_lut_);
            }));
    }
}


void ailut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);

    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "ailut_transform_cpu_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                const scalar_t *data_anc = vertices[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                ailut_transform_3d_cpu_forward_impl(
                    num_kernels, data_inp, data_lut, data_anc,
                    height, width, stride_lut, num_channels,
                    data_col);
            }));
    }
}


void ailut_transform_cpu_backward(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, const torch::Tensor &vertices,
    torch::Tensor grad_inp, torch::Tensor grad_lut, torch::Tensor grad_ver) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);

    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "ailut_transform_cpu_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                const scalar_t *data_anc = vertices[elt].data_ptr<scalar_t>();
                scalar_t *grad_inp_  = grad_inp[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_ver_ = grad_ver[elt].data_ptr<scalar_t>();

                ailut_transform_3d_cpu_backward_impl(
                    num_kernels, grad_out, data_inp, data_lut, data_anc,
                    height, width, stride_lut, num_channels,
                    grad_inp_, grad_lut_, grad_ver_);
            }));
    }
}
