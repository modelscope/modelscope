#ifndef TENSORFLOW_KERNELS_UTILITIES_H_
#define TENSORFLOW_KERNELS_UTILITIES_H_

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace util {

/**
 * @brief Area of a bounding box. Bboxes representation: x_min, y_min, x_max, y_max.
 */
template <typename T>
T bbox_area(const T* bbox) {
  T width  = std::max(bbox[2] - bbox[0], static_cast<T>(0));
  T height = std::max(bbox[3] - bbox[1], static_cast<T>(0));
  return width * height;
}

/**
 * @brief Calculate the intersection of bbox1 and bbox2
 * @note The output bbox could be invalid (x_min >= x_max || y_min >= y_max)
 */
template <typename T>
void bbox_intersection(const T* bbox1, const T* bbox2, T* bbox_inter) {
  bbox_inter[0] = std::max(bbox1[0], bbox2[0]);
  bbox_inter[1] = std::max(bbox1[1], bbox2[1]);
  bbox_inter[2] = std::min(bbox1[2], bbox2[2]);
  bbox_inter[3] = std::min(bbox1[3], bbox2[3]);
}

/**
 * @brief Calculate the union of bbox1 and bbox2
 */
template <typename T>
void bbox_union(const T* bbox1, const T* bbox2, T* bbox_united) {
  bbox_united[0] = std::min(bbox1[0], bbox2[0]);
  bbox_united[1] = std::min(bbox1[1], bbox2[1]);
  bbox_united[2] = std::max(bbox1[2], bbox2[2]);
  bbox_united[3] = std::max(bbox1[3], bbox2[3]);
}

/**
 * @brief Area of the intersection of two bboxes.
 */
template <typename T>
T bbox_inter_area(const T* bbox1, const T* bbox2) {
  T bbox_inter[4];
  bbox_intersection(bbox1, bbox2, bbox_inter);
  return bbox_area(bbox_inter);
}

/**
 * @brief Area of the union of two bboxes.
 */
template <typename T>
T bbox_union_area(const T* bbox1, const T* bbox2) {
  T bbox_united[4];
  bbox_union(bbox1, bbox2, bbox_united);
  return bbox_area(bbox_united);
}

/**
 * @brief Jaccard overlap of two bboxes.
 */
template <typename T>
T bbox_jaccard_overlap(const T* bbox1, const T* bbox2) {
  const T eps = 1e-6;
  T inter_area = bbox_inter_area(bbox1, bbox2);
  T union_area = bbox_union_area(bbox1, bbox2);
  T jaccard_overlap = inter_area / (union_area + eps);
  return jaccard_overlap;
}

/**
 * @brief Coverage of bbox1 by bbox2
 */
template <typename T>
T bbox_coverage(const T* bbox1, const T* bbox2) {
  const T eps = 1e-6;
  T inter_area = bbox_inter_area(bbox1, bbox2);
  T coverage = inter_area / (bbox_area(bbox1) + eps);
  return coverage;
}

/**
 * @brief Returns true if the center of bbox1 is covered by bbox2
 */
template <typename T>
bool center_covered(const T* bbox1, const T* bbox2) {
  T center_x = 0.5 * (bbox1[0] + bbox1[2]);
  T center_y = 0.5 * (bbox1[1] + bbox1[3]);
  bool center_covered = center_x >= bbox2[0] &&
                        center_y >= bbox2[1] &&
                        center_x <= bbox2[2] &&
                        center_y <= bbox2[3];
  return center_covered;
}

/**
 * @brief Rotate clockwisely around a point in a left-handed coordinate
 */
template <typename T>
void rotate_around(T x, T y, T pivot_x, T pivot_y, T theta, T* rx, T* ry) {
  *rx = std::cos(theta) * (x - pivot_x) - std::sin(theta) * (y - pivot_y) + pivot_x;
  *ry = std::sin(theta) * (x - pivot_x) + std::cos(theta) * (y - pivot_y) + pivot_y;
}

/**
 * @brief Distance of (px, py) to a line defined by two points (x1, y1) and (x2, y2)
 */
template <typename T>
T point_line_distance(const T px, const T py,
                      const T x1, const T y1,
                      const T x2, const T y2) {
  const T eps = 1e-6;
  T dx = x2 - x1;
  T dy = y2 - y1;
  T divider = std::sqrt(dx * dx + dy * dy) + eps;
  T dist = std::abs(px * dy - py * dx + x2 * y1 - y2 * x1) / divider;
  return dist;
}

/**
 * @brief Length of a ray defined by (cx, cy, theta) to a bounding box
 *        (cx, cy) should be inisde the box, otherwise inf is returned
 */
template <typename T>
T ray_length_in_bbox(const T x, const T y, const T theta,
                     const T* bbox) {
  T ray_length = std::numeric_limits<T>::infinity();
  T xmin = bbox[0];
  T ymin = bbox[1];
  T xmax = bbox[2];
  T ymax = bbox[3];

  // check if inside
  bool center_inside = x >= xmin && x <= xmax &&
                       y >= ymin && y <= ymax;
  if (!center_inside) {
    return ray_length;
  }

  // distance to four sides
  // negative values, inf or -inf indicates that ray does not intersect with this side
  // zero indicates that (x, y) is on this side
  T dt = (ymin - y) / std::sin(theta); // dist to top
  T db = (ymax - y) / std::sin(theta); // dist to bottom
  T dl = (xmin - x) / std::cos(theta); // dist to left
  T dr = (xmax - x) / std::cos(theta); // dist to right

  if (dt >= 0) { ray_length = std::min(ray_length, dt); }
  if (db >= 0) { ray_length = std::min(ray_length, db); }
  if (dl >= 0) { ray_length = std::min(ray_length, dl); }
  if (dr >= 0) { ray_length = std::min(ray_length, dr); }
  return ray_length;
}

/**
 * @brief Distance between points (x1, y1) and (x2, y2)
 */
template <typename T>
T point_distance(const T x1, const T y1, const T x2, const T y2) {
  T dx = x2 - x1;
  T dy = y2 - y1;
  T dist = std::sqrt(dx * dx + dy * dy);
  return dist;
}

/**
 * @brief Convert a polygon into a rbox
 */
template <typename T>
void polygon_to_rbox(const T* polygon, T* rbox) {
  T x_tl = polygon[0]; T y_tl = polygon[1];
  T x_tr = polygon[2]; T y_tr = polygon[3];
  T x_br = polygon[4]; T y_br = polygon[5];
  T x_bl = polygon[6]; T y_bl = polygon[7];

  // center is the mean of four polygon vertices
  T cx = (x_tl + x_tr + x_br + x_bl) / 4.;
  T cy = (y_tl + y_tr + y_br + y_bl) / 4.;

  // width is the mean of the top and bottom sides
  // height is (vertical distance from center to top) + (to bottom)
  T dist_tltr = point_distance(x_tl, y_tl, x_tr, y_tr);
  T dist_blbr = point_distance(x_bl, y_bl, x_br, y_br);
  T width = (dist_tltr + dist_blbr) / 2.;
  T dist_center_top = point_line_distance(cx, cy, x_tl, y_tl, x_tr, y_tr);
  T dist_center_bot = point_line_distance(cx, cy, x_bl, y_bl, x_br, y_br);
  T height = dist_center_top + dist_center_bot;

  // theta is the mean of the top and bottom side angles
  T theta1 = std::atan2(y_tr - y_tl, x_tr - x_tl);
  T theta2 = std::atan2(y_br - y_bl, x_br - x_bl);
  T theta = (theta1 + theta2) / 2.;

  rbox[0] = cx;
  rbox[1] = cy;
  rbox[2] = width;
  rbox[3] = height;
  rbox[4] = theta;
}

/**
 * @brief Convert a rbox into a polygon
 */
template <typename T>
void rbox_to_polygon(const T* rbox, T* polygon) {
  T cx = rbox[0];
  T cy = rbox[1];
  T width = rbox[2];
  T height = rbox[3];
  T theta = rbox[4];

  T x_tl = cx - std::cos(theta) * width / 2. + std::sin(theta) * height / 2.;
  T y_tl = cy - std::sin(theta) * width / 2. - std::cos(theta) * height / 2.;
  T x_tr = cx + std::cos(theta) * width / 2. + std::sin(theta) * height / 2.;
  T y_tr = cy + std::sin(theta) * width / 2. - std::cos(theta) * height / 2.;
  T x_br = cx + std::cos(theta) * width / 2. - std::sin(theta) * height / 2.;
  T y_br = cy + std::sin(theta) * width / 2. + std::cos(theta) * height / 2.;
  T x_bl = cx - std::cos(theta) * width / 2. - std::sin(theta) * height / 2.;
  T y_bl = cy - std::sin(theta) * width / 2. + std::cos(theta) * height / 2.;

  polygon[0] = x_tl;
  polygon[1] = y_tl;
  polygon[2] = x_tr;
  polygon[3] = y_tr;
  polygon[4] = x_br;
  polygon[5] = y_br;
  polygon[6] = x_bl;
  polygon[7] = y_bl;
}

/**
 * @brief Check if a point is inside a rbox, also returns the distances to the
          two axes of the rbox
 */
template <typename T>
bool point_inside_rbox(const T* rbox, T x, T y, T* dx, T* dy) {
  T cx = rbox[0];
  T cy = rbox[1];
  T w = rbox[2];
  T h = rbox[3];
  T theta = rbox[4];
  T dist_x = std::abs((cx - x) * std::cos(theta) + (cy - y) * std::sin(theta));
  T dist_y = std::abs((cx - x) * -std::sin(theta) + (cy - y) * std::cos(theta));
  if (dx != nullptr && dy != nullptr) {
    *dx = dist_x;
    *dy = dist_y;
  }
  bool is_inside = (dist_x < w / 2.) && (dist_y < h / 2.);
  return is_inside;
}

/**
 * @brief Check if a point is inside a rbox, also returns the distances to the
          two axes of the rbox
 */
template <typename T>
bool point_inside_rbox_shrink(const T* rbox, T x, T y, T* dx, T* dy) {
  T cx = rbox[0];
  T cy = rbox[1];
  T w = rbox[2];
  T h = rbox[3];
  T theta = rbox[4];
  T dist_x = std::abs((cx - x) * std::cos(theta) + (cy - y) * std::sin(theta));
  T dist_y = std::abs((cx - x) * -std::sin(theta) + (cy - y) * std::cos(theta));
  if (dx != nullptr && dy != nullptr) {
    *dx = dist_x;
    *dy = dist_y;
  }
  bool is_inside = (dist_x < w * 0.4) && (dist_y < h * 0.3);
  return is_inside;
}

template <typename T>
bool point_inside_rbox_match(const T* rbox, T x, T y, T anchor_size, T* dx, T* dy) {
  T cx = rbox[0];
  T cy = rbox[1];
  T w = rbox[2];
  T h = rbox[3];
  T theta = rbox[4];
  T dist_x = std::abs((cx - x));
  T dist_y = std::abs((cy - y));
  if (dx != nullptr && dy != nullptr) {
    *dx = dist_x;
    *dy = dist_y;
  }
  bool is_inside = (dist_x < anchor_size) && (dist_y < anchor_size);
  return is_inside;
}


} // namespace util
} // namespace tensorflow

#endif // TENSORFLOW_KERNELS_UTILITIES_H_
