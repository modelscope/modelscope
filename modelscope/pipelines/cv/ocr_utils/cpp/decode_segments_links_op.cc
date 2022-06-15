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

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DecodeSegmentsLinks")
    .Attr("N: int >= 1")
    .Attr("anchor_sizes: list(float)")
    .Input("image_size: int32")
    .Input("all_node_status: N * int32")
    .Input("all_link_status: N * int32")
    .Input("all_reg_maps: N * float32")
    .Output("segments: float32")
    .Output("group_indices: int32")
    .Output("segment_counts: int32")
    .Output("group_indices_all: int32");


template <typename Device, typename T>
class DecodeSegmentsLinksOp : public OpKernel {
  typedef std::array<T, 6> seg_t;

 public:
  explicit DecodeSegmentsLinksOp(OpKernelConstruction* context)
    : OpKernel(context),
      seg_dim_(6),
      offset_dim_(6),
      cross_stride_(2) {
    // anchor sizes
    OP_REQUIRES_OK(context, context->GetAttr("anchor_sizes", &anchor_sizes_));
    const int n_layers = anchor_sizes_.size();
    for (int i = 0; i < n_layers; i++) {
      OP_REQUIRES(context, anchor_sizes_[i] > 0,
                  errors::InvalidArgument("anchor size must be greater than 0"));
    }
  }

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& image_size = context->input(0);
    OpInputList all_node_status;
    OP_REQUIRES_OK(context, context->input_list("all_node_status", &all_node_status));
    OpInputList all_link_status;
    OP_REQUIRES_OK(context, context->input_list("all_link_status", &all_link_status));
    OpInputList all_reg_maps;
    OP_REQUIRES_OK(context, context->input_list("all_reg_maps", &all_reg_maps));

    OP_REQUIRES(context, image_size.dims() == 1 and image_size.dim_size(0) == 2,
                errors::InvalidArgument("Expected image_size has shape [2], got: ",
                                        image_size.shape().DebugString()));
    const int n_layers = all_node_status.size();
    const int batch_size = all_node_status[0].dim_size(0);
    for (int i = 0; i < n_layers; ++i) {
      OP_REQUIRES(context, all_node_status[i].dims() == 3,
                  errors::InvalidArgument("Expected all_node_status[i] be 3-d, got ",
                                          all_node_status[i].shape().DebugString()));
      OP_REQUIRES(context, all_link_status[i].dims() == 4,
                  errors::InvalidArgument("Expected all_link_status[i] be 4-d, got ",
                                          all_link_status[i].shape().DebugString()));
      OP_REQUIRES(context, all_reg_maps[i].dims() == 4 && all_reg_maps[i].dim_size(3) == offset_dim_,
                  errors::InvalidArgument("Expected all_reg_maps[i] has shape [*, *, *, 6], got ",
                                          all_reg_maps[i].shape().DebugString()));
    }

    std::vector<std::array<int, 2>> map_sizes;
    std::vector<int> all_n_links;
    for (int i = 0; i < n_layers; ++i) {
      const int map_h = all_node_status[i].dim_size(1);
      const int map_w = all_node_status[i].dim_size(2);
      map_sizes.push_back({map_h, map_w});
      const int n_links = all_link_status[i].dim_size(3);
      all_n_links.push_back(n_links);
    }

    const auto& image_size_tensor = image_size.tensor<int, 1>();
    const int image_h = image_size_tensor(0);
    const int image_w = image_size_tensor(1);

    std::vector<std::vector<seg_t>> batch_segments;
    std::vector<std::vector<int>> batch_group_indices;
    std::vector<std::vector<int>> batch_group_indices_all;
    int max_count = 0;
    int offsets = 0;

    // decode every example
    for (int i = 0; i < batch_size; ++i) {
      std::vector<const int*> all_node_status_i;
      std::vector<const int*> all_link_status_i;
      std::vector<const T*> all_reg_maps_i;
      for (int j = 0; j < n_layers; ++j) {
        const int map_size_j = map_sizes[j][0] * map_sizes[j][1];
        const int n_links = all_n_links[j];
        const int* node_status_ij = all_node_status[j].tensor<int, 3>().data() +
                                    i * map_size_j;
        const int* link_status_ij = all_link_status[j].tensor<int, 4>().data() +
                                    i * map_size_j * n_links;
        const T* reg_maps_ij = all_reg_maps[j].tensor<T, 4>().data() +
                                i * map_size_j * offset_dim_;
        all_node_status_i.push_back(node_status_ij);
        all_link_status_i.push_back(link_status_ij);
        all_reg_maps_i.push_back(reg_maps_ij);
      }

      std::vector<seg_t> segments;
      std::vector<int> group_indices;
      std::vector<int> group_indices_all;
      DecodeSegmentsLinksExample(all_node_status_i, all_link_status_i, all_reg_maps_i,
                                 map_sizes, image_h, image_w,
                                 &segments, &group_indices, &group_indices_all);
      batch_segments.push_back(segments);
      batch_group_indices.push_back(group_indices);
      batch_group_indices_all.push_back(group_indices_all);
      max_count = std::max(max_count, static_cast<int>(segments.size()));
      offsets = static_cast<int>(group_indices_all.size());
    }

    // output 1
    Tensor* output_segments = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, {batch_size, max_count, seg_dim_}, &output_segments));
    // output 2
    Tensor* output_group_indices = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(1, {batch_size, max_count}, &output_group_indices));
    // output 3
    Tensor* output_counts = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(2, {batch_size}, &output_counts));
    // output 4
    Tensor* output_group_indices_all = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(3, {batch_size, offsets}, &output_group_indices_all));

    auto segments_tensor = output_segments->tensor<T, 3>();
    auto group_indices_tensor = output_group_indices->tensor<int, 2>();
    auto counts_tensor = output_counts->tensor<int, 1>();
    auto group_indices_all_tensor = output_group_indices_all->tensor<int, 2>();
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < batch_segments[i].size(); j++) {
        for (int k = 0; k < seg_dim_; k++) {
          segments_tensor(i,j,k) = batch_segments[i][j][k];
        }
        group_indices_tensor(i,j) = batch_group_indices[i][j];
      }
      counts_tensor(i) = static_cast<int>(batch_segments[i].size());
    }
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < offsets; j++) {
        group_indices_all_tensor(i,j) = batch_group_indices_all[i][j];
      }
    }
  }

  /**
   * @brief Decode segments of an example and find their group indices
   * @param all_node_status, node status of all detection layers 
   * @param all_link_status, link status of all detection layers
   * @param all_reg_maps, regression maps of all detection layers
   * @param map_sizes, map sizes of all layers
   * @param image_h, image_w
   * @param segments, output combined segments
   * @param group_indices, connected subgraph labels
   * @param group_indices_all, connected subgraph labels
   */
  void DecodeSegmentsLinksExample(const std::vector<const int*>& all_node_status,
                                  const std::vector<const int*>& all_link_status,
                                  const std::vector<const T*>& all_reg_maps,
                                  const std::vector<std::array<int, 2>>& map_sizes,
                                  const int image_h, const int image_w,
                                  std::vector<seg_t>* segments,
                                  std::vector<int>* group_indices,
                                  std::vector<int>* group_indices_all) {
    const int n_layers = all_node_status.size();

    // nodes are indexed by id
    // node id offsets of each layer
    std::vector<int> node_id_offsets(n_layers);
    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
      if (layer_idx == 0) {
        node_id_offsets[layer_idx] = 0;
      } else {
        const int below_h = map_sizes[layer_idx-1][0];
        const int below_w = map_sizes[layer_idx-1][1];
        node_id_offsets[layer_idx] = node_id_offsets[layer_idx-1] + below_h * below_w;
      }
    }

    // graph node type
    typedef struct {
      std::vector<int> adjacent_gnodes;
      seg_t segment;
      int group_idx;
    } gnode_t;

    std::map<int, gnode_t> graph; // node_id => adjacent node_ids
    auto add_edge = [&graph](int u, int v) {
      assert(u != v);
      graph[u].adjacent_gnodes.push_back(v);
      graph[v].adjacent_gnodes.push_back(u);
    };

    // add graph nodes
    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
      const int* node_status = all_node_status[layer_idx];
      const T* reg_map = all_reg_maps[layer_idx];
      const int map_h = map_sizes[layer_idx][0];
      const int map_w = map_sizes[layer_idx][1];
      const T rs = anchor_sizes_[layer_idx];
      const T step_x = static_cast<T>(image_w) / map_w;
      const T step_y = static_cast<T>(image_h) / map_h;

      for (int p = 0; p < map_h * map_w; ++p) {
        const int node_status_p = node_status[p];
        if (node_status_p == 1) {
          const int node_id = node_id_offsets[layer_idx] + p;

          // decode local segment
          seg_t segment;
          const int px = p % map_w;
          const int py = p / map_w;
          const T grid_cx = step_x * (px + 0.5);
          const T grid_cy = step_y * (py + 0.5);
          const T* local_reg_p = reg_map + p * offset_dim_;
          T encoded_cx = local_reg_p[0];
          T encoded_cy = local_reg_p[1];
          T encoded_width = local_reg_p[2];
          T encoded_height = local_reg_p[3];
          T encoded_theta_sin = local_reg_p[4];
          T encoded_theta_cos = local_reg_p[5];
          const T eps = 1e-6; // FIXME: move to somewhere else
          segment[0] = encoded_cx * rs + grid_cx;
          segment[1] = encoded_cy * rs + grid_cy;
          segment[2] = std::exp(encoded_width) * rs - eps;
          segment[3] = std::exp(encoded_height) * rs - eps;
          segment[4] = encoded_theta_sin;
          segment[5] = encoded_theta_cos;

          graph[node_id] = {.adjacent_gnodes = std::vector<int>(),
                            .segment = segment,
                            .group_idx = -1};
        }
      }
    }

    // add graph edges
    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
      const int* link_status = all_link_status[layer_idx];

      const bool has_cross_links = layer_idx > 0;
      const int n_local_links = 8;
      const int n_cross_links = 4;
      const int n_links = has_cross_links ? n_local_links + n_cross_links : n_local_links;

      const int map_h = map_sizes[layer_idx][0];
      const int map_w = map_sizes[layer_idx][1];
      for (int p = 0; p < map_h * map_w; ++p) {
        const int node_id = node_id_offsets[layer_idx] + p;
        if (graph.find(node_id) == graph.end()) {
          continue;
        }

        const int* link_status_p = link_status + p * n_links;
        const int px = p % map_w;
        const int py = p / map_w;

        // iterate through same-layer neighbors
        int link_idx = 0;
        for (int ny = py - 1; ny <= py + 1; ++ny) {
          for (int nx = px - 1; nx <= px + 1; ++nx) {
            if (!(nx == px && ny == py)) {
              bool out_of_boundary = ny < 0 || nx < 0 ||
                                     ny >= map_h || nx >= map_w;
              bool positive_link = (link_status_p[link_idx] == 1);
              if (!out_of_boundary && positive_link) { // found a linking neighbor
                const int np = ny * map_w + nx;
                const int neighbor_id = node_id_offsets[layer_idx] + np;
                if (graph.find(neighbor_id) != graph.end()) {
                  add_edge(node_id, neighbor_id);
                }
              }
              link_idx++;
            }
          }
        }

        // iterate through cross-layer neighbors
        if (has_cross_links) {
          const int below_h = map_sizes[layer_idx-1][0];
          const int below_w = map_sizes[layer_idx-1][1];
          int y_start = std::min(cross_stride_ * py, below_h - cross_stride_);
          int y_end = std::min(cross_stride_ * (py + 1), below_h);
          int x_start = std::min(cross_stride_ * px, below_w - cross_stride_);
          int x_end = std::min(cross_stride_ * (px + 1), below_w);

          int cross_link_idx = 0;
          for (int ny = y_start; ny < y_end; ++ny) {
            for (int nx = x_start; nx < x_end; ++nx) {
              bool positive_link = (link_status_p[n_local_links + cross_link_idx] == 1);
              if (positive_link) {
                int np = ny * below_w + nx;
                int neighbor_id = node_id_offsets[layer_idx-1] + np;
                if (graph.find(neighbor_id) != graph.end()) {
                  add_edge(node_id, neighbor_id);
                }
              }
              cross_link_idx++;
            }
          }
        }
      }
    }

    // find connected components in graph
    int group_idx = 0;
    std::function<void (int)> dfs_labelling = [&](int node_id) {
      if (graph[node_id].group_idx != -1) { // already visited
        return;
      }
      graph[node_id].group_idx = group_idx;
      for (int v : graph[node_id].adjacent_gnodes) {
        dfs_labelling(v);
      }
    };
    for (const auto& kv : graph) {
      const int node_id = kv.first;
      dfs_labelling(node_id);
      group_idx++;
    }

    // output
    for (const auto& kv : graph) {
      const auto& v = kv.second;
      segments->push_back(v.segment);
      group_indices->push_back(v.group_idx);
    }

    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
      const int map_h = map_sizes[layer_idx][0];
      const int map_w = map_sizes[layer_idx][1];
      for (int p = 0; p < map_h * map_w; ++p) {
        const int node_id = node_id_offsets[layer_idx] + p;
        if (graph.find(node_id) != graph.end()) {
          group_indices_all->push_back(graph[node_id].group_idx); 
        } else {
          group_indices_all->push_back(-1); 
        }
      }
    }    
  }

 private:
  const int seg_dim_;
  const int offset_dim_;
  std::vector<T> anchor_sizes_;
  int n_max_output_;
  int cross_stride_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeSegmentsLinks").Device(DEVICE_CPU),
                        DecodeSegmentsLinksOp<CPUDevice, float>)
