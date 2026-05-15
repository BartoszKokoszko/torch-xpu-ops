#include <ATen/ATen.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

namespace {

constexpr const char* kDevice = "xpu";
constexpr at::ScalarType kDtype = at::kFloat;
constexpr int64_t kBatchSize = 2;
constexpr int64_t kSeed = 0;

const std::vector<int64_t> kStride = {2, 2, 2};
const std::vector<int64_t> kPadding = {0, 0, 0};
const std::vector<int64_t> kDilation = {1, 1, 1};
const bool kTransposed = false;
const std::vector<int64_t> kOutputPadding = {0, 0, 0};
constexpr int64_t kGroups = 1;

at::Tensor reshape_dim_into(int64_t src, int64_t dst, const at::Tensor& x) {
  const auto x_dim = x.dim();
  src = at::maybe_wrap_dim(src, x_dim);
  dst = at::maybe_wrap_dim(dst, x_dim - 1);

  std::vector<int64_t> new_shape(x.sizes().begin(), x.sizes().end());
  const int64_t src_size = new_shape[src];
  new_shape.erase(new_shape.begin() + src);
  new_shape[dst] *= src_size;

  return x.movedim(src, dst).reshape(new_shape);
}

at::Tensor reshape_dim_outof(int64_t src, int64_t size1, const at::Tensor& x) {
  src = at::maybe_wrap_dim(src, x.dim());
  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());

  if (shape[src] != 0) {
    TORCH_CHECK(shape[src] % size1 == 0, "shape[src] must be divisible by size1");
  }
  const int64_t size2 = shape[src] == 0 ? 0 : shape[src] / size1;
  shape[src] = size1;
  shape.insert(shape.begin() + src + 1, size2);

  return x.reshape(shape);
}

at::Tensor make_dummy(
    const at::Tensor& tensor,
    std::optional<int64_t> tensor_bdim,
    int64_t dim,
    int64_t batch_size) {
  auto t = tensor_bdim.has_value() ? tensor.select(*tensor_bdim, 0) : tensor;
  const int64_t orig_size = t.size(dim);
  t = t.narrow(dim, 0, 1);

  std::vector<int64_t> expand_shape(t.sizes().begin(), t.sizes().end());
  expand_shape[dim] = batch_size * orig_size;

  return t.new_empty({}).expand(expand_shape);
}

at::Tensor batch_last(const at::Tensor& tensor, int64_t batch_size) {
  auto t = tensor.unsqueeze(-1);
  std::vector<int64_t> sizes(t.sizes().begin(), t.sizes().end());
  sizes.back() = batch_size;
  return t.expand(sizes).contiguous();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> make_inputs(const at::Device& device) {
  at::manual_seed(kSeed);
  const auto opts = at::TensorOptions().device(device).dtype(kDtype);

  auto x = at::randn({1, 1, 4, 4, 4}, opts);
  auto w = at::randn({1, 1, 4, 4, 4}, opts);
  auto b = at::randn({1}, opts);
  auto grad_out = at::randn({1, 1, 1, 1, 1}, opts);
  return {x, w, b, grad_out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> loop_reference_grads(
    const at::Tensor& bx,
    const at::Tensor& bw,
    const at::Tensor& bb,
    const at::Tensor& bg,
    int64_t batch_size) {
  std::vector<at::Tensor> grad_inputs;
  std::vector<at::Tensor> grad_weights;
  std::vector<at::Tensor> grad_biases;
  grad_inputs.reserve(batch_size);
  grad_weights.reserve(batch_size);
  grad_biases.reserve(batch_size);

  for (int64_t i = 0; i < batch_size; ++i) {
    auto x_i = bx.select(-1, i);
    auto w_i = bw.select(-1, i);
    auto b_i = bb.select(-1, i);
    auto g_i = bg.select(-1, i);

    auto grad_input_i = at::empty_like(x_i);
    auto grad_weight_i = at::empty_like(w_i);
    auto grad_bias_i = at::empty_like(b_i);

    auto data_event = at::native::onednn::convolution_backward_data(
      grad_input_i,
      g_i,
      w_i,
      kPadding,
      kPadding,
      kStride,
      kDilation,
      kGroups,
      true);

    auto wgrad_event = at::native::onednn::convolution_backward_weights(
      grad_weight_i,
      grad_bias_i,
      g_i,
      x_i,
      w_i.sizes(),
      kPadding,
      kPadding,
      kStride,
      kDilation,
      kGroups);

    data_event.wait();
    wgrad_event.wait();

    grad_inputs.push_back(grad_input_i);
    grad_weights.push_back(grad_weight_i);
    grad_biases.push_back(grad_bias_i);
  }

  return {
      at::stack(grad_inputs),
      at::stack(grad_weights),
      at::stack(grad_biases),
  };
}

std::tuple<at::Tensor, int64_t> simulate_convolution_backward_input_batch_rule(
    const at::Tensor& grad_output,
    int64_t grad_output_bdim,
    const at::Tensor& input,
    std::optional<int64_t> input_bdim,
    const at::Tensor& weight,
    int64_t weight_bdim) {
  // Matches branch:
  // if (grad_output_bdim && weight_bdim) { ... }
  const int64_t batch_size = weight.size(weight_bdim);

  const auto grad_output_ = reshape_dim_into(grad_output_bdim, 1, grad_output);
  const auto weight_ = reshape_dim_into(weight_bdim, 0, weight);
  const auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);

    auto grad_input_flat = at::empty_like(dummy_input);
    auto event = at::native::onednn::convolution_backward_data(
      grad_input_flat,
      grad_output_,
      weight_,
      kPadding,
      kPadding,
      kStride,
      kDilation,
      kGroups * batch_size,
      false);
    event.wait();

    auto grad_input = reshape_dim_outof(1, batch_size, grad_input_flat);

  // C++ batching rule returns (grad_input, bdim=1)
  return {grad_input, 1};
}

void print_sizes_and_strides(const std::string& name, const at::Tensor& t) {
  std::cout << name << " sizes=" << t.sizes() << " strides=" << t.strides() << "\n";
}

} // namespace

int main() {
  const at::Device device(kDevice);
  if (device.type() == at::kXPU && !at::xpu::is_available()) {
    std::cerr << "XPU is not available\n";
    return 1;
  }

  auto [x, w, b, grad_out] = make_inputs(device);

  auto bx = batch_last(x, kBatchSize);
  auto bw = batch_last(w, kBatchSize);
  auto bb = batch_last(b, kBatchSize);
  auto bg = batch_last(grad_out, kBatchSize);

  auto [loop_grad_input, loop_grad_weight, loop_grad_bias] =
      loop_reference_grads(bx, bw, bb, bg, kBatchSize);

  auto [simulated_tensor, simulated_bdim] =
      simulate_convolution_backward_input_batch_rule(
          bg, -1, bx, -1, bw, -1);

  auto simulated_batch_first = simulated_tensor.movedim(simulated_bdim, 0);

  print_sizes_and_strides("loop_grad_input", loop_grad_input);
  print_sizes_and_strides("simulated_tensor", simulated_tensor);
  print_sizes_and_strides("simulated_batch_first", simulated_batch_first);
  print_sizes_and_strides("loop_grad_weight", loop_grad_weight);
  print_sizes_and_strides("loop_grad_bias", loop_grad_bias);

  auto folded_grad_output = reshape_dim_into(-1, 1, bg);
  print_sizes_and_strides("folded_grad_output", folded_grad_output);

  const bool ok = at::allclose(simulated_batch_first, loop_grad_input, 1e-4, 1e-4);
  if (!ok) {
    std::cerr << "FAIL: simulated lowering does not match loop_out grad_input\n";
    return 2;
  }

  std::cout << "PASS: simulated lowering matches loop_out grad_input\n";
  return 0;
}
