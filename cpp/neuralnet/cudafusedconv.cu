// KataGame fork (2026-11-10, the engine repository's abscls-rvu1): CUTLASS convolutions with a residual block's
// scale, bias and ReLU in their epilogues (see cudafusedconv.h).
//
// The kernel is CUTLASS 2.x's DefaultConv2dFpropWithBroadcast, whose epilogue (EpilogueWithBroadcast) gives its functor
// the accumulators, the source tensor C and a vector broadcast over the output channels, and stores two outputs: Z (the
// destination) and, optionally, T. The functor below computes, in float:
//   without the source:  Z = relu(scale * acc + bias)
//   with the source:     T = acc + C, Z = relu(scale * T + bias)
// The mainloop is that of the kernel cuDNN chooses for KataGo's 3x3 convolutions of 128 channels in FP16 (the GPU
// profile of 2026-11-10: cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x3_nhwc_align8): threadblock
// 128x128x32, warp 64x64x32, mma 16x8x16, 3 stages, FP32 accumulation, the optimized iterators.

#include "../neuralnet/cudafusedconv.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"

namespace {

using Half = cutlass::half_t;

// The functor of EpilogueWithBroadcast (the concept of cutlass/epilogue/thread/linear_combination_bias_elementwise.h).
// The broadcast vector is the packed words of cudafusedconv.h's packScaleBias, declared as float: the epilogue loads
// it as ElementVector and converts it to ElementCompute, and a float to float conversion is a copy, so each element
// arrives with its 32 bits unchanged and is split here into the scale's and the bias's FP16 bits.
template <bool AddSource>
class ScaleBiasReluOp {
public:
  using ElementOutput = Half;
  using ElementD = ElementOutput;
  using ElementC = Half;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;
  using ElementZ = Half;
  using ElementT = Half;
  using ElementVector = float;
  static int const kElementsPerAccess = 8;
  static int const kCount = kElementsPerAccess;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
  using FragmentSource = FragmentC;
  using FragmentOutput = FragmentZ;

  static bool const kIsSingleSource = true;
  static bool const kIsHeavy = false;
  static bool const kStoreZ = true;
  static bool const kStoreT = AddSource;  // T is the trunk, stored only when the source (the old trunk) is added

  struct Params {};

  CUTLASS_HOST_DEVICE
  explicit ScaleBiasReluOp(Params const&) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return AddSource; }

  // Split-K is not used (one slice), so there is no partial sum to skip.
  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  // With the source: T = acc + C, Z = relu(scale * T + bias).
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& fragZ, FragmentT& fragT, FragmentAccumulator const& acc, FragmentC const& fragC,
                  FragmentCompute const& scaleBias) const {
    FragmentCompute sum;
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kElementsPerAccess; ++i)
      sum[i] = acc[i] + float(fragC[i]);
    apply(fragZ, fragT, sum, scaleBias);
  }

  // Without the source: Z = relu(scale * acc + bias).
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& fragZ, FragmentT& fragT, FragmentAccumulator const& acc,
                  FragmentCompute const& scaleBias) const {
    apply(fragZ, fragT, acc, scaleBias);
  }

private:
  CUTLASS_HOST_DEVICE
  static void apply(FragmentZ& fragZ, FragmentT& fragT, FragmentCompute const& sum, FragmentCompute const& scaleBias) {
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kElementsPerAccess; ++i) {
      uint32_t word;
      memcpy(&word, &scaleBias[i], sizeof(word));
      const float scale = float(Half::bitcast((uint16_t)(word & 0xFFFFu)));
      const float bias = float(Half::bitcast((uint16_t)(word >> 16)));
      const float z = scale * sum[i] + bias;
      fragZ[i] = Half(z > 0.0f ? z : 0.0f);  // as KataGo's ReLU kernels: a NaN becomes 0
      if constexpr(kStoreT)
        fragT[i] = Half(sum[i]);
    }
  }
};

template <bool AddSource>
using ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
  Half, cutlass::layout::TensorNHWC,
  Half, cutlass::layout::TensorNHWC,
  Half, cutlass::layout::TensorNHWC,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  ScaleBiasReluOp<AddSource>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

template <bool AddSource>
using Conv = cutlass::conv::device::ImplicitGemmConvolution<ConvKernel<AddSource>>;

// The arguments of one convolution. trunk is the source and T for AddSource, else unused (nullptr).
template <bool AddSource>
typename Conv<AddSource>::Arguments makeArgs(
  const half* in, const half* filter, half* trunk, const uint32_t* scaleBias, half* out,
  int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize
) {
  cutlass::conv::Conv2dProblemSize problem(
    batchSize, ySize, xSize, inChannels, outChannels, convYSize, convXSize, ySize, xSize,
    convYSize / 2, convXSize / 2, 1, 1, 1, 1, cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  using Layout = cutlass::layout::TensorNHWC;
  const Layout inLayout = Layout::packed({batchSize, ySize, xSize, inChannels});
  const Layout filterLayout = Layout::packed({outChannels, convYSize, convXSize, inChannels});
  const Layout outLayout = Layout::packed({batchSize, ySize, xSize, outChannels});
  return typename Conv<AddSource>::Arguments(
    problem,
    {(Half*)in, inLayout},
    {(Half*)filter, filterLayout},
    {(Half*)(AddSource ? trunk : out), outLayout},
    {(Half*)out, outLayout},
    typename ScaleBiasReluOp<AddSource>::Params(),
    cutlass::conv::SplitKMode::kSerial,
    (void*)scaleBias,
    AddSource ? (void*)trunk : nullptr,
    0,              // ldr: one vector for every row
    outChannels);   // ldt: T's row stride, as the output's
}

template <bool AddSource>
bool implementable(typename Conv<AddSource>::Arguments const& args) {
  return Conv<AddSource>::can_implement(args) == cutlass::Status::kSuccess
    && Conv<AddSource>::get_workspace_size(args) == 0;
}

template <bool AddSource>
cutlass::Status runConv(typename Conv<AddSource>::Arguments const& args, cudaStream_t stream) {
  Conv<AddSource> op;
  cutlass::Status status = op.initialize(args, nullptr, stream);
  if(status == cutlass::Status::kSuccess)
    status = op.run(stream);
  return status;
}

void requireAligned(const void* a, const void* b, const void* c, const void* d, const void* scaleBias) {
  if((uintptr_t(a) | uintptr_t(b) | uintptr_t(c) | uintptr_t(d)) & 15 || uintptr_t(scaleBias) & 31)
    throw std::runtime_error("CudaFusedConv: operand pointers must be 16-byte aligned (the scale and bias 32-byte)");
}

void throwIfFailed(cutlass::Status status, const char* what, int batchSize, int ySize, int xSize, int inChannels,
                   int outChannels) {
  if(status == cutlass::Status::kSuccess)
    return;
  // The caller checked supportsShape at model load, so this is a real failure (a CUDA error, or an earlier
  // asynchronous error on this stream that run()'s cudaGetLastError reports), never a reason to fall back.
  throw std::runtime_error(
    std::string("CudaFusedConv::") + what + " failed (or a prior CUDA error was pending): " +
    cutlass::cutlassGetStatusString(status) + ", batch " + std::to_string(batchSize) + ", board " +
    std::to_string(xSize) + "x" + std::to_string(ySize) + ", channels " + std::to_string(inChannels) + " -> " +
    std::to_string(outChannels));
}

}  // namespace

namespace CudaFusedConv {

bool supportedOnCurrentDevice() {
  // Never launch below sm_80: CUTLASS's below-arch path is a device-side trap, a sticky error (cudafusedffn.cu).
  {
    int device = 0;
    int major = 0;
    if(cudaGetDevice(&device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
      return false;
    if(major < 8)
      return false;
  }
  // Both kernels on a tiny convolution with zero inputs and filters: with scale 1 and bias 0.5, the first writes 0.5;
  // the second adds 0 to a trunk of 1 and writes 1.5. The outputs start as NaN, so a kernel that did not run (an
  // empty JIT stub) fails.
  constexpr int N = 1, Y = 4, X = 4, C = 32, K = 32, R = 3, S = 3;
  constexpr size_t numIn = (size_t)N * Y * X * C, numFilter = (size_t)K * R * S * C, numOut = (size_t)N * Y * X * K;
  half* buf = nullptr;
  uint32_t* scaleBias = nullptr;
  if(cudaMalloc(&buf, (numIn + numFilter + 3 * numOut) * sizeof(half)) != cudaSuccess)
    return false;
  bool ok = cudaMalloc(&scaleBias, K * sizeof(uint32_t)) == cudaSuccess;
  half* in = buf;
  half* filter = in + numIn;
  half* out1 = filter + numFilter;
  half* trunk = out1 + numOut;
  half* out2 = trunk + numOut;
  {
    uint32_t words[K];
    for(int k = 0; k < K; k++)
      words[k] = packScaleBias(0x3C00u, 0x3800u);  // FP16's 1.0 and 0.5
    half ones[numOut];
    for(size_t i = 0; i < numOut; i++)
      ones[i] = __float2half(1.0f);
    ok = ok && cudaMemcpy(scaleBias, words, sizeof(words), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemset(in, 0, (numIn + numFilter) * sizeof(half)) == cudaSuccess;
    ok = ok && cudaMemset(out1, 0xFF, numOut * sizeof(half)) == cudaSuccess;
    ok = ok && cudaMemcpy(trunk, ones, sizeof(ones), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemset(out2, 0xFF, numOut * sizeof(half)) == cudaSuccess;
  }
  if(ok) {
    auto args1 = makeArgs<false>(in, filter, nullptr, scaleBias, out1, N, Y, X, C, K, R, S);
    auto args2 = makeArgs<true>(in, filter, trunk, scaleBias, out2, N, Y, X, C, K, R, S);
    ok = implementable<false>(args1) && implementable<true>(args2);
    ok = ok && runConv<false>(args1, nullptr) == cutlass::Status::kSuccess;
    ok = ok && runConv<true>(args2, nullptr) == cutlass::Status::kSuccess;
  }
  half host[3 * numOut];
  ok = ok && cudaMemcpy(host, out1, sizeof(host), cudaMemcpyDeviceToHost) == cudaSuccess;
  if(ok) {
    for(size_t i = 0; i < numOut; i++) {
      if(__half2float(host[i]) != 0.5f || __half2float(host[numOut + i]) != 1.0f
         || __half2float(host[2 * numOut + i]) != 1.5f) {
        ok = false;
        break;
      }
    }
  }
  // Clear a recoverable launch error so that it cannot leak into later, unrelated CUDA calls.
  (void)cudaGetLastError();
  (void)cudaFree(buf);
  (void)cudaFree(scaleBias);
  return ok;
}

bool supportsShape(int inChannels, int outChannels, int convYSize, int convXSize) {
  if(inChannels <= 0 || outChannels <= 0 || inChannels % 8 != 0 || outChannels % 8 != 0)
    return false;
  if(convYSize <= 0 || convXSize <= 0 || convYSize % 2 != 1 || convXSize % 2 != 1)
    return false;
  // can_implement depends on the channels, the kernel's size and the operands' alignment, not on the batch or the
  // board. The dummy pointers carry the alignment that the backend's device allocations give.
  half* dummy = reinterpret_cast<half*>(uintptr_t(256));
  const uint32_t* dummyScaleBias = reinterpret_cast<const uint32_t*>(uintptr_t(256));
  const int Y = 2 * convYSize, X = 2 * convXSize;
  return implementable<false>(
           makeArgs<false>(dummy, dummy, nullptr, dummyScaleBias, dummy, 1, Y, X, inChannels, outChannels, convYSize,
                           convXSize))
    && implementable<true>(
           makeArgs<true>(dummy, dummy, dummy, dummyScaleBias, dummy, 1, Y, X, inChannels, outChannels, convYSize,
                          convXSize));
}

void runConvScaleBiasRelu(
  const half* in, const half* filter, const uint32_t* scaleBias, half* out,
  int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
  cudaStream_t stream
) {
  requireAligned(in, filter, out, out, scaleBias);
  auto args = makeArgs<false>(in, filter, nullptr, scaleBias, out, batchSize, ySize, xSize, inChannels, outChannels,
                              convYSize, convXSize);
  throwIfFailed(runConv<false>(args, stream), "runConvScaleBiasRelu", batchSize, ySize, xSize, inChannels,
                outChannels);
}

void runConvAddScaleBiasRelu(
  const half* in, const half* filter, half* trunk, const uint32_t* scaleBias, half* next,
  int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
  cudaStream_t stream
) {
  requireAligned(in, filter, trunk, next, scaleBias);
  auto args = makeArgs<true>(in, filter, trunk, scaleBias, next, batchSize, ySize, xSize, inChannels, outChannels,
                             convYSize, convXSize);
  throwIfFailed(runConv<true>(args, stream), "runConvAddScaleBiasRelu", batchSize, ySize, xSize, inChannels,
                outChannels);
}

}  // namespace CudaFusedConv
