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
//
// The FP8 kernels (2026-11-10, abscls-jtbd) follow the FP16 ones: the same convolutions with E4M3 inputs and filters
// (Fp8ScaleBiasReluOp, Fp8ConvKernel).

#include "../neuralnet/cudafusedconv.h"

#include <algorithm>
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

// Builds the kernel's Params (initialize) on every call, as cudafusedffn.cu does: a few microseconds of host time,
// while the launch stays asynchronous, against about 0.6 ms of GPU time for each convolution. A cache of the Params
// for each layer and batch would have to patch the epilogue's vector and T pointers itself (update() keeps the old
// ones).
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

// addNCBiasScaleBiasRelu's kernel: each thread works on 8 channels of one point (16-byte loads and stores), in float.
// sumOut may be in: each thread reads its 8 values before it writes them.
__global__ void addNCBiasScaleBiasReluKernel(
  const half* in, half* sumOut, const half* rowBias, const half* scale, const half* bias, half* out,
  int xySize, int c8Size, int total8
) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < total8; i += gridDim.x * blockDim.x) {
    const int c8 = i % c8Size;
    const int n = i / (c8Size * xySize);
    const uint4 xv = ((const uint4*)in)[i];
    const uint4 rowBiasV = ((const uint4*)rowBias)[n * c8Size + c8];
    const uint4 sv = ((const uint4*)scale)[c8];
    const uint4 bv = ((const uint4*)bias)[c8];
    const half* x = (const half*)&xv;
    const half* rowBiasH = (const half*)&rowBiasV;
    const half* s = (const half*)&sv;
    const half* b = (const half*)&bv;
    uint4 sumv, outv;
    half* sum = (half*)&sumv;
    half* o = (half*)&outv;
    #pragma unroll
    for(int j = 0; j < 8; j++) {
      const float t = __half2float(x[j]) + __half2float(rowBiasH[j]);
      const float z = __half2float(s[j]) * t + __half2float(b[j]);
      sum[j] = __float2half(t);
      o[j] = __float2half(z > 0.0f ? z : 0.0f);
    }
    if(sumOut != nullptr)
      ((uint4*)sumOut)[i] = sumv;
    ((uint4*)out)[i] = outv;
  }
}

// ---- The FP8 kernels (abscls-jtbd) ----

using Fp8 = cutlass::float_e4m3_t;
using CudaFusedConv::Fp8Channel;
static_assert(sizeof(Fp8Channel) == sizeof(double), "an Fp8Channel travels as the 64 bits of a double");

// The functor of the FP8 kernels' EpilogueWithBroadcast. The broadcast vector holds one Fp8Channel (64 bits) for each
// output channel, declared as double: the epilogue loads it as ElementVector and converts it to ElementCompute, and a
// double to double conversion is a copy, so the 64 bits arrive unchanged (ScaleBiasReluOp's 32-bit word, widened to
// carry the dequantisation too). The epilogue uses ElementCompute for the broadcast vector only.
//
// The two outputs: Z is the output tile iterator's tensor, in FP16, which also reads the source C. With the source, C
// is the old trunk and Z the new one. Without it, Z is not stored. T is the stage after's input, of ElementNext: FP8,
// or FP16 when the stage after reads FP16. In float:
//   without the source:  T = outScale * relu(scale * (dequant * acc) + bias)
//   with the source:     Z = C + dequant * acc, T = outScale * relu(scale * Z + bias)
// T's conversion to FP8 rounds to nearest even and saturates at +-448 (cvt.rn.satfinite).
template <bool AddSource, typename ElementNext>
class Fp8ScaleBiasReluOp {
public:
  using ElementOutput = Half;
  using ElementD = ElementOutput;
  using ElementC = Half;
  using ElementAccumulator = float;
  using ElementCompute = double;
  using ElementScalar = float;
  using ElementZ = ElementOutput;
  using ElementT = ElementNext;
  using ElementVector = double;
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
  static bool const kStoreZ = AddSource;  // the new trunk, only when the old one is added
  static bool const kStoreT = true;

  struct Params {
    float outScale;
  };

  float outScale;

  CUTLASS_HOST_DEVICE
  explicit Fp8ScaleBiasReluOp(Params const& params): outScale(params.outScale) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return AddSource; }

  // Split-K is not used (one slice), so there is no partial sum to skip.
  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  // With the source: Z = C + dequant * acc, T = outScale * relu(scale * Z + bias).
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& fragZ, FragmentT& fragT, FragmentAccumulator const& acc, FragmentC const& fragC,
                  FragmentCompute const& channels) const {
    cutlass::Array<float, kElementsPerAccess> next;
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kElementsPerAccess; ++i) {
      const Fp8Channel ch = unpack(channels[i]);
      const float sum = float(fragC[i]) + ch.dequant * acc[i];
      fragZ[i] = Half(sum);
      next[i] = normRelu(ch.scaleBias, sum);
    }
    fragT = convertNext(next);
  }

  // Without the source: T = outScale * relu(scale * (dequant * acc) + bias).
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& fragZ, FragmentT& fragT, FragmentAccumulator const& acc,
                  FragmentCompute const& channels) const {
    (void)fragZ;
    cutlass::Array<float, kElementsPerAccess> next;
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kElementsPerAccess; ++i) {
      const Fp8Channel ch = unpack(channels[i]);
      next[i] = normRelu(ch.scaleBias, ch.dequant * acc[i]);
    }
    fragT = convertNext(next);
  }

private:
  CUTLASS_HOST_DEVICE
  static Fp8Channel unpack(double carrier) {
    Fp8Channel ch;
    memcpy(&ch, &carrier, sizeof(ch));
    return ch;
  }

  // outScale * relu(scale * x + bias), the scale and bias the FP16 halves of the word (ScaleBiasReluOp's).
  CUTLASS_HOST_DEVICE
  float normRelu(uint32_t word, float x) const {
    const float scale = float(Half::bitcast((uint16_t)(word & 0xFFFFu)));
    const float bias = float(Half::bitcast((uint16_t)(word >> 16)));
    const float z = scale * x + bias;
    return z > 0.0f ? z * outScale : 0.0f;  // as KataGo's ReLU kernels: a NaN becomes 0
  }

  CUTLASS_HOST_DEVICE
  static FragmentT convertNext(cutlass::Array<float, kElementsPerAccess> const& next) {
    cutlass::NumericArrayConverter<ElementT, float, kElementsPerAccess, cutlass::FloatRoundStyle::round_to_nearest> cvt;
    return cvt(next);
  }
};

// The mainloop of the FP8 kernels: threadblock 128x64x64, warp 64x32x64, mma 16x8x32, 4 stages, the optimized
// iterators, and OpMultiplyAdd, which for FP8 on SM89 is CUTLASS's staged accumulation: each warp sums a k-tile of
// 64 products in a second set of accumulators and adds them to the FP32 ones, since the instruction's own
// accumulation is not full FP32. The second set doubles the accumulators' registers, so the warp tile is half the
// FP16 kernel's: at 64x64, a thread held 256 accumulators and spilled (255 registers, about 800 bytes of spills), and
// the kernel ran at 0.97 ms against the FP16 kernel's 0.62 ms (batch 48 on the 67 x 67 view, 128 channels, the
// trunk add with the next norm). At 128x64 with 64x32 warps, 0.33 ms.
template <bool AddSource, typename ElementNext>
using Fp8ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
  Fp8, cutlass::layout::TensorNHWC,
  Fp8, cutlass::layout::TensorNHWC,
  Half, cutlass::layout::TensorNHWC,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm89,
  cutlass::gemm::GemmShape<128, 64, 64>,
  cutlass::gemm::GemmShape<64, 32, 64>,
  cutlass::gemm::GemmShape<16, 8, 32>,
  Fp8ScaleBiasReluOp<AddSource, ElementNext>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  4,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

template <bool AddSource, typename ElementNext>
using Fp8Conv = cutlass::conv::device::ImplicitGemmConvolution<Fp8ConvKernel<AddSource, ElementNext>>;

// The arguments of one FP8 convolution. With the source, trunk is C and Z; without it, Z is not stored and next stands
// in for C and Z (never read or written).
template <bool AddSource, typename ElementNext>
typename Fp8Conv<AddSource, ElementNext>::Arguments makeFp8Args(
  const uint8_t* in, const uint8_t* filter, half* trunk, const Fp8Channel* channels, float outScale, void* next,
  int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize
) {
  cutlass::conv::Conv2dProblemSize problem(
    batchSize, ySize, xSize, inChannels, outChannels, convYSize, convXSize, ySize, xSize,
    convYSize / 2, convXSize / 2, 1, 1, 1, 1, cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  using Layout = cutlass::layout::TensorNHWC;
  const Layout inLayout = Layout::packed({batchSize, ySize, xSize, inChannels});
  const Layout filterLayout = Layout::packed({outChannels, convYSize, convXSize, inChannels});
  const Layout outLayout = Layout::packed({batchSize, ySize, xSize, outChannels});
  Half* z = (Half*)(AddSource ? (void*)trunk : next);
  typename Fp8ScaleBiasReluOp<AddSource, ElementNext>::Params params;
  params.outScale = outScale;
  return typename Fp8Conv<AddSource, ElementNext>::Arguments(
    problem,
    {(Fp8*)in, inLayout},
    {(Fp8*)filter, filterLayout},
    {z, outLayout},
    {z, outLayout},
    params,
    cutlass::conv::SplitKMode::kSerial,
    (void*)channels,
    next,
    0,              // ldr: one vector for every row
    outChannels);   // ldt: T's row stride, as the output's
}

template <bool AddSource, typename ElementNext>
bool fp8Implementable(typename Fp8Conv<AddSource, ElementNext>::Arguments const& args) {
  return Fp8Conv<AddSource, ElementNext>::can_implement(args) == cutlass::Status::kSuccess
    && Fp8Conv<AddSource, ElementNext>::get_workspace_size(args) == 0;
}

// As runConv: the Params on every call.
template <bool AddSource, typename ElementNext>
cutlass::Status runFp8Conv(typename Fp8Conv<AddSource, ElementNext>::Arguments const& args, cudaStream_t stream) {
  Fp8Conv<AddSource, ElementNext> op;
  cutlass::Status status = op.initialize(args, nullptr, stream);
  if(status == cutlass::Status::kSuccess)
    status = op.run(stream);
  return status;
}

void requireFp8Aligned(const void* in, const void* filter, const void* trunk, const void* next, const void* channels) {
  if((uintptr_t(in) | uintptr_t(filter) | uintptr_t(trunk) | uintptr_t(next)) & 15 || uintptr_t(channels) & 63)
    throw std::runtime_error("CudaFusedConv: FP8 operand pointers must be 16-byte aligned (the channels' words 64-byte)");
}

// addNCBiasScaleBiasReluKernel with out in FP8: each thread works on 8 channels of one point.
__global__ void addNCBiasScaleBiasReluFp8Kernel(
  const half* in, half* sumOut, const half* rowBias, const half* scale, const half* bias, float outScale, uint8_t* out,
  int xySize, int c8Size, int total8
) {
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < total8; i += gridDim.x * blockDim.x) {
    const int c8 = i % c8Size;
    const int n = i / (c8Size * xySize);
    const uint4 xv = ((const uint4*)in)[i];
    const uint4 rowBiasV = ((const uint4*)rowBias)[n * c8Size + c8];
    const uint4 sv = ((const uint4*)scale)[c8];
    const uint4 bv = ((const uint4*)bias)[c8];
    const half* x = (const half*)&xv;
    const half* rowBiasH = (const half*)&rowBiasV;
    const half* s = (const half*)&sv;
    const half* b = (const half*)&bv;
    uint4 sumv;
    half* sum = (half*)&sumv;
    cutlass::Array<float, 8> o;
    #pragma unroll
    for(int j = 0; j < 8; j++) {
      const float t = __half2float(x[j]) + __half2float(rowBiasH[j]);
      const float z = __half2float(s[j]) * t + __half2float(b[j]);
      sum[j] = __float2half(t);
      o[j] = z > 0.0f ? z * outScale : 0.0f;
    }
    if(sumOut != nullptr)
      ((uint4*)sumOut)[i] = sumv;
    cutlass::NumericArrayConverter<Fp8, float, 8, cutlass::FloatRoundStyle::round_to_nearest> cvt;
    const cutlass::Array<Fp8, 8> o8 = cvt(o);
    ((uint2*)out)[i] = *reinterpret_cast<const uint2*>(&o8);
  }
}

// absMax's kernel: the largest |x| of each warp's values, then one atomicMax a warp on the float's bits, which order
// as the floats do for values that are not negative.
__global__ void absMaxKernel(const half* in, size_t total8, float* amax) {
  float m = 0.0f;
  for(size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total8; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = ((const uint4*)in)[i];
    const half* h = (const half*)&v;
    #pragma unroll
    for(int j = 0; j < 8; j++)
      m = fmaxf(m, fabsf(__half2float(h[j])));  // fmaxf skips a NaN
  }
  #pragma unroll
  for(int offset = 16; offset > 0; offset /= 2)
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, offset));
  if((threadIdx.x & 31) == 0)
    atomicMax((unsigned int*)amax, __float_as_uint(m));
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
  if(convYSize != 3 || convXSize != 3)
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

void addNCBiasScaleBiasRelu(
  const half* in, half* sumOut, const half* rowBias, const half* scale, const half* bias, half* out,
  int batchSize, int xySize, int channels, cudaStream_t stream
) {
  if(channels % 8 != 0)
    throw std::runtime_error("CudaFusedConv::addNCBiasScaleBiasRelu: channels must be a multiple of 8");
  if((uintptr_t(in) | uintptr_t(sumOut) | uintptr_t(rowBias) | uintptr_t(scale) | uintptr_t(bias) | uintptr_t(out))
     & 15)
    throw std::runtime_error("CudaFusedConv::addNCBiasScaleBiasRelu: buffers must be 16-byte aligned");
  // The kernel indexes the elements in int.
  if((int64_t)batchSize * xySize * channels > INT32_MAX)
    throw std::runtime_error("CudaFusedConv::addNCBiasScaleBiasRelu: more elements than an int indexes");
  const int c8Size = channels / 8;
  const int total8 = batchSize * xySize * c8Size;
  if(total8 <= 0)
    return;
  const int threads = 256;
  const int blocks = (total8 + threads - 1) / threads;
  addNCBiasScaleBiasReluKernel<<<blocks, threads, 0, stream>>>(
    in, sumOut, rowBias, scale, bias, out, xySize, c8Size, total8);
}

// ---- The FP8 kernels (abscls-jtbd) ----

uint8_t floatToFp8(float x) {
  return Fp8(x).raw();
}

float fp8ToFloat(uint8_t x) {
  return float(Fp8::bitcast(x));
}

bool fp8SupportedOnCurrentDevice() {
  // SM89's FP8 instruction: compute capability 8.9 or more. Below, CUTLASS's kernel would trap (supportedOnCurrentDevice).
  {
    int device = 0;
    int major = 0;
    int minor = 0;
    if(cudaGetDevice(&device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
       || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
      return false;
    if(major < 8 || (major == 8 && minor < 9))
      return false;
  }
  // The three kernels on a tiny convolution with zero inputs and filters, dequant 1, scale 1, bias 0.5 and outScale 2:
  // the first writes fp8(2 * 0.5) = 1; the second adds 0 to a trunk of 1 and writes fp8(2 * 1.5) = 3; the third writes
  // 1.5 in FP16. The outputs start as NaN (FP8's 0x7F, FP16's 0xFFFF), so a kernel that did not run fails.
  constexpr int N = 1, Y = 4, X = 4, C = 32, K = 32, R = 3, S = 3;
  constexpr size_t numIn = (size_t)N * Y * X * C, numFilter = (size_t)K * R * S * C, numOut = (size_t)N * Y * X * K;
  uint8_t* buf8 = nullptr;
  half* buf16 = nullptr;
  Fp8Channel* channels = nullptr;
  bool ok = cudaMalloc(&buf8, numIn + numFilter + 2 * numOut) == cudaSuccess;
  ok = cudaMalloc(&buf16, 2 * numOut * sizeof(half)) == cudaSuccess && ok;
  ok = cudaMalloc(&channels, K * sizeof(Fp8Channel)) == cudaSuccess && ok;
  uint8_t* in = buf8;
  uint8_t* filter = in + numIn;
  uint8_t* out1 = filter + numFilter;
  uint8_t* out2 = out1 + numOut;
  half* trunk = buf16;
  half* out3 = trunk + numOut;
  if(ok) {
    Fp8Channel words[K];
    for(int k = 0; k < K; k++)
      words[k] = Fp8Channel{1.0f, packScaleBias(0x3C00u, 0x3800u)};  // FP16's 1.0 and 0.5
    half ones[numOut];
    for(size_t i = 0; i < numOut; i++)
      ones[i] = __float2half(1.0f);
    ok = ok && cudaMemcpy(channels, words, sizeof(words), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemset(in, 0, numIn + numFilter) == cudaSuccess;
    ok = ok && cudaMemset(out1, 0x7F, 2 * numOut) == cudaSuccess;
    ok = ok && cudaMemcpy(trunk, ones, sizeof(ones), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemset(out3, 0xFF, numOut * sizeof(half)) == cudaSuccess;
  }
  if(ok) {
    auto args1 = makeFp8Args<false, Fp8>(in, filter, nullptr, channels, 2.0f, out1, N, Y, X, C, K, R, S);
    auto args2 = makeFp8Args<true, Fp8>(in, filter, trunk, channels, 2.0f, out2, N, Y, X, C, K, R, S);
    auto args3 = makeFp8Args<true, Half>(in, filter, trunk, channels, 1.0f, out3, N, Y, X, C, K, R, S);
    ok = fp8Implementable<false, Fp8>(args1) && fp8Implementable<true, Fp8>(args2)
      && fp8Implementable<true, Half>(args3);
    ok = ok && runFp8Conv<false, Fp8>(args1, nullptr) == cutlass::Status::kSuccess;
    ok = ok && runFp8Conv<true, Fp8>(args2, nullptr) == cutlass::Status::kSuccess;
    ok = ok && runFp8Conv<true, Half>(args3, nullptr) == cutlass::Status::kSuccess;
  }
  uint8_t host8[2 * numOut];
  half host16[2 * numOut];
  ok = ok && cudaMemcpy(host8, out1, sizeof(host8), cudaMemcpyDeviceToHost) == cudaSuccess;
  ok = ok && cudaMemcpy(host16, trunk, sizeof(host16), cudaMemcpyDeviceToHost) == cudaSuccess;
  if(ok) {
    for(size_t i = 0; i < numOut; i++) {
      if(fp8ToFloat(host8[i]) != 1.0f || fp8ToFloat(host8[numOut + i]) != 3.0f
         || __half2float(host16[i]) != 1.0f || __half2float(host16[numOut + i]) != 1.5f) {
        ok = false;
        break;
      }
    }
  }
  // Clear a recoverable launch error so that it cannot leak into later, unrelated CUDA calls.
  (void)cudaGetLastError();
  (void)cudaFree(buf8);
  (void)cudaFree(buf16);
  (void)cudaFree(channels);
  return ok;
}

bool fp8SupportsShape(int inChannels, int outChannels, int convYSize, int convXSize) {
  if(inChannels <= 0 || outChannels <= 0 || inChannels % 16 != 0 || outChannels % 8 != 0)
    return false;
  if(convYSize != 3 || convXSize != 3)
    return false;
  // As supportsShape: the dummy pointers carry the alignment of the device allocations.
  uint8_t* dummy8 = reinterpret_cast<uint8_t*>(uintptr_t(256));
  half* dummy16 = reinterpret_cast<half*>(uintptr_t(256));
  const Fp8Channel* dummyChannels = reinterpret_cast<const Fp8Channel*>(uintptr_t(256));
  const int Y = 2 * convYSize, X = 2 * convXSize;
  return fp8Implementable<false, Fp8>(
           makeFp8Args<false, Fp8>(dummy8, dummy8, nullptr, dummyChannels, 1.0f, dummy8, 1, Y, X, inChannels,
                                   outChannels, convYSize, convXSize))
    && fp8Implementable<true, Fp8>(
           makeFp8Args<true, Fp8>(dummy8, dummy8, dummy16, dummyChannels, 1.0f, dummy8, 1, Y, X, inChannels,
                                  outChannels, convYSize, convXSize))
    && fp8Implementable<true, Half>(
           makeFp8Args<true, Half>(dummy8, dummy8, dummy16, dummyChannels, 1.0f, dummy16, 1, Y, X, inChannels,
                                   outChannels, convYSize, convXSize));
}

void runFp8ConvScaleBiasRelu(
  const uint8_t* in, const uint8_t* filter, const Fp8Channel* channels, float outScale, uint8_t* out,
  int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
  cudaStream_t stream
) {
  requireFp8Aligned(in, filter, out, out, channels);
  auto args = makeFp8Args<false, Fp8>(in, filter, nullptr, channels, outScale, out, batchSize, ySize, xSize,
                                      inChannels, outChannels, convYSize, convXSize);
  throwIfFailed(runFp8Conv<false, Fp8>(args, stream), "runFp8ConvScaleBiasRelu", batchSize, ySize, xSize, inChannels,
                outChannels);
}

void runFp8ConvAddScaleBiasRelu(
  const uint8_t* in, const uint8_t* filter, half* trunk, const Fp8Channel* channels, float outScale, void* next,
  bool nextFp8, int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
  cudaStream_t stream
) {
  requireFp8Aligned(in, filter, trunk, next, channels);
  if(nextFp8) {
    auto args = makeFp8Args<true, Fp8>(in, filter, trunk, channels, outScale, next, batchSize, ySize, xSize,
                                       inChannels, outChannels, convYSize, convXSize);
    throwIfFailed(runFp8Conv<true, Fp8>(args, stream), "runFp8ConvAddScaleBiasRelu", batchSize, ySize, xSize,
                  inChannels, outChannels);
  }
  else {
    auto args = makeFp8Args<true, Half>(in, filter, trunk, channels, 1.0f, next, batchSize, ySize, xSize, inChannels,
                                        outChannels, convYSize, convXSize);
    throwIfFailed(runFp8Conv<true, Half>(args, stream), "runFp8ConvAddScaleBiasRelu", batchSize, ySize, xSize,
                  inChannels, outChannels);
  }
}

void addNCBiasScaleBiasReluFp8(
  const half* in, half* sumOut, const half* rowBias, const half* scale, const half* bias, float outScale,
  uint8_t* out, int batchSize, int xySize, int channels, cudaStream_t stream
) {
  if(channels % 8 != 0)
    throw std::runtime_error("CudaFusedConv::addNCBiasScaleBiasReluFp8: channels must be a multiple of 8");
  if((uintptr_t(in) | uintptr_t(sumOut) | uintptr_t(rowBias) | uintptr_t(scale) | uintptr_t(bias) | uintptr_t(out))
     & 15)
    throw std::runtime_error("CudaFusedConv::addNCBiasScaleBiasReluFp8: buffers must be 16-byte aligned");
  if((int64_t)batchSize * xySize * channels > INT32_MAX)
    throw std::runtime_error("CudaFusedConv::addNCBiasScaleBiasReluFp8: more elements than an int indexes");
  const int c8Size = channels / 8;
  const int total8 = batchSize * xySize * c8Size;
  if(total8 <= 0)
    return;
  const int threads = 256;
  const int blocks = (total8 + threads - 1) / threads;
  addNCBiasScaleBiasReluFp8Kernel<<<blocks, threads, 0, stream>>>(
    in, sumOut, rowBias, scale, bias, outScale, out, xySize, c8Size, total8);
}

void absMax(const half* in, size_t n, float* amax, cudaStream_t stream) {
  if(n % 8 != 0 || uintptr_t(in) & 15)
    throw std::runtime_error("CudaFusedConv::absMax: n must be a multiple of 8 and in 16-byte aligned");
  const size_t total8 = n / 8;
  if(total8 == 0)
    return;
  const int threads = 256;
  const int blocks = (int)std::min<size_t>((total8 + threads - 1) / threads, 1024);
  absMaxKernel<<<blocks, threads, 0, stream>>>(in, total8, amax);
}

}  // namespace CudaFusedConv
