// KataGame fork (2026-11-10, the engine repository's abscls-rvu1; ENCLOSURE_FUSED_TRUNK=1, off by default): CUTLASS
// convolutions for the CUDA backend's convolutional trunk, with a residual block's scale, bias and ReLU passes in their
// epilogues. Implemented in cudafusedconv.cu, which is compiled only when the build finds the vendored CUTLASS (CMake
// defines USE_CUTLASS_FUSED_CONV), so call sites must be guarded on that define.
//
// The kernel is CUTLASS 2.x's implicit-GEMM forward convolution with the tile that cuDNN chooses for KataGo's 3x3
// convolutions of 128 channels in FP16 (cutlass_tensorop_f16_s16816fprop_optimized_f16_128x128_32x3_nhwc_align8: FP16
// inputs and outputs, FP32 accumulation, NHWC, the "optimized" iterators, 3 stages). Only its epilogue differs:
//   - runConvScaleBiasRelu: out = relu(scale * conv(in) + bias), the scale and bias applied to the float accumulator
//     before its single rounding to FP16 (a block's first convolution with the block's middle normalisation);
//   - runConvAddScaleBiasRelu: trunk = trunk + conv(in), and next = relu(scale * trunk + bias), both from the float
//     sum before its rounding (a block's last convolution, which also writes the next block's normalised, activated
//     input, or the trunk's final normalisation).
// The scale and bias are per output channel, given as packed words (packScaleBias). addNCBiasScaleBiasRelu is the
// fused trunk's one pass that is not a convolution. The FP8 forms of these (abscls-jtbd) are at the end.

#ifndef NEURALNET_CUDAFUSEDCONV_H_
#define NEURALNET_CUDAFUSEDCONV_H_

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace CudaFusedConv {
  // Whether the kernels work on the current device: a compute capability of 8.0 or more, and a real tiny convolution
  // that runs and writes the expected values. Synchronous: call once per handle creation.
  bool supportedOnCurrentDevice();

  // Whether a convolution of these sizes is supported: channels multiples of 8 and a 3 x 3 kernel (every residual
  // block's convolutions, the shape that testfusedconv checks), padded to the same size. The batch and the board do
  // not change it.
  bool supportsShape(int inChannels, int outChannels, int convYSize, int convXSize);

  // The word for one channel that the epilogues read: the FP16 bits of the scale in the low half and those of the bias
  // in the high half. The epilogue's broadcast vector holds one 32-bit element per output channel, and these kernels
  // need two numbers a channel, so each element carries both.
  inline uint32_t packScaleBias(uint16_t scaleBits, uint16_t biasBits) {
    return (uint32_t)scaleBits | ((uint32_t)biasBits << 16);
  }

  // out = relu(scale * conv(in, filter) + bias). in: [batch, y, x, inChannels], filter: [outChannels, convY, convX,
  // inChannels] (cuDNN's NHWC filter), scaleBias: outChannels packed words, out: [batch, y, x, outChannels]. The
  // convolution is a cross-correlation padded with zeros to the input's size, stride 1, no dilation. All pointers are
  // 16-byte aligned (scaleBias 32-byte). Throws std::runtime_error on a CUDA or CUTLASS failure.
  void runConvScaleBiasRelu(
    const half* in, const half* filter, const uint32_t* scaleBias, half* out,
    int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
    cudaStream_t stream
  );

  // trunk = trunk + conv(in, filter), and next = relu(scale * trunk + bias), both computed from the float sum.
  // trunk and next: [batch, y, x, outChannels], distinct buffers, and in is neither of them.
  void runConvAddScaleBiasRelu(
    const half* in, const half* filter, half* trunk, const uint32_t* scaleBias, half* next,
    int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
    cudaStream_t stream
  );

  // out = relu(scale * (in + rowBias) + bias), computed in float, and when sumOut is not null, also
  // sumOut = in + rowBias (sumOut may be in). in, sumOut and out: [batch, xy, channels]; rowBias: [batch, channels] (a
  // bias for each row of the batch, such as a global pooling block's pooled bias); scale and bias: [channels], FP16 (a
  // BatchNormLayer's merged buffers). No mask. channels a multiple of 8, every buffer 16-byte aligned: else it throws
  // std::runtime_error.
  void addNCBiasScaleBiasRelu(
    const half* in, half* sumOut, const half* rowBias, const half* scale, const half* bias, half* out,
    int batchSize, int xySize, int channels, cudaStream_t stream
  );

  // KataGame fork (2026-11-10, the engine repository's abscls-jtbd; ENCLOSURE_FP8_TRUNK=1 with ENCLOSURE_FUSED_TRUNK=1,
  // off by default): the same convolutions with 8-bit inputs and filters, FP8 E4M3 (1 sign bit, 4 exponent bits, 3
  // mantissa bits, largest value 448). The mainloop is CUTLASS 2.x's implicit-GEMM forward convolution with SM89's FP8
  // tensor-core instruction (mma.sync m16n8k32 e4m3 x e4m3, FP32 accumulation, CUTLASS's staged accumulation, which
  // adds each k-tile's sum to the FP32 accumulators in full precision), which sm_89 and later GPUs run, sm_120 too. The
  // epilogues are those above, with the FP8 accumulator first turned into the convolution's value. A tensor in FP8
  // holds x / s: s is the tensor's scale (one for each tensor of activations, one for each output channel of a
  // filter). Bytes of FP8 are passed as uint8_t.

  // Whether the FP8 kernels work on the current device: compute capability 8.9 or more, and a real tiny convolution of
  // each kernel that writes the expected values. Synchronous: call once per handle creation.
  bool fp8SupportedOnCurrentDevice();

  // Whether an FP8 convolution of these sizes is supported: input channels multiples of 16, output channels multiples
  // of 8, a 3 x 3 kernel.
  bool fp8SupportsShape(int inChannels, int outChannels, int convYSize, int convXSize);

  // What an FP8 convolution's epilogue reads for each output channel (8 bytes, 8-byte aligned in an array): dequant,
  // the factor that turns the FP8 accumulator into the convolution's value (the input's scale times the channel's
  // weight scale), and scaleBias, the normalisation after it (packScaleBias's word).
  struct Fp8Channel {
    float dequant;
    uint32_t scaleBias;
  };

  // FP8 E4M3 conversions on the host: round to nearest even, saturated at +-448 (as the kernels' cvt.rn.satfinite).
  uint8_t floatToFp8(float x);
  float fp8ToFloat(uint8_t x);

  // out = fp8(outScale * relu(scale * dequant * conv(in, filter) + bias)): in, filter and out in FP8, the layouts of
  // runConvScaleBiasRelu. channels: outChannels words (8-byte aligned), outScale: 1 / out's scale. All other pointers
  // 16-byte aligned. Throws std::runtime_error on a CUDA or CUTLASS failure.
  void runFp8ConvScaleBiasRelu(
    const uint8_t* in, const uint8_t* filter, const Fp8Channel* channels, float outScale, uint8_t* out,
    int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
    cudaStream_t stream
  );

  // trunk = trunk + dequant * conv(in, filter), and next = relu(scale * trunk + bias), both from the float sum. in and
  // filter in FP8, trunk in FP16. next in FP8 when nextFp8 (next = fp8(outScale * relu(...))), else in FP16 (outScale
  // unused). trunk and next distinct buffers.
  void runFp8ConvAddScaleBiasRelu(
    const uint8_t* in, const uint8_t* filter, half* trunk, const Fp8Channel* channels, float outScale, void* next,
    bool nextFp8, int batchSize, int ySize, int xSize, int inChannels, int outChannels, int convYSize, int convXSize,
    cudaStream_t stream
  );

  // addNCBiasScaleBiasRelu with out in FP8: out = fp8(outScale * relu(scale * (in + rowBias) + bias)).
  void addNCBiasScaleBiasReluFp8(
    const half* in, half* sumOut, const half* rowBias, const half* scale, const half* bias, float outScale,
    uint8_t* out, int batchSize, int xySize, int channels, cudaStream_t stream
  );

  // *amax = max(*amax, the largest |x| of in's n values): in FP16, n a multiple of 8, in 16-byte aligned, amax a float
  // on the device that is not negative. NaNs are skipped.
  void absMax(const half* in, size_t n, float* amax, cudaStream_t stream);
}

#endif  // NEURALNET_CUDAFUSEDCONV_H_
