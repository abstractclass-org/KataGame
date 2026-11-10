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
// The scale and bias are per output channel, given as packed words (packScaleBias).

#ifndef NEURALNET_CUDAFUSEDCONV_H_
#define NEURALNET_CUDAFUSEDCONV_H_

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace CudaFusedConv {
  // Whether the kernels work on the current device: a compute capability of 8.0 or more, and a real tiny convolution
  // that runs and writes the expected values. Synchronous: call once per handle creation.
  bool supportedOnCurrentDevice();

  // Whether a convolution of these sizes is supported: channels multiples of 8, an odd square or rectangular kernel
  // of at most 32 a side, padded to the same size. The batch and the board do not change it.
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
}

#endif  // NEURALNET_CUDAFUSEDCONV_H_
