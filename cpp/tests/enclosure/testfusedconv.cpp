// KataGame fork (2026-11-10, the engine repository's abscls-rvu1): the CUDA backend's fused trunk kernels
// (ENCLOSURE_FUSED_TRUNK) against a direct convolution in float on the CPU, on random data:
//   - CudaFusedConv::runConvScaleBiasRelu: out = relu(scale * conv(in) + bias);
//   - CudaFusedConv::runConvAddScaleBiasRelu: trunk += conv(in), next = relu(scale * trunk + bias);
//   - CudaFusedConv::addNCBiasScaleBiasRelu: out = relu(scale * (in + rowBias) + bias), and in + rowBias.
// Each at several batches and boards (an 11 x 7 board for the borders and partial tiles, the 67 x 67 view), with 128
// and 96 input channels. supportsShape takes 3 x 3 convolutions only. The inputs, filters, scales and biases are
// rounded to FP16 first, so the reference sees the GPU's inputs, and a value passes within 2e-3 + 2e-3 * |reference|
// (FP16's rounding of the output is 4.9e-4 of it).
// The FP8 kernels (abscls-jtbd, ENCLOSURE_FP8_TRUNK) on a GPU that runs them (compute capability 8.9 or more): the
// same convolutions with FP8 inputs and filters, against the reference on the FP8 values in float. An FP16 output
// passes within the tolerance above, and an FP8 output when it is the reference's FP8 or the code next to it.
// Builds against a CUDA build of KataGame that has CUTLASS (tests/enclosure/build.sh BUILD_DIR, with
// TESTS=testfusedconv), and runs on the GPU:
//   testfusedconv            (exit 0: every case passed)
//   testfusedconv bench      each kernel's mean time at batch 48 on the 67 x 67 view, 128 channels

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../neuralnet/cudafusedconv.h"

using namespace std;

namespace {

int failures = 0;

void check(cudaError_t e, const char* what) {
  if(e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}

// Random values rounded to FP16, kept as floats for the reference.
vector<float> randomHalfs(mt19937& rng, size_t n, float lo, float hi) {
  uniform_real_distribution<float> d(lo, hi);
  vector<float> v(n);
  for(float& x: v)
    x = __half2float(__float2half(d(rng)));
  return v;
}

void store(half* d, const vector<float>& v) {
  vector<half> h(v.size());
  for(size_t i = 0; i < v.size(); i++)
    h[i] = __float2half(v[i]);
  check(cudaMemcpy(d, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy");
}

half* upload(const vector<float>& v) {
  half* d = nullptr;
  check(cudaMalloc(&d, v.size() * sizeof(half)), "cudaMalloc");
  store(d, v);
  return d;
}

vector<float> download(const half* d, size_t n) {
  check(cudaDeviceSynchronize(), "kernel");
  vector<half> h(n);
  check(cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost), "cudaMemcpy");
  vector<float> v(n);
  for(size_t i = 0; i < n; i++)
    v[i] = __half2float(h[i]);
  return v;
}

uint32_t* uploadScaleBias(const vector<float>& scale, const vector<float>& bias) {
  vector<uint32_t> packed(scale.size());
  for(size_t i = 0; i < scale.size(); i++) {
    __half_raw s = __float2half(scale[i]), b = __float2half(bias[i]);
    packed[i] = CudaFusedConv::packScaleBias(s.x, b.x);
  }
  uint32_t* d = nullptr;
  check(cudaMalloc(&d, packed.size() * sizeof(uint32_t)), "cudaMalloc");
  check(cudaMemcpy(d, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy");
  return d;
}

// conv(in, filter) in float: in [n, y, x, ic], filter [oc, ky, kx, ic], zero padding, out [n, y, x, oc].
vector<float> convReference(const vector<float>& in, const vector<float>& filter, int n, int ys, int xs, int ic, int oc,
                            int ky, int kx) {
  vector<float> out((size_t)n * ys * xs * oc, 0.0f);
  for(int b = 0; b < n; b++)
    for(int y = 0; y < ys; y++)
      for(int x = 0; x < xs; x++)
        for(int o = 0; o < oc; o++) {
          double sum = 0.0;
          for(int dy = 0; dy < ky; dy++) {
            int yy = y + dy - ky / 2;
            if(yy < 0 || yy >= ys)
              continue;
            for(int dx = 0; dx < kx; dx++) {
              int xx = x + dx - kx / 2;
              if(xx < 0 || xx >= xs)
                continue;
              const float* a = &in[(((size_t)b * ys + yy) * xs + xx) * ic];
              const float* w = &filter[(((size_t)o * ky + dy) * kx + dx) * ic];
              for(int i = 0; i < ic; i++)
                sum += (double)a[i] * w[i];
            }
          }
          out[(((size_t)b * ys + y) * xs + x) * oc + o] = (float)sum;
        }
  return out;
}

// Prints the largest error of got against want and counts a failure above the tolerance.
void compare(const char* what, const vector<float>& got, const vector<float>& want) {
  double maxErr = 0.0, maxRef = 0.0;
  size_t bad = 0;
  for(size_t i = 0; i < want.size(); i++) {
    double err = fabs((double)got[i] - want[i]);
    maxErr = max(maxErr, err);
    maxRef = max(maxRef, fabs((double)want[i]));
    if(!(err <= 2e-3 + 2e-3 * fabs((double)want[i])))
      bad++;
  }
  printf("  %-44s max error %.3g (largest value %.3g)%s\n", what, maxErr, maxRef, bad ? "  FAIL" : "");
  if(bad) {
    printf("    %zu of %zu values outside the tolerance\n", bad, want.size());
    failures++;
  }
}

void testCase(mt19937& rng, int n, int ys, int xs, int ic, int oc) {
  const int ky = 3, kx = 3;
  printf("batch %d, board %d x %d, %d -> %d channels\n", n, xs, ys, ic, oc);
  const size_t inSize = (size_t)n * ys * xs * ic, outSize = (size_t)n * ys * xs * oc;
  const float w = 1.0f / sqrtf((float)ky * kx * ic);
  vector<float> in = randomHalfs(rng, inSize, 0.0f, 2.0f);  // a normalised, activated input: not negative
  vector<float> filter = randomHalfs(rng, (size_t)oc * ky * kx * ic, -2.0f * w, 2.0f * w);
  vector<float> scale = randomHalfs(rng, oc, -0.5f, 1.5f);  // some scales negative, as a trained network may have
  vector<float> bias = randomHalfs(rng, oc, -0.5f, 0.5f);
  vector<float> trunk = randomHalfs(rng, outSize, -3.0f, 3.0f);
  vector<float> conv = convReference(in, filter, n, ys, xs, ic, oc, ky, kx);

  half* dIn = upload(in);
  half* dFilter = upload(filter);
  uint32_t* dScaleBias = uploadScaleBias(scale, bias);
  half* dTrunk = upload(trunk);
  half* dOut = upload(vector<float>(outSize, 7.0f));  // not a possible output: an unwritten value fails

  CudaFusedConv::runConvScaleBiasRelu(dIn, dFilter, dScaleBias, dOut, n, ys, xs, ic, oc, ky, kx, 0);
  vector<float> want(outSize), wantTrunk(outSize);
  for(size_t i = 0; i < outSize; i++) {
    int c = (int)(i % oc);
    want[i] = max(0.0f, scale[c] * conv[i] + bias[c]);
  }
  compare("relu(scale * conv + bias)", download(dOut, outSize), want);

  store(dOut, vector<float>(outSize, 7.0f));
  CudaFusedConv::runConvAddScaleBiasRelu(dIn, dFilter, dTrunk, dScaleBias, dOut, n, ys, xs, ic, oc, ky, kx, 0);
  for(size_t i = 0; i < outSize; i++) {
    int c = (int)(i % oc);
    wantTrunk[i] = trunk[i] + conv[i];
    want[i] = max(0.0f, scale[c] * wantTrunk[i] + bias[c]);
  }
  compare("trunk + conv", download(dTrunk, outSize), wantTrunk);
  compare("relu(scale * (trunk + conv) + bias)", download(dOut, outSize), want);

  cudaFree(dIn);
  cudaFree(dFilter);
  cudaFree(dScaleBias);
  cudaFree(dTrunk);
  cudaFree(dOut);
}

void testAddNCBias(mt19937& rng, int n, int xy, int c) {
  printf("batch %d, %d points, %d channels: add the biases of each row, then scale, bias and ReLU\n", n, xy, c);
  const size_t size = (size_t)n * xy * c;
  vector<float> in = randomHalfs(rng, size, -3.0f, 3.0f);
  vector<float> rowBias = randomHalfs(rng, (size_t)n * c, -1.0f, 1.0f);
  vector<float> scale = randomHalfs(rng, c, -0.5f, 1.5f);
  vector<float> bias = randomHalfs(rng, c, -0.5f, 0.5f);
  half* dIn = upload(in);
  half* dRowBias = upload(rowBias);
  half* dScale = upload(scale);
  half* dBias = upload(bias);
  half* dOut = upload(vector<float>(size, 7.0f));
  vector<float> want(size), wantSum(size);
  for(size_t i = 0; i < size; i++) {
    int ch = (int)(i % c);
    size_t b = i / ((size_t)xy * c);
    wantSum[i] = in[i] + rowBias[b * c + ch];
    want[i] = max(0.0f, scale[ch] * wantSum[i] + bias[ch]);
  }
  CudaFusedConv::addNCBiasScaleBiasRelu(dIn, nullptr, dRowBias, dScale, dBias, dOut, n, xy, c, 0);
  compare("relu(scale * (in + rowBias) + bias)", download(dOut, size), want);
  compare("in unchanged without the sum's output", download(dIn, size), in);
  CudaFusedConv::addNCBiasScaleBiasRelu(dIn, dIn, dRowBias, dScale, dBias, dOut, n, xy, c, 0);
  compare("in + rowBias, in place", download(dIn, size), wantSum);
  compare("relu(scale * (in + rowBias) + bias), in place", download(dOut, size), want);
  cudaFree(dIn);
  cudaFree(dRowBias);
  cudaFree(dScale);
  cudaFree(dBias);
  cudaFree(dOut);
}

// ---- The FP8 kernels (abscls-jtbd) ----

using CudaFusedConv::Fp8Channel;
using CudaFusedConv::floatToFp8;
using CudaFusedConv::fp8ToFloat;

// Random values in [lo, hi] rounded to FP8, as bytes and as floats.
void randomFp8(mt19937& rng, size_t n, float lo, float hi, vector<uint8_t>& bytes, vector<float>& values) {
  uniform_real_distribution<float> d(lo, hi);
  bytes.resize(n);
  values.resize(n);
  for(size_t i = 0; i < n; i++) {
    bytes[i] = floatToFp8(d(rng));
    values[i] = fp8ToFloat(bytes[i]);
  }
}

template <typename T>
T* uploadBytes(const vector<T>& v) {
  T* d = nullptr;
  check(cudaMalloc(&d, v.size() * sizeof(T)), "cudaMalloc");
  check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
  return d;
}

vector<uint8_t> downloadFp8(const uint8_t* d, size_t n) {
  check(cudaDeviceSynchronize(), "kernel");
  vector<uint8_t> h(n);
  check(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost), "cudaMemcpy");
  return h;
}

// FP8 outputs against the reference's values (not negative, as every FP8 output of these kernels): the FP8 of each
// reference value and the output may differ by one step at most, the next code (E4M3's codes of values that are not
// negative order as the values do), as the float sums of the GPU and of the reference may round to the two sides of
// a halfway point.
void compareFp8(const char* what, const vector<uint8_t>& got, const vector<float>& want) {
  size_t bad = 0, oneStep = 0;
  double maxRef = 0.0;
  for(size_t i = 0; i < want.size(); i++) {
    const int g = got[i], w = floatToFp8(want[i]);
    maxRef = max(maxRef, (double)want[i]);
    if(g != w) {
      if(abs(g - w) == 1 && g < 0x7F)
        oneStep++;
      else
        bad++;
    }
  }
  printf("  %-44s %zu of %zu values one FP8 step off (largest value %.3g)%s\n", what, oneStep, want.size(), maxRef,
         bad ? "  FAIL" : "");
  if(bad) {
    printf("    %zu values more than one step off\n", bad);
    failures++;
  }
}

Fp8Channel* uploadChannels(const vector<float>& dequant, const vector<float>& scale, const vector<float>& bias) {
  vector<Fp8Channel> ch(dequant.size());
  for(size_t i = 0; i < ch.size(); i++) {
    __half_raw s = __float2half(scale[i]), b = __float2half(bias[i]);
    ch[i] = Fp8Channel{dequant[i], CudaFusedConv::packScaleBias(s.x, b.x)};
  }
  return uploadBytes(ch);
}

// The three FP8 convolutions and their epilogues, against convReference on the FP8 values in float.
void testFp8Case(mt19937& rng, int n, int ys, int xs, int ic, int oc) {
  const int ky = 3, kx = 3;
  printf("FP8: batch %d, board %d x %d, %d -> %d channels\n", n, xs, ys, ic, oc);
  const size_t inSize = (size_t)n * ys * xs * ic, outSize = (size_t)n * ys * xs * oc;
  vector<uint8_t> in8, filter8;
  vector<float> in, filter;
  randomFp8(rng, inSize, 0.0f, 448.0f, in8, in);  // a normalised, activated input in FP8: not negative
  randomFp8(rng, (size_t)oc * ky * kx * ic, -448.0f, 448.0f, filter8, filter);
  // dequant (a float, below FP16's range) brings the sums (about sqrt(ky * kx * ic) * 260 * 260) to about 1.
  const float unit = 1.0f / (260.0f * 260.0f * sqrtf((float)ky * kx * ic));
  uniform_real_distribution<float> dequantDist(0.5f * unit, 2.0f * unit);
  vector<float> dequant(oc);
  for(float& d: dequant)
    d = dequantDist(rng);
  vector<float> scale = randomHalfs(rng, oc, -0.5f, 1.5f);
  vector<float> bias = randomHalfs(rng, oc, -0.5f, 0.5f);
  vector<float> trunk = randomHalfs(rng, outSize, -3.0f, 3.0f);
  const float outScale = 100.0f;  // 1 / the output's scale: outputs up to about 4 become up to about 400
  vector<float> conv = convReference(in, filter, n, ys, xs, ic, oc, ky, kx);

  uint8_t* dIn = uploadBytes(in8);
  uint8_t* dFilter = uploadBytes(filter8);
  Fp8Channel* dChannels = uploadChannels(dequant, scale, bias);
  half* dTrunk = upload(trunk);
  uint8_t* dOut8 = uploadBytes(vector<uint8_t>(outSize, 0x7F));  // FP8's NaN: an unwritten value fails
  half* dOut16 = upload(vector<float>(outSize, 7.0f));

  CudaFusedConv::runFp8ConvScaleBiasRelu(dIn, dFilter, dChannels, outScale, dOut8, n, ys, xs, ic, oc, ky, kx, 0);
  vector<float> want(outSize), wantTrunk(outSize);
  for(size_t i = 0; i < outSize; i++) {
    int c = (int)(i % oc);
    want[i] = outScale * max(0.0f, scale[c] * (dequant[c] * conv[i]) + bias[c]);
  }
  compareFp8("fp8(outScale * relu(scale * conv + bias))", downloadFp8(dOut8, outSize), want);

  for(size_t i = 0; i < outSize; i++) {
    int c = (int)(i % oc);
    wantTrunk[i] = trunk[i] + dequant[c] * conv[i];
    want[i] = outScale * max(0.0f, scale[c] * wantTrunk[i] + bias[c]);
  }
  check(cudaMemset(dOut8, 0x7F, outSize), "cudaMemset");
  CudaFusedConv::runFp8ConvAddScaleBiasRelu(dIn, dFilter, dTrunk, dChannels, outScale, dOut8, true, n, ys, xs, ic, oc,
                                            ky, kx, 0);
  compare("FP8 conv: trunk + conv", download(dTrunk, outSize), wantTrunk);
  compareFp8("fp8(outScale * relu(scale * (trunk + conv) + bias))", downloadFp8(dOut8, outSize), want);

  store(dTrunk, trunk);
  CudaFusedConv::runFp8ConvAddScaleBiasRelu(dIn, dFilter, dTrunk, dChannels, outScale, dOut16, false, n, ys, xs, ic,
                                            oc, ky, kx, 0);
  for(size_t i = 0; i < outSize; i++)
    want[i] /= outScale;
  compare("FP8 conv: trunk + conv, next in FP16", download(dTrunk, outSize), wantTrunk);
  compare("relu(scale * (trunk + conv) + bias) in FP16", download(dOut16, outSize), want);

  cudaFree(dIn);
  cudaFree(dFilter);
  cudaFree(dChannels);
  cudaFree(dTrunk);
  cudaFree(dOut8);
  cudaFree(dOut16);
}

void testAddNCBiasFp8(mt19937& rng, int n, int xy, int c) {
  printf("FP8: batch %d, %d points, %d channels: add the biases of each row, then scale, bias, ReLU and FP8\n", n, xy, c);
  const size_t size = (size_t)n * xy * c;
  vector<float> in = randomHalfs(rng, size, -3.0f, 3.0f);
  vector<float> rowBias = randomHalfs(rng, (size_t)n * c, -1.0f, 1.0f);
  vector<float> scale = randomHalfs(rng, c, -0.5f, 1.5f);
  vector<float> bias = randomHalfs(rng, c, -0.5f, 0.5f);
  const float outScale = 50.0f;
  half* dIn = upload(in);
  half* dRowBias = upload(rowBias);
  half* dScale = upload(scale);
  half* dBias = upload(bias);
  uint8_t* dOut = uploadBytes(vector<uint8_t>(size, 0x7F));
  vector<float> want(size), wantSum(size);
  for(size_t i = 0; i < size; i++) {
    int ch = (int)(i % c);
    size_t b = i / ((size_t)xy * c);
    wantSum[i] = in[i] + rowBias[b * c + ch];
    // The kernel scales the FP16 sum's float, as the FP16 kernel does.
    want[i] = outScale * max(0.0f, scale[ch] * wantSum[i] + bias[ch]);
  }
  CudaFusedConv::addNCBiasScaleBiasReluFp8(dIn, dIn, dRowBias, dScale, dBias, outScale, dOut, n, xy, c, 0);
  compare("in + rowBias, in place", download(dIn, size), wantSum);
  compareFp8("fp8(outScale * relu(scale * (in + rowBias) + bias))", downloadFp8(dOut, size), want);
  cudaFree(dIn);
  cudaFree(dRowBias);
  cudaFree(dScale);
  cudaFree(dBias);
  cudaFree(dOut);
}

void testAbsMax(mt19937& rng, size_t n) {
  vector<float> v = randomHalfs(rng, n, -5.0f, 3.0f);
  float want = 0.0f;
  for(float x: v)
    want = max(want, fabsf(x));
  half* d = upload(v);
  float* dMax = nullptr;
  const float start = 1.0f;  // below the largest: the kernel raises it
  check(cudaMalloc(&dMax, sizeof(float)), "cudaMalloc");
  check(cudaMemcpy(dMax, &start, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  CudaFusedConv::absMax(d, n, dMax, 0);
  float got = 0.0f;
  check(cudaDeviceSynchronize(), "kernel");
  check(cudaMemcpy(&got, dMax, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
  printf("absMax of %zu values: %g, want %g%s\n", n, got, want, got == want ? "" : "  FAIL");
  if(got != want)
    failures++;
  cudaFree(d);
  cudaFree(dMax);
}

void testFp8Conversions() {
  const float values[] = {0.0f, 0.5f, 1.0f, -3.0f, 448.0f, 1000.0f, -1e9f, 0.001953125f, 1.0625f, 1.1875f};
  const float want[] = {0.0f, 0.5f, 1.0f, -3.0f, 448.0f, 448.0f, -448.0f, 0.001953125f, 1.0f, 1.25f};
  bool ok = true;
  for(size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++)
    ok = ok && fp8ToFloat(floatToFp8(values[i])) == want[i];
  printf("floatToFp8 and fp8ToFloat (rounding to nearest even, saturation): %s\n", ok ? "as expected" : "FAIL");
  if(!ok)
    failures++;
}

// testfusedconv bench: each kernel's mean time at the view's size and a self-play batch, FP16 against FP8.
void bench() {
  const int n = 48, ys = 67, xs = 67, c = 128, k = 3, reps = 200;
  const size_t act = (size_t)n * ys * xs * c, filt = (size_t)c * k * k * c;
  half *in16 = nullptr, *filter16 = nullptr, *trunk = nullptr, *out16 = nullptr;
  uint8_t *in8 = nullptr, *filter8 = nullptr, *out8 = nullptr;
  uint32_t* scaleBias = nullptr;
  Fp8Channel* channels = nullptr;
  check(cudaMalloc(&in16, act * 2), "cudaMalloc");
  check(cudaMalloc(&filter16, filt * 2), "cudaMalloc");
  check(cudaMalloc(&trunk, act * 2), "cudaMalloc");
  check(cudaMalloc(&out16, act * 2), "cudaMalloc");
  check(cudaMalloc(&in8, act), "cudaMalloc");
  check(cudaMalloc(&filter8, filt), "cudaMalloc");
  check(cudaMalloc(&out8, act), "cudaMalloc");
  check(cudaMalloc(&scaleBias, c * 4), "cudaMalloc");
  check(cudaMalloc(&channels, c * 8), "cudaMalloc");
  // Inputs and filters of 1.06 (FP16's 0x3C3C) and 1.5 (FP8's 0x3C), not zeros; the scales are 0, so the outputs are.
  check(cudaMemset(in16, 0x3C, act * 2), "cudaMemset");
  check(cudaMemset(filter16, 0x3C, filt * 2), "cudaMemset");
  check(cudaMemset(trunk, 0, act * 2), "cudaMemset");
  check(cudaMemset(in8, 0x3C, act), "cudaMemset");
  check(cudaMemset(filter8, 0x3C, filt), "cudaMemset");
  check(cudaMemset(scaleBias, 0, c * 4), "cudaMemset");
  check(cudaMemset(channels, 0, c * 8), "cudaMemset");
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "cudaEventCreate");
  check(cudaEventCreate(&stop), "cudaEventCreate");
  auto time = [&](const char* what, auto launch) {
    for(int i = 0; i < 20; i++)
      launch();
    check(cudaEventRecord(start), "cudaEventRecord");
    for(int i = 0; i < reps; i++)
      launch();
    check(cudaEventRecord(stop), "cudaEventRecord");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    const double flop = 2.0 * act * k * k * c;
    printf("  %-40s %.3f ms, %.1f TFLOP/s\n", what, ms / reps, flop / (ms / reps * 1e-3) / 1e12);
  };
  printf("bench: batch %d, %d x %d, %d -> %d channels, 3 x 3\n", n, xs, ys, c, c);
  time("FP16 conv, scale, bias, ReLU", [&] {
    CudaFusedConv::runConvScaleBiasRelu(in16, filter16, scaleBias, out16, n, ys, xs, c, c, k, k, 0);
  });
  time("FP16 conv, trunk add, next norm", [&] {
    CudaFusedConv::runConvAddScaleBiasRelu(in16, filter16, trunk, scaleBias, out16, n, ys, xs, c, c, k, k, 0);
  });
  time("FP8 conv, scale, bias, ReLU, FP8 out", [&] {
    CudaFusedConv::runFp8ConvScaleBiasRelu(in8, filter8, channels, 1.0f, out8, n, ys, xs, c, c, k, k, 0);
  });
  time("FP8 conv, trunk add, next norm in FP8", [&] {
    CudaFusedConv::runFp8ConvAddScaleBiasRelu(in8, filter8, trunk, channels, 1.0f, out8, true, n, ys, xs, c, c, k, k,
                                              0);
  });
  time("FP8 conv, trunk add, next norm in FP16", [&] {
    CudaFusedConv::runFp8ConvAddScaleBiasRelu(in8, filter8, trunk, channels, 1.0f, out16, false, n, ys, xs, c, c, k,
                                              k, 0);
  });
  cudaFree(in16);
  cudaFree(filter16);
  cudaFree(trunk);
  cudaFree(out16);
  cudaFree(in8);
  cudaFree(filter8);
  cudaFree(out8);
  cudaFree(scaleBias);
  cudaFree(channels);
}

}  // namespace

int main(int argc, char** argv) {
  if(argc == 2 && string(argv[1]) == "bench") {
    if(!CudaFusedConv::supportedOnCurrentDevice() || !CudaFusedConv::fp8SupportedOnCurrentDevice()) {
      printf("the kernels do not run on this device\n");
      return 1;
    }
    bench();
    return 0;
  }
  if(!CudaFusedConv::supportedOnCurrentDevice()) {
    printf("CudaFusedConv::supportedOnCurrentDevice() is false: FAIL\n");
    return 1;
  }
  const bool shapes = CudaFusedConv::supportsShape(128, 128, 3, 3) && CudaFusedConv::supportsShape(96, 128, 3, 3)
    && !CudaFusedConv::supportsShape(100, 128, 3, 3) && !CudaFusedConv::supportsShape(128, 100, 3, 3)
    && !CudaFusedConv::supportsShape(128, 128, 1, 1) && !CudaFusedConv::supportsShape(128, 128, 5, 5);
  printf("supportsShape: %s\n", shapes ? "as expected" : "FAIL");
  if(!shapes)
    failures++;
  mt19937 rng(20261110);
  testCase(rng, 1, 7, 11, 128, 128);
  testCase(rng, 5, 7, 11, 128, 128);
  testCase(rng, 64, 7, 11, 128, 128);
  testCase(rng, 3, 7, 11, 96, 128);
  testCase(rng, 2, 67, 67, 128, 128);
  testAddNCBias(rng, 3, 77, 96);
  testAddNCBias(rng, 2, 67 * 67, 128);

  testFp8Conversions();
  if(!CudaFusedConv::fp8SupportedOnCurrentDevice()) {
    printf("CudaFusedConv::fp8SupportedOnCurrentDevice() is false: the FP8 kernels are not checked%s\n",
           CudaFusedConv::fp8SupportsShape(128, 128, 3, 3) ? "" : " (fp8SupportsShape(128, 128, 3, 3) false too)");
  }
  else {
    const bool fp8Shapes = CudaFusedConv::fp8SupportsShape(128, 128, 3, 3)
      && CudaFusedConv::fp8SupportsShape(96, 128, 3, 3) && CudaFusedConv::fp8SupportsShape(128, 96, 3, 3)
      && !CudaFusedConv::fp8SupportsShape(120, 128, 3, 3) && !CudaFusedConv::fp8SupportsShape(128, 100, 3, 3)
      && !CudaFusedConv::fp8SupportsShape(128, 128, 1, 1);
    printf("fp8SupportsShape: %s\n", fp8Shapes ? "as expected" : "FAIL");
    if(!fp8Shapes)
      failures++;
    testFp8Case(rng, 1, 7, 11, 128, 128);
    testFp8Case(rng, 5, 7, 11, 128, 128);
    testFp8Case(rng, 64, 7, 11, 128, 128);
    testFp8Case(rng, 3, 7, 11, 96, 128);
    testFp8Case(rng, 2, 67, 67, 128, 128);
    testAddNCBiasFp8(rng, 3, 77, 96);
    testAddNCBiasFp8(rng, 2, 67 * 67, 128);
    testAbsMax(rng, 8 * 1000);
    testAbsMax(rng, (size_t)2 * 67 * 67 * 128);
  }
  printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
