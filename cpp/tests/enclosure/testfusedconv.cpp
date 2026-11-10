// KataGame fork (2026-11-10, the engine repository's abscls-rvu1): the CUDA backend's fused trunk kernels
// (ENCLOSURE_FUSED_TRUNK) against a direct convolution in float on the CPU, on random data:
//   - CudaFusedConv::runConvScaleBiasRelu: out = relu(scale * conv(in) + bias);
//   - CudaFusedConv::runConvAddScaleBiasRelu: trunk += conv(in), next = relu(scale * trunk + bias);
//   - customCudaAddNCBiasApplyCScaleBiasReluNHWC: out = relu(scale * (in + nb) + bias), and in + nb.
// Each at several batches and boards (an 11 x 7 board for the borders and partial tiles, the 67 x 67 view), with 128
// and 96 input channels. The inputs, filters, scales and biases are rounded to FP16 first, so the reference sees the
// GPU's inputs, and a value passes within 2e-3 + 2e-3 * |reference| (FP16's rounding of the output is 4.9e-4 of it).
// Builds against a CUDA build of KataGame that has CUTLASS (tests/enclosure/build.sh BUILD_DIR, with
// TESTS=testfusedconv), and runs on the GPU:
//   testfusedconv            (exit 0: every case passed)

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../neuralnet/cudafusedconv.h"
#include "../../neuralnet/cudahelpers.h"

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
  vector<float> nb = randomHalfs(rng, (size_t)n * c, -1.0f, 1.0f);
  vector<float> scale = randomHalfs(rng, c, -0.5f, 1.5f);
  vector<float> bias = randomHalfs(rng, c, -0.5f, 0.5f);
  half* dIn = upload(in);
  half* dNb = upload(nb);
  half* dScale = upload(scale);
  half* dBias = upload(bias);
  half* dOut = upload(vector<float>(size, 7.0f));
  vector<float> want(size), wantSum(size);
  for(size_t i = 0; i < size; i++) {
    int ch = (int)(i % c);
    size_t b = i / ((size_t)xy * c);
    wantSum[i] = in[i] + nb[b * c + ch];
    want[i] = max(0.0f, scale[ch] * wantSum[i] + bias[ch]);
  }
  customCudaAddNCBiasApplyCScaleBiasReluNHWC(dIn, nullptr, dNb, dScale, dBias, dOut, n, xy, c, 0);
  compare("relu(scale * (x + nb) + bias)", download(dOut, size), want);
  compare("x unchanged without the sum's output", download(dIn, size), in);
  customCudaAddNCBiasApplyCScaleBiasReluNHWC(dIn, dIn, dNb, dScale, dBias, dOut, n, xy, c, 0);
  compare("x + nb, in place", download(dIn, size), wantSum);
  compare("relu(scale * (x + nb) + bias), in place", download(dOut, size), want);
  cudaFree(dIn);
  cudaFree(dNb);
  cudaFree(dScale);
  cudaFree(dBias);
  cudaFree(dOut);
}

}  // namespace

int main() {
  if(!CudaFusedConv::supportedOnCurrentDevice()) {
    printf("CudaFusedConv::supportedOnCurrentDevice() is false: FAIL\n");
    return 1;
  }
  const bool shapes = CudaFusedConv::supportsShape(128, 128, 3, 3) && CudaFusedConv::supportsShape(96, 128, 3, 3)
    && !CudaFusedConv::supportsShape(100, 128, 3, 3) && !CudaFusedConv::supportsShape(128, 100, 3, 3);
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
  printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
