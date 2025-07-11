// forloop_accum.h - Implementation of accumulating the output
#pragma once

#include "cute/config.hpp"
#include <cute/layout.hpp>
using namespace cute;

namespace tb {

// Clear the accumulator
// (Just fill the accumulator with zeros)
template <typename T, int NUM_ELEMS, int NUM_THREADS>
class ClearAccumlatorKernel {
public:
  static constexpr int GROUP_SIZE = 16 / sizeof(T);
  static_assert(NUM_ELEMS % GROUP_SIZE ==
                0); // NUM_ELEMS should always be multiple of GROUP_SIZE
                    // (guaranteed by layout resolution)

  static __device__ __forceinline__ void run(T *__restrict__ accum,
                                             int thread_idx) {
    uint128_t *accum_128 = reinterpret_cast<uint128_t *>(accum);
    for (int elem_idx = thread_idx; elem_idx < NUM_ELEMS / GROUP_SIZE;
         elem_idx += NUM_THREADS) {
      accum_128[elem_idx] = 0ul;
    }
  }
};

template <typename T, class AccumLayout, class SrcLayout, int NUM_THREADS>
class ForloopAccumKernel {
public:
  using Numel = decltype(size(AccumLayout{}));
  CUTE_STATIC_ASSERT_V(Numel{} == size(SrcLayout{}));

  // TODO(intlsy) Use half2
  static __device__ __forceinline__ void
      run(T *__restrict__ accum, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto accum_layout = AccumLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      accum[accum_layout(elem_idx)] += src[src_layout(elem_idx)];
    }
  }
};

// Only support matrix/vector
template <typename T,
          class AccumLayout,
          class SrcLayout,
          class RescaleLayout,
          int NUM_THREADS>
class ForloopAccumRescaleKernel {
public:
  using AccumNumel = decltype(size(AccumLayout{}));
  using RescaleNumel = decltype(size(RescaleLayout{}));

  CUTE_STATIC_ASSERT_V(AccumNumel{} == size(SrcLayout{}));
  CUTE_STATIC_ASSERT_V(RescaleNumel{} == size<0>(shape(AccumLayout{})));

  static __device__ __forceinline__ void run(T *__restrict__ accum,
                                             T const *__restrict__ src,
                                             T const *__restrict__ rescale,
                                             int thread_idx) {
    constexpr auto accum_layout = AccumLayout{};
    constexpr auto src_layout = SrcLayout{};
    constexpr auto rescale_layout = RescaleLayout{};

    constexpr auto numel = AccumNumel{};
    constexpr auto rescale_numel = RescaleNumel{};

    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      accum[accum_layout(elem_idx)] =
          accum[accum_layout(elem_idx)] *
              rescale[rescale_layout(elem_idx % rescale_numel)] +
          src[src_layout(elem_idx)];
    }
    // Print debug info
    // if (thread_idx == 0) {
    //   printf("**********ForloopAccumRescaleKernel**********\n"
    //          "accum[0]: %f, rescale[0]: %f, src[0]: %f\n"
    //          "---------------------------------------------\n",
    //          (float)accum[accum_layout(0)],
    //          (float)rescale[rescale_layout(0)],
    //          (float)src[src_layout(0)]);
    // }
  }
};

} // namespace tb
