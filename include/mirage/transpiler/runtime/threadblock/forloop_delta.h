// forloop_delta.h - Implementation of computing the delta of the input between
// two iterations
#pragma once

#include "cute/config.hpp"
#include <cute/layout.hpp>
using namespace cute;

namespace tb {

// Clear the delta
// (Just fill the delta with zeros)
template <typename T, int NUM_ELEMS, int NUM_THREADS>
class ClearDeltaRecordKernel {
public:
  static constexpr int GROUP_SIZE = 16 / sizeof(T);
  static_assert(NUM_ELEMS % GROUP_SIZE ==
                0); // NUM_ELEMS should always be multiple of GROUP_SIZE
                    // (guaranteed by layout resolution)

  static __device__ __forceinline__ void run(T *__restrict__ record,
                                             int thread_idx) {
    uint128_t *record_128 = reinterpret_cast<uint128_t *>(record);
    for (int elem_idx = thread_idx; elem_idx < NUM_ELEMS / GROUP_SIZE;
         elem_idx += NUM_THREADS) {
      record_128[elem_idx] = 0ul;
    }
  }
};

template <typename T,
          class DeltaLayout,
          class RecordLayout,
          class SrcLayout,
          int NUM_THREADS>
class ForloopDeltaKernel {
public:
  using Numel = decltype(size(DeltaLayout{}));
  CUTE_STATIC_ASSERT_V(Numel{} == size(RecordLayout{}));
  CUTE_STATIC_ASSERT_V(Numel{} == size(SrcLayout{}));

  // TODO(intlsy) Use half2
  static __device__ __forceinline__ void run(T *__restrict__ delta,
                                             T *__restrict__ record,
                                             T const *__restrict__ src,
                                             int thread_idx) {
    constexpr auto numel = Numel{};
    auto delta_layout = DeltaLayout{};
    auto record_layout = RecordLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      delta[delta_layout(elem_idx)] =
          src[src_layout(elem_idx)] -
          record[delta_layout(elem_idx)]; // delta = src - record
      record[record_layout(elem_idx)] = src[src_layout(elem_idx)];
    }
  }
};

} // namespace tb