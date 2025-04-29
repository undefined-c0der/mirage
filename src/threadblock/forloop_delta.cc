/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mirage/threadblock/forloop_delta.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

std::vector<STensor> Graph::forloop_delta(STensor const &input) {
  TBOperator *op = create_forloop_delta_op(input);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

std::vector<STensor *> Graph::forloop_delta(STensor const *input) {
  TBOperator *op = create_forloop_delta_op(*input);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 2);
  return std::vector<STensor *>{&op->output_tensors[0], &op->output_tensors[1]};
}

TBOperator *Graph::create_forloop_delta_op(STensor const &input) {
  TBOperator *op = new TBForloopDeltaOp(this, input);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
}

TBForloopDeltaOp::TBForloopDeltaOp(Graph *_graph, STensor const &input)
    : TBOperator(_graph, mirage::type::TB_FORLOOP_DELTA_OP, input) {
  STensor delta = input;
  delta.owner_op = this;
  delta.owner_ts_idx = 0;
  delta.guid = STensor::next_guid++;
  delta.after_accum = input.after_accum;
  delta.smem_offset = bgraph->allocate_fingerprint(delta);
  output_tensors.push_back(delta);
  STensor record = delta;
  record.owner_ts_idx = 1;
  record.guid = STensor::next_guid++;
  record.smem_offset = bgraph->allocate_fingerprint(record);
  output_tensors.push_back(record);
}

TBForloopDeltaOp::~TBForloopDeltaOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBForloopDeltaOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
