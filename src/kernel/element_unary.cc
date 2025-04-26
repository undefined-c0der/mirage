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

#include "mirage/kernel/element_unary.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

DTensor Graph::exp(DTensor const &input) {
  return elementunary(input, mirage::type::KN_EXP_OP);
}

DTensor *Graph::exp(DTensor const *input) {
  return elementunary(input, mirage::type::KN_EXP_OP);
}

DTensor Graph::square(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SQUARE_OP);
}

DTensor *Graph::square(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SQUARE_OP);
}

DTensor Graph::sqrt(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SQRT_OP);
}

DTensor *Graph::sqrt(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SQRT_OP);
}

DTensor Graph::silu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SILU_OP);
}

DTensor *Graph::silu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SILU_OP);
}

DTensor Graph::gelu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_GELU_OP);
}

DTensor *Graph::gelu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_GELU_OP);
}

DTensor Graph::relu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_RELU_OP);
}

DTensor *Graph::relu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_RELU_OP);
}

DTensor Graph::clamp(DTensor const &input,
                     float const &min_val,
                     float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor *Graph::clamp(DTensor const *input,
                      float const &min_val,
                      float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor Graph::elementunary_clamp(DTensor const &input,
                                  float const &min_val,
                                  float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary_clamp(DTensor const *input,
                                   float const &min_val,
                                   float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(*input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_clamp_op(DTensor const &input,
                                                float const &min_val,
                                                float const &max_val) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNClampUnaryOp(this, input, min_val, max_val);

  return op;
}

DTensor Graph::mul_scalar(DTensor const &input, float const &scalar) {
  return elementunary(input, mirage::type::KN_MUL_SCALAR_OP, scalar);
}
DTensor *Graph::mul_scalar(DTensor const *input, float const &scalar) {
  return elementunary(input, mirage::type::KN_MUL_SCALAR_OP, scalar);
}

DTensor Graph::elementunary(DTensor const &input,
                            mirage::type::KNOperatorType type,
                            float const &scalar) {
  KNOperator *op = create_elementunary_op(input, type, scalar);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary(DTensor const *input,
                             mirage::type::KNOperatorType type,
                             float const &scalar) {
  KNOperator *op = create_elementunary_op(*input, type, scalar);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_op(DTensor const &input,
                                          mirage::type::KNOperatorType type,
                                          float const &scalar) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNElementUnaryOp(this, input, type, scalar);

  return op;
}

KNClampUnaryOp::KNClampUnaryOp(Graph *_kgraph,
                               DTensor const &input,
                               float min_val,
                               float max_val)
    : KNElementUnaryOp(_kgraph, input, mirage::type::KN_CLAMP_OP, 0.0f),
      min_val(min_val), max_val(max_val) {}

KNElementUnaryOp::KNElementUnaryOp(Graph *_kgraph,
                                   DTensor const &input,
                                   mirage::type::KNOperatorType type,
                                   float const &scalar)
    : mirage::kernel::KNOperator(_kgraph, type, input), scalar(scalar) {
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementUnaryOp::~KNElementUnaryOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace mirage
