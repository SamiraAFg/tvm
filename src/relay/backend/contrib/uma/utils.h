/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file relay/backend/contrib/uma/utils.h
 * \brief Utilities for microNPU codegen
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_UMA_UTILS_H_
#define TVM_RELAY_BACKEND_CONTRIB_UMA_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace uma {

/*!
 * \brief Captures all the binary artifactes required to create
 * the C-source runtime module
 */
struct CompilationArtifactNode : public Object {
  /*! \brief The function name for this artifact belongs to */
  String function_name;
  /*! \brief The binary command stream (CS) in hex format */
  Array<Integer> command_stream;
  /*! \brief The encoded biases and weights in hex format */
  Array<Integer> zero_points;

  Array<String> base_addresses;
  

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("function_name", &function_name);
    v->Visit("command_stream", &command_stream);
    v->Visit("zero_points", &zero_points);
    v->Visit("base_addresses", &base_addresses);
  }

  bool SEqualReduce(const CompilationArtifactNode* other, SEqualReducer equal) const {
    return equal(function_name, other->function_name) &&
           equal(command_stream, other->command_stream) &&
           equal(zero_points, other->zero_points) &&
           equal(base_addresses, other->base_addresses);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(function_name);
    hash_reduce(command_stream);
    hash_reduce(zero_points);
    hash_reduce(base_addresses);
  }

  static constexpr const char* _type_key = "relay.ext.uma.CompilationArtifact";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompilationArtifactNode, Object);
};

class CompilationArtifact : public ObjectRef {
 public:
  TVM_DLL CompilationArtifact(String function_name, Array<Integer> command_stream, Array<Integer> zero_points,
                              Array<String> base_addresses);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CompilationArtifact, ObjectRef, CompilationArtifactNode);
};

}  // namespace uma
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_UMA_UTILS_H_
