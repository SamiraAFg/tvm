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
 * \file relay/backend/contrib/uma/utils.cc
 * \brief Utilities for microNPU codegen
 */

#include "utils.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/usmp/utils.h>

#include <utility>

namespace tvm {
namespace relay {
namespace contrib {
namespace uma {


CompilationArtifact::CompilationArtifact(String function_name, Array<Integer> command_stream,
                                         Array<Integer> zero_points,
                                         double scale,
                                         Array<String> base_addresses) {
  auto compilation_artifact_node = make_object<CompilationArtifactNode>();
  compilation_artifact_node->function_name = function_name;
  compilation_artifact_node->command_stream = command_stream;
  compilation_artifact_node->zero_points = zero_points;
  compilation_artifact_node->scale = scale;
  compilation_artifact_node->base_addresses = base_addresses;
  data_ = std::move(compilation_artifact_node);
}

TVM_REGISTER_NODE_TYPE(CompilationArtifactNode);
TVM_REGISTER_GLOBAL("relay.ext.uma.CompilationArtifact")
    .set_body_typed([](String function_name, Array<Integer> command_stream, Array<Integer> zero_points, double scale,
                       Array<String> base_addresses) {
      return CompilationArtifact(function_name, command_stream, zero_points, scale, base_addresses);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CompilationArtifactNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CompilationArtifactNode*>(ref.get());
      p->stream << "CompilationArtifactNode(\n"
                << "function_name=" << node->function_name
                << ",\n  command_stream=" << node->command_stream
                << ",\n  zero_points=" << node->zero_points
                << ",\n  scale=" << node->scale
                << ",\n  base_addresses=" << node->base_addresses << ")";
    });

}  // namespace uma
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
