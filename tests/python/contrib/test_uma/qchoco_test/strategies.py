# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Strategies for the qchocolate_accelerator accelerator"""

# Example how to integrate a custom conv1d strategy:

# @relay.op.strategy.override_native_generic_func("custom_conv1d_strategy")
# def custom_conv1d_strategy(attrs, inputs, out_type, target):
#     strategy = _op.OpStrategy()
#     strategy.add_implementation(
#         wrap_compute_conv1d(custom_conv1d_compute),
#         wrap_topi_schedule(custom_conv1d_schedule),
#         name="custom_conv1d.generic",
#     return strategy
#

# For further details see:
# - github.com/apache/tvm-rfcs/blob/main/rfcs/0060_UMA_Unified_Modular_Accelerator_Interface.md
# - $TVM_HOME/python/tvm/relay/op/strategy/x86.py
import tvm
from tvm import relay, te
from tvm.relay import op as _op
from tvm.relay.qnn.strategy.hexagon import *
from tvm import topi
from tvm.topi.utils import get_const_tuple
from tvm.topi.hexagon.utils import is_scalar


@relay.op.strategy.override_native_generic_func("qnn_batch_matmul_strategy")
def qnn_batch_matmul_strategy(attrs, inputs, out_type, target):
    print("strategy for batch matmul...")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_qnn_batch_matmul(topi.hexagon.qnn_batch_matmul),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_batch_matmul),
    )
    
    return strategy

@relay.op.strategy.override_native_generic_func("qnn_dense_strategy")
def qnn_dense_strategy(attrs, inputs, out_type, target):
    print("strategy for dense...")
    
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_qnn_dense(topi.hexagon.qnn_dense),
        wrap_topi_schedule(topi.hexagon.schedule_qnn_dense),
        name="my_qnn_dense.hexagon",
    )
    return strategy