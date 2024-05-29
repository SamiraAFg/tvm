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
"""UMA backend for the qchocolate_accelerator accelerator"""
from passes import QchocolateAcceleratorBatchMatMulPass, QchocolateAcceleratorDensePass
from tvm.relay.backend.contrib.uma.api.utils import PassPhase
from tvm.relay.backend.contrib.uma.backend import UMABackend
from codegen import gen_includes
from patterns import qnn_batch_matmul_pattern, qnn_dense_pattern
from strategies import qnn_batch_matmul_strategy, qnn_dense_strategy




class QchocolateAcceleratorBackend(UMABackend):
    """UMA backend for the QchocolateAccelerator accelerator."""

    def __init__(self):
        super().__init__()

        # Target configuration
        self._register_target_attr("dimension")

        # Relay Pattern registration
        # self._register_pattern("qnn_batch_matmul", qnn_batch_matmul_pattern())
        
        self._register_pattern("qnn_dense", qnn_dense_pattern())

        # Relay to TIR function registration

        # self._register_operator_strategy("qnn.batch_matmul", qnn_batch_matmul_strategy)
        self._register_operator_strategy("qnn.dense", qnn_dense_strategy)
        
        # self._register_tir_pass(PassPhase.TIR_PHASE_0, QchocolateAcceleratorBatchMatMulPass())
        self._register_tir_pass(PassPhase.TIR_PHASE_0, QchocolateAcceleratorDensePass())

        # TIR to runtime function registration
        self._register_codegen(fmt="c", includes=gen_includes)

    @property
    def target_name(self):
        return "qchocolate_accelerator"
