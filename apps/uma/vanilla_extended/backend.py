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
"""UMA backend for the vanilla_extended accelerator"""

from passes import (VanillaExtendedConv2dPass,
                    VanillaExtendedDense,
                    VanillaExtendedDepthwiseConv2dPass,
                    VanillaExtendedReluPass,
                    ConvertLayout)
from patterns import (conv2d_pattern,
                      dense_pattern,
                      depthwise_conv2d_pattern,
                      relu_pattern
                      )
from codegen import gen_includes
from tvm.relay.backend.contrib.uma.api.utils import PassPhase
from tvm.relay.backend.contrib.uma.backend import UMABackend


class VanillaExtendedBackend(UMABackend):
    """UMA backend for the VanillaExtended accelerator."""

    def __init__(self):
        super().__init__()

        # Target configuration
        self._register_target_attr("dimension")

        # Relay Pattern registration
        self._register_pattern("conv2d", conv2d_pattern())
        self._register_pattern("dense", dense_pattern())
        self._register_pattern("depthwise_conv2d", depthwise_conv2d_pattern())
        self._register_pattern("relu", relu_pattern())

        # Relay to Relay function registration
        self._register_relay_pass(PassPhase.PRE_PARTITIONING, ConvertLayout()) # Needed for tflite

        # Relay to TIR function registration
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaExtendedConv2dPass())
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaExtendedDense())
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaExtendedDepthwiseConv2dPass())
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaExtendedReluPass())

        # TIR to runtime function registration
        self._register_codegen(fmt="c", includes=gen_includes)

    @property
    def target_name(self):
        "Return accelerator name"
        return "vanilla_extended"
