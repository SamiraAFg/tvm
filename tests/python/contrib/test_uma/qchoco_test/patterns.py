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
"""Relay graph patterns for the qchocolate_accelerator accelerator"""

from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant
from tvm import relay, tir

def qnn_dense_pattern():

    # n, m = relay.var("n", dtype="int64"), tir.Var("m", dtype="int64")
    
    pattern = is_op("qnn.dense")(wildcard(), wildcard(), is_constant(),
            is_constant(), is_constant(), is_constant(),)

    return pattern   

def qnn_batch_matmul_pattern():
    pattern = is_op("qnn.batch_matmul")(wildcard(), wildcard(), is_constant(),
                         is_constant(), is_constant(), is_constant(),)
    pattern = pattern.has_attr({"out_dtype":"int32", "transpose_b":True}) 
    return pattern

