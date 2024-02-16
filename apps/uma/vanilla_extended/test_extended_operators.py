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
from backend import VanillaExtendedBackend
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from tvm.relay import transform
from collections import OrderedDict
import numpy as np
import os

from tvm.testing.aot import (AOTTestModel as AOTModel,
                             generate_ref_data,
                             compile_and_run,)


def create_conv2d():
    "Create conv2d layer in relay to offload to vanilla accelerator"
    dtype = "float32"
    k = 3
    ishape = (1, 32, 14, 14)
    wshape = (32, 32, k, k)
    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)

    out = relay.nn.conv2d(data0, weight0, kernel_size=(k, k), padding=(1, 1), groups=1)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f

    mod = transform.InferType()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)

    return mod, inputs, output_list

def create_depthconv2d():
    "Create depth conv2d layer in relay to offload to vanilla accelerator"
    dtype = "float32"
    k = 3
    ishape = (1, 3, 14, 14)
    wshape = (3, 1, k, k)

    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(k, k),
                          padding=(1, 1), strides=(1,1),
                          groups=3, channels=3,
                          data_layout="NCHW", kernel_layout="OIHW",)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)

    i_data = np.random.uniform(1, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(1, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)

    return mod, inputs, output_list

def create_dense():
    "Create dense layer in relay to offload to vanilla accelerator"
    dtype = "float32"
    out_units = 4
    in_units = 2
    ishape = (5, in_units)

    wshape = (out_units, in_units)
    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.dense(data=data0, weight=weight0, units=out_units)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)

    return mod, inputs, output_list

def create_relu():
    "Create relu layer in relay to offload to vanilla accelerator"
    dtype = "float32"
    ishape = (2, 3, 4)

    data0 = relay.var("data", shape=ishape, dtype=dtype)
    out = relay.nn.relu(data=data0)
    main_f = relay.Function([data0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f

    i_data = np.random.uniform(-1, 1, ishape).astype(dtype)
    inputs = OrderedDict([("data", i_data)])
    output_list = generate_ref_data(mod, inputs)

    return mod, inputs, output_list

def offload_to_vanilla(mod, input_list, output_list, test_case, depthconv2d_flag):
    uma_backend = VanillaExtendedBackend(depthconv2d_flag)
    uma_backend.register()
    mod = uma_backend.partition(mod)

    target = tvm.target.Target("vanilla_extended", host=tvm.target.Target("c"))

    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    export_directory = os.path.join("./out_tflite/", test_case)
    if os.path.exists(export_directory):
        os.system("rm -rf " + export_directory)
    print(f"Generated files are in {export_directory}")

    aot_test_model = AOTModel(module=mod, inputs=input_list, outputs=output_list)
    test_runner = AOT_DEFAULT_RUNNER

    compile_and_run(aot_test_model,
                    test_runner,
                    interface_api="c",
                    use_unpacked_api=True,
                    target=target,
                    verbose=True,
                    test_dir=str(export_directory),)


def main():
    "Call relay operators and offload to VanillaExtended backend"
    mod, input_list, output_list = create_conv2d()
    offload_to_vanilla(mod, input_list, output_list, "conv2d", depthconv2d_flag=False)

    mod, input_list, output_list = create_depthconv2d()
    offload_to_vanilla(mod, input_list, output_list, "depthconv2d", depthconv2d_flag=True)

    mod, input_list, output_list = create_dense()
    offload_to_vanilla(mod, input_list, output_list, "dense", depthconv2d_flag=False)

    mod, input_list, output_list = create_relu()
    offload_to_vanilla(mod, input_list, output_list, "relu", depthconv2d_flag=False)

if __name__ == "__main__":
    main()
