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
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from backend import QchocolateAcceleratorBackend
from tvm.relay import transform
from collections import OrderedDict
import numpy as np

import sys
import logging


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)

def make_int_configuration(
    requantize_output=False,
    batch_size=1
):
    
    x_shape, y_shape, output_shape = (batch_size, 1, 16), (batch_size, 16, 16), (batch_size, 1, 16)
    x_zero_point = -123
    y_zero_point = -123
    in_dtype = "int8"
    out_dtype = "int32" if not requantize_output else "int8"

    quantized_x_np = (
        np.array(
            [
                1,
                3,
                5,
                7,
                9,  # sum = 25
                11,
                13,
                15,
                -19,
                -21,  # sum = -1
                1,
                3,
                5,
                7,
                9,  # sum = 25
                11,
            ]
        )[  # sum = 3
            np.newaxis, np.newaxis, :
        ]
        .repeat(batch_size, axis=1)
        .astype(in_dtype)
        .reshape(x_shape)
    )
    initial_array = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 1, 3, 5, 7, 9, 1]).reshape(1, 16)
    array_16x16 = initial_array * np.ones((16, 16), dtype=initial_array.dtype)
    quantized_y_np = (
        array_16x16
        .repeat(batch_size, axis=1)
        .astype(in_dtype)
        .reshape(y_shape)
    )
    x_scale = 0.5
    y_scale = 0.5
    output_scale = 2.0

    config = {
        "quantized_x": quantized_x_np,
        "quantized_y": quantized_y_np,
        "dtype": in_dtype,
        "x_shape": x_shape,
        "y_shape": y_shape,
        "x_zero_point": x_zero_point,
        "y_zero_point": y_zero_point,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "out_dtype": out_dtype,
    }

    return config



def create_model(runner=AOT_DEFAULT_RUNNER):
    cfg = make_int_configuration()

    in_dtype = cfg["dtype"]
    out_dtype = cfg["out_dtype"]
    quantized_x_name = "data"
    quantized_y_name = "kernel"
    expected_out_dtype = cfg["out_dtype"]
    quantized_x = relay.var(quantized_x_name, shape=cfg["x_shape"], dtype=in_dtype)
    quantized_y = relay.var(quantized_y_name, shape=cfg["y_shape"], dtype=in_dtype)
    mod = relay.qnn.batch_matmul(
        quantized_x,
        quantized_y,
        relay.const(cfg["x_zero_point"], "int32"),
        relay.const(cfg["y_zero_point"], "int32"),
        relay.const(cfg["x_scale"], "float32"),
        relay.const(cfg["y_scale"], "float32"),
    )
    mod = relay.Function(relay.analysis.free_vars(mod), mod)
    mod = tvm.IRModule.from_expr(mod)
    # mod = relay.transform.InferType()(mod)

    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    
    #mod = transform.InferType()(mod)
    i_data = np.random.uniform(-20, 20, cfg["x_shape"]).astype(in_dtype)
    w1_data = np.random.uniform(0, 20, cfg["y_shape"]).astype(in_dtype)
    inputs = OrderedDict([("data", i_data), ("kernel", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner


def create_dense_model(runner=AOT_DEFAULT_RUNNER, tile=False):
    ci = 4 if tile else 1
    cw = 5 if tile else 1
    x_shape, y_shape, output_shape = (ci, 16), (16*cw, 16), (ci, 16*cw)

    x_zero_point = -123
    y_zero_point = -123
    in_dtype = "int8"
    out_dtype = "int32"
    x_scale = 0.5
    y_scale = 0.5
    
    
    quantized_x_name = "data"
    quantized_y_name = "kernel"
    
    quantized_x = relay.var(quantized_x_name, shape=x_shape, dtype=in_dtype)
    quantized_y = relay.var(quantized_y_name, shape=y_shape, dtype=in_dtype)
    mod = relay.qnn.dense(
        quantized_x,
        quantized_y,
        relay.const(x_zero_point, "int32"),
        relay.const(y_zero_point, "int32"),
        relay.const(x_scale, "float32"),
        relay.const(y_scale, "float32"),
        units=y_shape[0],
    )
    mod = relay.Function(relay.analysis.free_vars(mod), mod)
    mod = tvm.IRModule.from_expr(mod)
    # mod = relay.transform.InferType()(mod)

    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    
    #mod = transform.InferType()(mod)
    i_data = np.random.uniform(-20, 20, x_shape).astype(in_dtype)
    w1_data = np.random.uniform(0, 20, y_shape).astype(in_dtype)
    inputs = OrderedDict([("data", i_data), ("kernel", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner


def main():
    # mod, inputs, output_list, runner = create_model()
    mod, inputs, output_list, runner = create_dense_model(tile=False)
   

    uma_backend = QchocolateAcceleratorBackend()
    uma_backend.register()
    print(f"mod before partitioning: \n {mod}")
    mod = uma_backend.partition(mod)
    print(f"mod after partitioning: \n {mod}")
    target = tvm.target.Target("qchocolate_accelerator", host=tvm.target.Target("c"))
    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    print(f"Generated files are in {export_directory}")
    logging.getLogger("te_compiler").setLevel(logging.INFO)
    logging.getLogger("te_compiler").addHandler(logging.StreamHandler(sys.stdout))
    compile_and_run(
        AOTModel(module=mod, inputs=inputs, outputs=output_list),
        runner,
        interface_api="c",
        use_unpacked_api=True,
        target=target,
        test_dir=str(export_directory),
    )


if __name__ == "__main__":
    main()