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
# pylint: disable=use-list-literal, invalid-name
"""This source will contain code to convert TIR, as produced by
the Relay to TIR compilation process, to Vela API calls to
generate command stream.
"""
from typing import Dict, NamedTuple, Tuple, Union, List
from enum import auto
from enum import Enum
import numpy as np  # type: ignore


import tvm
# from tvm.tir import stmt_functor
from tvm.relay.backend.contrib.uma.api import utils
from tvm.relay.backend.contrib.uma.api.qchoco_cmds import (memcmd, gemcmd)
# from tvm.runtime import convert


@tvm._ffi.register_func("relay.ext.uma.primfunc_to_artifact")
def primfunc_to_artifact(primfunc: tvm.tir.PrimFunc) -> utils.CompilationArtifact:
    """
    This is the hook for python-based lowering of TIR PrimFunc
    that has undergone unified optimization to compilation
    artifact destined for the microNPU.

    Parameters
    ----------
    primfunc : tir.PrimFunc
        TIR PrimFunc that has undergone unified optimizations

    Returns
    -------
    CompilationArtifact
        This is a structure that holds the binary artifacts
        for the microNPU
    """
    primfunc.show()

    zp_args, scale = extract_zps(primfunc)
    print("MW"*10)
    print(scale)
    print(zp_args)

    symbol = str(primfunc.attrs["global_symbol"])
    # const_dict = primfunc.attrs["ethos-u.constants"]
    tir_mod = tvm.IRModule()
    tir_mod[symbol] = primfunc
    ## TODO: you probabely need to handle the args for multiple extern_list
    call_extern_list = extract_call_extern_list(tir_mod)
    print(f"#externs: {len(call_extern_list)}")
    # zp_args = list()
    base_addresses = list()
    if call_extern_list:
        extern_0 = call_extern_list[0]
        if extern_0.args[0] == "qchocolate_batch_matmul":
            print(f"call_extern.args in normal: {extern_0.args}")
            print(f"type of first arg: {type(extern_0.args[1])}")
            base_addresses.append(extern_0.args[1].name)
            base_addresses.append(extern_0.args[2].name)
            base_addresses.append(extern_0.args[3].name)
        cmd_stream = generate_command_stream_2(call_extern_list)
    # for call_extern in call_extern_list:
    #     if call_extern.args[0] == "qchocolate_batch_matmul":
    #         # for i in range(len(call_extern.args)):
    #         print(f"call_extern.args in normal: {call_extern.args}")
    #         print(f"type of first arg: {type(call_extern.args[1])}")
    #         base_addresses.append(call_extern.args[1].name)
    #         base_addresses.append(call_extern.args[2].name)
    #         base_addresses.append(call_extern.args[3].name)
    #         zp_args.append(call_extern.args[4].value)
    #         zp_args.append(call_extern.args[5].value)
    #         cmd_stream = generate_command_stream(call_extern.args[0])
    if cmd_stream is None:
        print("*"*20)
        print("no cmd_stream generated")
    else:
        print("*"*20)
        print(f"sizeof(cmd_stream): {len(cmd_stream)}")

    return utils.CompilationArtifact(symbol, cmd_stream, zp_args, scale, base_addresses)
           
     

def extract_call_extern_list(mod):
    """This function will obtain all extern
    calls from a TIR module
    Parameters
    ----------
    mod : tvm.IRModule
        The TIR Module for NPU

    Returns
    -------
    list
        of tvm.tir.Call objects
        that are tir extern calls
    """
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]

    call_extern_list = list()

    def populate_call_extern_list(stmt):
        if isinstance(stmt, tvm.tir.Call) and stmt.op.name == "tir.call_extern":
            call_extern_list.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(primfunc.body, populate_call_extern_list)
    return call_extern_list



# @tvm._ffi.register_func("relay.ext.uma.generate_command_stream")
def generate_command_stream(extern_name):
    cmd_stream = []
    # if extern_name == "qchocolate_batch_matmul"
        # cmd_stream.append(str(memcmd.LOAD_A.value))
        # cmd_stream.append(str(memcmd.LOAD_B.value))
        # cmd_stream.append(str(gemcmd.GEMM.value))
        # cmd_stream.append(str(memcmd.STORE_RES.value))
    cmd_stream.append(memcmd.LOAD_A.value)
    cmd_stream.append(memcmd.LOAD_B.value)
    cmd_stream.append(gemcmd.GEMM.value)
    cmd_stream.append(memcmd.STORE_RES.value)
    return cmd_stream

def generate_command_stream_2(extern_list):
    cmd_stream = []
    qchoco_load_uops = ("uop_load_inp", "uop_load_wgt", "uop_gemm", "uop_store")
    # cmd = 0
    for extern in extern_list:
        if extern.args[0] == qchoco_load_uops[0]:
            idx = extern.args[5].value & 0xFF
            cmd = memcmd.LOAD_A.value | (idx << 16)
            cmd_stream.append(cmd)
        elif extern.args[0] == qchoco_load_uops[1]:
            idx = extern.args[5].value & 0xFF
            cmd = memcmd.LOAD_B.value | (idx << 16)
            cmd_stream.append(cmd)
        elif extern.args[0] == qchoco_load_uops[2]:
            cmd_stream.append(gemcmd.GEMM.value)
        elif extern.args[0] == qchoco_load_uops[3]:
            idx = extern.args[5].value & 0xFF
            cmd = memcmd.STORE_RES.value | (idx << 16)
            cmd_stream.append(cmd)
        # if (cmd & 0x00FF0000) != 0:
        #     print(f"error: {cmd}")
    return cmd_stream


def extract_exp_zps(primfunc):
    s1 = []
    s2 = []
    zp = list()
    def _visit(s):
        if isinstance(s, tvm.tir.BufferStore):
            # stores.append(s.value)
            s1.append(s.buffer.data)
            s2.append(s.value)
    tvm.tir.stmt_functor.post_order_visit(primfunc.body, _visit)
    # print("-"*50)
    # print(s1)
    # print(s2)
    for i in range(len(s1)):
        if s1[i].name == "compile_engine_const_let":
            zp.append(s2[i])         
    print(f"zps: {zp}")
    return zp

def extract_zps(func):
    zps = list()
    attrs = func.attrs
    in1_zp = None
    in2_zp = None
    scale = None
    out_zp = None
    print(attrs)
    for key in attrs.keys():
        if key == "in1_zp":
            in1_zp = attrs[key]
        elif key == "in2_zp":
            in2_zp = attrs[key]
        elif key == "scale":
            scale = attrs[key].value
        elif key == "out_zp":
            out_zp = attrs[key]
    # assert in2_zp is not None, "in2_zp is None"
    # assert in1_zp is not None, "in1_zp is None"
    if in2_zp is not None:
        zps.append(in1_zp)
        zps.append(in2_zp)
        zps.append(out_zp)
    else: return extract_exp_zps(func), scale

    return zps, scale