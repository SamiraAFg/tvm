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
"""Transform passes for the qchocolate_accelerator accelerator"""

import tvm
from tvm import tir
from tvm.relay.backend.contrib.uma.api.utils import add_llvm_to_block
import intrin

@tvm.tir.transform.prim_func_pass(opt_level=2)
class QchocolateAcceleratorBatchMatMulPass:
    _EXTERNAL_FUNCTION_NAME = "qchocolate_batch_matmul"
    _TVM_BLOCK_MATCH_NAME = "T_batch_matmul_NT"
    

    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        print(f"PrimFunc: \n {func}")
        return self._qchocolate_accelerator_batch_matmul_pass(func, mod, ctx)

    @classmethod
    def _qchocolate_accelerator_batch_matmul_pass(cls, func, mod, ctx):
        # _loops = dict()
        _handles = []
        _entry_node = None
        zp = []
        block_idx = 0

        def _has_block(name: str, func: tvm.tir.PrimFunc) -> bool:
            """
            Determine of a tir.block with `name` exists in `func`
            """

            def _hb(op):
                if isinstance(op, tvm.tir.Block):
                    _found_blocks.append(op.name_hint)

            _found_blocks = []
            tvm.tir.stmt_functor.post_order_visit(func.body, _hb)
            return name in _found_blocks

        def _detect_and_replace_batch_matmul(
            func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
        ) -> tvm.tir.PrimFunc:
            def _replace_batch_matmul(op):
                if op == _entry_node:
                    irb = tvm.tir.ir_builder.create()
                    # Collection of buffer address
                    buffers = [b[1].data for b in _handles]
                    print(f"buffers: {buffers[0].name}")
                    offsets = []
                    offsets.append(zp[0])
                    offsets.append(zp[1])
                    print(offsets)
                    args = buffers + offsets
                    print(f"args: {args}")
                    irb.emit(tir_call(irb, True, cls._EXTERNAL_FUNCTION_NAME, *args))
                    irb_result = irb.get()
                    print(f"irb_result: \n {irb_result}")
                    return irb_result
                elif isinstance(op, tvm.tir.SeqStmt):
                    # Remove that pad block of TOPI's conv2DNCHW by only returning the 2nd statement
                    return op.seq[block_idx]
                return op

            sch = tir.Schedule(func)

            if _has_block(cls._TVM_BLOCK_MATCH_NAME, func):

                s1 = []
                s2 = []

                def _visit(s):
                    if isinstance(s, tvm.tir.BufferStore):
                        # stores.append(s.value)
                        s1.append(s.buffer.data)
                        s2.append(s.value)
                tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
                for i in range(len(s1)):
                    if s1[i].name == "compile_engine_const":
                        zp.append(s2[i])         
                block_idx = len(s1) - 2
                ###

                batch_matmul_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                rv_loops = sch.get_loops(batch_matmul_block)
                assert len(rv_loops) == 4
                _entry_node = sch.get(rv_loops[1])
                _handles = func.buffer_map.items()
                print(f"_handles: {_handles}")
                x = tvm.tir.stmt_functor.ir_transform(
                    func.body, None, _replace_batch_matmul, ["tir.For", "tir.SeqStmt"]
                )
                print(f"func.body after replacing extern: \n {func.with_body(x)}")
                return func.with_body(x)
            else:
                return func

        r = _detect_and_replace_batch_matmul(func, mod, ctx)
        return r
    

@tvm.tir.transform.prim_func_pass(opt_level=2)
class QchocolateAcceleratorDensePass:
    _EXTERNAL_FUNCTION_NAME = "qchocolate_batch_matmul"
    _TVM_BLOCK_MATCH_NAME = "dense_compute"
    _INTRIN_NAME_2 = "qchocolate_gemv_intrin"
    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        return self._qchocolate_accelerator_dense_pass(func, mod, ctx)

    @classmethod
    def _qchocolate_accelerator_dense_pass(cls, func, mod, ctx):
        # _loops = dict()
        _handles = []
        _entry_node = None
        zp = []
        block_idx = 0

        def _has_block(name: str, func: tvm.tir.PrimFunc) -> bool:
            """
            Determine of a tir.block with `name` exists in `func`
            """

            def _hb(op):
                if isinstance(op, tvm.tir.Block):
                    _found_blocks.append(op.name_hint)

            _found_blocks = []
            tvm.tir.stmt_functor.post_order_visit(func.body, _hb)
            return name in _found_blocks

        def _detect_and_replace_dense(
            func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
        ) -> tvm.tir.PrimFunc:
            
            def _tile_dense(sch, lo, li):                    

                # ## tiling
                dense_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                x_a, y_a, k_a = sch.get_loops(dense_block)
                assert li % 16 == 0, "kernel size should be a multiple of 16"
                if lo > 1 and li > 16:
                    print("case 1")
                    x_a_o , x_a_i = sch.split(x_a, [None, 1])
                    y_a_o , y_a_i = sch.split(y_a, [None, 16])
                    # print("reorder: \n")
                    sch.reorder(x_a_o, y_a_o, x_a_i, y_a_i, k_a)
                    # sch.mod.show()
                    print("blockoize: \n")
                    tiled_blk = sch.blockize(x_a_i)
                    sch.mod.show()
                    # print("decompose_reduction: \n")
                    sch.decompose_reduction(tiled_blk, y_a_o)
                    # sch.mod.show()
                    sch.tensorize(x_a_i, cls._INTRIN_NAME_2)
                    print("after tensorization..")
                    sch.mod.show()
                    #unroll
                    sch.unroll(y_a_o)
                    sch.unroll(x_a_o)
                    # print("after unroll..")
                    # sch.mod.show()
                elif lo >= 1 and li == 16:
                    print("case 2")
                    x_a_o , x_a_i = sch.split(x_a, [None, 1])

                    sch.reorder(x_a_o, x_a_i, y_a, k_a)
                    tiled_blk = sch.blockize(x_a_i)
                    
                    sch.decompose_reduction(tiled_blk, x_a_o)
               
                    sch.tensorize(x_a_i, cls._INTRIN_NAME_2)

                    sch.unroll(x_a_o)
                elif lo == 1 and li > 16:
                    print("case 3")
                    y_a_o , y_a_i = sch.split(y_a, [None, 16])
              
                    sch.reorder(y_a_o, x_a, y_a_i, k_a)

                    tiled_blk = sch.blockize(x_a)

                    sch.decompose_reduction(tiled_blk, y_a_o)

                    sch.tensorize(x_a, cls._INTRIN_NAME_2)

                    sch.unroll(y_a_o)
                # elif lo == 1 and li == 16: #I removed this because it does not match the intrin
                #     print("case 4")
                #     sch.tensorize(x_a, cls._INTRIN_NAME_2)

                return sch



            sch = tir.Schedule(func)
            
            if _has_block(cls._TVM_BLOCK_MATCH_NAME, func):

                print("func before:")
                func.show()
                _handles = func.buffer_map.items()
                buffers = [b[1].shape for b in _handles]
                print(f"buffers: {buffers}")
                shapes = dict(
                    inp_h = buffers[0][0], 
                    inp_w = buffers[0][1], 
                    k_h = buffers[1][0],
                    k_w = buffers[1][1],
                )
                
                assert shapes["inp_w"] == 16, "the tiling of reduce_axis has not yet implemented plz change it to 16"
                assert shapes["k_w"] == 16, "the tiling of reduce_axis has not yet implemented plz change it to 16"

                # ## tiling
                sch_ = _tile_dense(sch, shapes["inp_h"], shapes["k_h"])
                func = sch_.mod["main"]
                print("after:\n")
                func.show()
                return func

            else:
                return func

        r = _detect_and_replace_dense(func, mod, ctx)
        return r

def tir_call(ib: tvm.tir.ir_builder, extern: bool, name: str, *args):
    """
    ib: ir_builder
    extern: bool
        True  --> tvm.tir.call_extern
        False --> tvm.tir.call_packed
    name: str
        function name
    *args:
        arguments for function call
    """

    def buf_from_array(ib, arr, dtype):
        # Allocate enough memory to store the whole array
        var = ib.allocate("int32", (len(arr),), scope="global")
        for i, v in enumerate(arr):
            var[i] = v
        # Declare a buffer, which is basically a view on the chunk of memory that we allocated
        buf = tvm.tir.decl_buffer((len(arr),), dtype, data=var, scope="global")
        return buf

    if extern:
        args = [i.data if isinstance(i, tvm.tir.Buffer) else i for i in args]
        return tvm.tir.call_extern("int32", name, *args)
    else:
        args = [
            buf_from_array(ib, i, "int32")
            if isinstance(i, (tuple, list, tvm.ir.container.Array))
            else i
            for i in args
        ]
        return tvm.tir.call_packed(name, *args)
