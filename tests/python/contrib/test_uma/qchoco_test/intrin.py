"""QChocolate related intrinsics"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm.script import tir as T

# memory scopes
inp_scope = "local.inp_buffer"
wgt_scope = "local.wgt_buffer"
acc_scope = "local.acc_buffer"

INP_WIDTH = 1 << 3
WGT_WIDTH = 1 << 3
ACC_WIDTH = 1 << 5
OUT_WIDTH = 1 << 5

_INTRIN_NAME_2 = "qchocolate_gemv_intrin"
#considering the instructions here...
@T.prim_func
def gemv16_desc_2(a: T.handle, b: T.handle, c: T.handle, A_zp: T.Buffer((), "int32"), B_zp: T.Buffer((), "int32")) -> None:
    A = T.match_buffer(a, (1 , 16), "int%d" % INP_WIDTH, offset_factor=1)
    B = T.match_buffer(b, (16, 16), "int%d" % WGT_WIDTH, offset_factor=1)
    C = T.match_buffer(c, (1 , 16), "int%d" % OUT_WIDTH, offset_factor=1)

    with T.block("root"):
        T.reads(C[0:1, 0:16], A[0:1, 0:16], A_zp[()], B[0:16, 0:16], B_zp[()])
        # T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:1, 0:16])
        for i, j, k in T.grid(1, 16, 16):
            with T.block(""):
                vjj, vkk = T.axis.remap("SR", [j, k])
                C[0, vjj] = C[0, vjj] + (T.Cast("int32", A[0, vkk]) - A_zp[()]) * (T.Cast("int32", B[vjj, vkk]) - B_zp[()])

@T.prim_func
def gemv16_impl_2(a: T.handle, b: T.handle, c: T.handle, A_zp: T.Buffer((), "int32"), B_zp: T.Buffer((), "int32")) -> None:
    sa = T.int32()
    sb = T.int32()
    sc = T.int32()
    A = T.match_buffer(a, (1, 16), "int%d" % INP_WIDTH, offset_factor=1, strides=[sa, 1])
    B = T.match_buffer(b, (16, 16), "int%d" % WGT_WIDTH, offset_factor=1, strides=[sb, 1])
    C = T.match_buffer(c, (1, 16), "int%d" % OUT_WIDTH, offset_factor=1, strides=[sc, 1])

    
    with T.block("root"):
        T.reads(C[0:1, 0:16], A[0:1, 0:16], A_zp[()], B[0:16, 0:16], B_zp[()])
        # T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:1, 0:16])
        T.evaluate(
            T.call_extern(
                "qchocolate_batch_matmul",
                A.data,
                B.data,
                C.data,
                dtype="int32",
            )
        )
        T.evaluate(
            T.call_extern(
                "uop_load_inp",
                0, #opcode
                0, #memory_type
                0, #sram_base
                A.data, #dram_base
                A.elem_offset // 16, #dram_base
                1, #size
                dtype="",
            )
        )
        T.evaluate(
            T.call_extern(
                "uop_load_wgt",
                0, #opcode
                1, #memory_type
                0, #sram_base
                B.data, #dram_base
                B.elem_offset // 256, #dram_base
                1, #size
                dtype="",
            )
        )
        T.evaluate(
            T.call_extern(
                "uop_gemm",
                2, #opcode
                0, #activation
                0, #bias
                0, #convolution
                dtype="",
            )
        )
        T.evaluate(
            T.call_extern(
                "uop_store",
                1, #opcode
                2, #memory_type ?
                0, #sram_base
                C.data, #dram_base
                C.elem_offset // 16, #dram_base
                1, #size
                dtype="int32",
            )
        )

tvm.tir.TensorIntrin.register(_INTRIN_NAME_2, gemv16_desc_2, gemv16_impl_2)