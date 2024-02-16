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
"""Transform passes for the q_vanilla_accelerator accelerator"""

from functools import reduce
import tvm
from tvm import tir
from tvm.relay.backend.contrib.uma.api.utils import add_llvm_to_block
from tvm import relay
from tvm.tir import buffer


@tvm.tir.transform.prim_func_pass(opt_level=2)
class QVanillaAcceleratorConv2dPass:
    """
      detects the matched block, extracts the required arguments and maps it to
      the external call function (the respective accelerator kernel).

      _EXTERNAL_FUNCTION_NAME : name of the accelerator kernel defined in the included 
      interface file in codegen.
      _TVM_BLOCK_MATCH_NAME : name of the matched TIR block in PrimFunc (based on the strategy)

    """
    _EXTERNAL_FUNCTION_NAME = "q_vanilla_accelerator_conv2dnchw"
    _TVM_BLOCK_MATCH_NAME = "compute_2"
    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        
        return self._q_vanilla_accelerator_conv2d_pass(func, mod, ctx)

    @classmethod
    def _q_vanilla_accelerator_conv2d_pass(cls, func, mod, ctx):
        _loops = dict()
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

        def _detect_and_replace_conv2d(func: tvm.tir.PrimFunc, 
                                       mod: tvm.ir.IRModule, 
                                       ctx: tvm.ir.transform.PassContext) -> tvm.tir.PrimFunc:
            def _replace_conv2d(op):
                if op == _entry_node:
                    irb = tvm.tir.ir_builder.create()
                    # Collection of buffer address
                    buffers = [b[1].data for b in _handles]
                    # extraction of loop offsets
                    for k, v in _loops.items():
                        assert v.min.value == 0
                    offset_order = ["co", "w", "h", "ci", "kh", "kw"]
                    offsets = [_loops[i].extent.value for i in offset_order]
                    
                    offsets.append(zp[0])
                    offsets.append(zp[1])

                    args = buffers + offsets
                    print(f"Your operator will be called with these arguments : ({', '.join(str(arg) for arg in args)})")
                    irb.emit(tir_call(irb, True, cls._EXTERNAL_FUNCTION_NAME, *args))
                    irb_result = irb.get()
                   
                    return irb_result
                elif isinstance(op, tvm.tir.SeqStmt):
                    # Remove that pad block of TOPI's conv2DNCHW by only returning the 2nd statement
                    return op.seq[block_idx]   # the line that I've changed to replace the compute_2 block
                
                return op

            sch = tir.Schedule(func)

            if _has_block(cls._TVM_BLOCK_MATCH_NAME, func):
                #find the zp values
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
                block_idx = len(s1) - 3      

                conv2d_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                rv_loops = sch.get_loops(conv2d_block)
                
                assert len(rv_loops) == 7
                loops = dict(
                    n=rv_loops[0],
                    co=rv_loops[1],
                    h=rv_loops[2],
                    w=rv_loops[3],
                    ci=rv_loops[4],
                    kh=rv_loops[5],
                    kw=rv_loops[6],
                )
                _entry_node = sch.get(rv_loops[1])
                
                _loops = {k: sch.get(v) for k, v in loops.items()}
                _handles = func.buffer_map.items()
                
                kh_val = _loops["kh"].extent.value          
                kw_val = _loops["kw"].extent.value         
                if (kh_val % 2 == 0) and (kw_val % 2 == 0):
                    print("Even kernel sizes are not supported in QVanilla. Change your kernel_size to odd numbers.")
                    return func

                x = tvm.tir.stmt_functor.ir_transform(func.body, 
                                                      None, 
                                                      _replace_conv2d, 
                                                      ["tir.For", "tir.SeqStmt"]
                                                      )
                print("Successfully replaced the TVM operator by your operator in TIR... ")
                return func.with_body(x)
            else:
                print(f"TVM didn't find a tvm.tir.Block that match with '{cls._EXTERNAL_FUNCTION_NAME}()' function...")
                return func

        r = _detect_and_replace_conv2d(func, mod, ctx)
        return r

@tvm.tir.transform.prim_func_pass(opt_level=2)
class QVanillaAcceleratorDensePass:
    _EXTERNAL_FUNCTION_NAME = "q_vanilla_accelerator_dense"
    _TVM_BLOCK_MATCH_NAME = "compute"

    def transform_function(self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext) -> tvm.tir.PrimFunc:
        return self._q_vanilla_accelerator_dense_pass(func, mod, ctx)

    @classmethod
    def _q_vanilla_accelerator_dense_pass(cls, func, mod, ctx):
        _loops = dict()
        _handles = []
        _entry_node = None
        zp = []
        
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

        def _detect_and_replace_dense(func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, 
                                      ctx: tvm.ir.transform.PassContext) -> tvm.tir.PrimFunc:
            def _replace_dense(op):
                if op == _entry_node:
                    irb = tvm.tir.ir_builder.create()
                    # Collection of buffer (inputs/outputs) address
                    buffers = [b[1].data for b in _handles]
                    # extraction of loop offsets
                    for k, v in _loops.items(): # All iterators start in 0
                        assert v.min.value == 0
                    offset_order = ["ilen", "olen", "hidden"] 
                    offsets = [_loops[i].extent.value for i in offset_order]
                    offsets.append(zp[0])
                    offsets.append(zp[1])
                    args = buffers + offsets
                    
                    irb.emit(tir_call(irb, True, cls._EXTERNAL_FUNCTION_NAME, *args))
                    irb_result = irb.get()
                    return irb_result
                elif isinstance(op, tvm.tir.SeqStmt): 
                    # remove constant_engines_call and T_add
                    return op.seq[block_idx] 
                return op

            sch = tir.Schedule(func)
            
            if _has_block(cls._TVM_BLOCK_MATCH_NAME, func):
                # find the zp values
                s1 = []
                s2 = []
                def _visit(s):
                    if isinstance(s, tvm.tir.BufferStore):
                        s1.append(s.buffer.data)
                        s2.append(s.value)
                tvm.tir.stmt_functor.post_order_visit(func.body, _visit)

                for i in range(len(s1)):
                    if s1[i].name == "compile_engine_const":
                        zp.append(s2[i])
                block_idx = len(s1) - 3   
                dense_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                rv_loops = sch.get_loops(dense_block)

                assert len(rv_loops) == 3
                loops = dict(hidden=rv_loops[0],
                             olen=rv_loops[1],
                             ilen=rv_loops[2],)
                
                _entry_node = sch.get(rv_loops[0])
                _loops = {k: sch.get(v) for k, v in loops.items()}
                _handles = func.buffer_map.items()

                x = tvm.tir.stmt_functor.ir_transform(func.body, 
                                                      None, 
                                                      _replace_dense, 
                                                      ["tir.For", "tir.SeqStmt"])
                return func.with_body(x)
            else:
                return func
        r = _detect_and_replace_dense(func, mod, ctx)
        return r


#TODO(DSL): This pass still not working properly. Adapt for qnn depthconv!!
@tvm.tir.transform.prim_func_pass(opt_level=2)
class QVanillaAcceleratorDepthwiseConv2dPass:
    _EXTERNAL_FUNCTION_NAME = "q_vanilla_accelerator_depthconv2dnchw"
    _TVM_BLOCK_MATCH_NAME = "compute_2"

    def transform_function(self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext) -> tvm.tir.PrimFunc:
        return self._q_vanilla_accelerator_depthconv2d_pass(func, mod, ctx)
        
    @classmethod
    def _q_vanilla_accelerator_depthconv2d_pass(cls, func, mod, ctx):
        _loops = dict()
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

        def _detect_and_replace_depthconv2d(func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext) -> tvm.tir.PrimFunc:
            def _replace_depthconv2d(op):
                if op == _entry_node:
                    irb = tvm.tir.ir_builder.create()
                    # Collection of buffer address
                    buffers = [b[1].data for b in _handles]
                    # extraction of loop offsets
                    for k, v in _loops.items():
                        assert v.min.value == 0
                    offset_order = ["b", "c", "i", "j", "di", "dj"]
                    offsets = [_loops[i].extent.value for i in offset_order]
                    offsets.append(zp[0])
                    offsets.append(zp[1])

                    args = buffers + offsets
                    irb.emit(tir_call(irb, True, cls._EXTERNAL_FUNCTION_NAME, *args))
                    irb_result = irb.get()
                    return irb_result
                elif isinstance(op, tvm.tir.SeqStmt):
                    # Remove that pad block of TOPI's conv2DNCHW by only returning the 2nd statement
                    return op.seq[1]
                return op

            sch = tir.Schedule(func)

            if _has_block(cls._TVM_BLOCK_MATCH_NAME, func):
                #find the zp values
                s1 = []
                s2 = []
                def _visit(s):
                    if isinstance(s, tvm.tir.BufferStore):
                        s1.append(s.buffer.data)
                        s2.append(s.value)

                tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
                for i in range(len(s1)):
                    if s1[i].name == "compile_engine_const":
                        zp.append(s2[i])
                block_idx = len(s1) - 3      
                
                conv2d_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                rv_loops = sch.get_loops(conv2d_block)
                print("&%$§"*100)
                print( len(rv_loops) )
                print(rv_loops)
                assert len(rv_loops) == 7
                loops = dict(
                    n=rv_loops[0],
                    b=rv_loops[1],
                    c=rv_loops[2],
                    i=rv_loops[3],
                    j=rv_loops[4],
                    di=rv_loops[5],
                    dj=rv_loops[6],
                )
                _entry_node = sch.get(rv_loops[1]) 
                _loops = {k: sch.get(v) for k, v in loops.items()}
                _handles = func.buffer_map.items()
                x = tvm.tir.stmt_functor.ir_transform(
                    func.body, None, _replace_depthconv2d, ["tir.For", "tir.SeqStmt"]
                )
                return func.with_body(x)
            else:
                return func
            
        print("BEFORE REPLACE")
        print(func)
        print("%&$$§"*100)
        r = _detect_and_replace_depthconv2d(func, mod, ctx)
        print("After replacing",func)
        print("%&$$§"*100)
        return r
    


@tvm.tir.transform.prim_func_pass(opt_level=2)
class QVanillaAcceleratorReluPass:
    # Name of the function in your_operator.c 
    _EXTERNAL_FUNCTION_NAME = "q_vanilla_accelerator_relu"
    
    #Name of TVM block to replace by your operator. Accepting tvm.tir.Block
    _TVM_BLOCK_MATCH_NAME = "compute"

    # If you want to pass size of input to your operator
    _ADD_INPUT_SIZE = True 

    def transform_function(self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, 
                           ctx: tvm.ir.transform.PassContext) -> tvm.tir.PrimFunc:
        return self._q_vanilla_accelerator_relu_pass(func, mod, ctx)

    @classmethod
    def _q_vanilla_accelerator_relu_pass(cls, func, mod, ctx):
        _loops = dict() #Store loops of TIR representation corresponding to their parameters. 
        # Ex : {function paramater (width, ...) -> str : corresponding For that a LoopRV evaluates to -> tvm.tir.schedule.schedule.LoopRV}
        _handles = [] #Buffer binding map.
        _entry_node = None 

        # If debugging
        print(f"TIR Representation : {func}")
        def _has_block(name: str, func: tvm.tir.PrimFunc) -> bool:
            #Determine of a tir.block with `name` exists in `func`
            def _hb(op):
                if isinstance(op, tvm.tir.Block):
                    _found_blocks.append(op.name_hint)

            _found_blocks = []
            tvm.tir.stmt_functor.post_order_visit(func.body, _hb)
            return name in _found_blocks

        def _detect_and_replace_q_vanilla_accelerator_relu(func: tvm.tir.PrimFunc, 
                                                           mod: tvm.ir.IRModule, 
                                                           ctx: tvm.ir.transform.PassContext
                                                           ) -> tvm.tir.PrimFunc:
            def _replace_q_vanilla_accelerator_relu(op):
                if op == _entry_node:
                    irb = tvm.tir.ir_builder.create()
                    # Collection of buffer address
                    buffers = [b[1].data for b in _handles]
                    # extraction of loop offsets
                    for k, v in _loops.items():
                        assert v.min.value == 0
             
                    # If you have to use specific order for offset (see Vanilla example for more details)
                    #offset_order = ["x"] # Refer to key on line loops = dict(x = rv_loops[0]) # Extract from the loops each parameters
                    #offsets = [_loops[i].extent.value for i in offset_order]
                    args = buffers
                    if(cls._ADD_INPUT_SIZE):
                        print([sch.get(size).extent for size in rv_loops])
                        input_tensor_size = [sch.get(size).extent for size in rv_loops]
                        input_tensor_size = reduce(lambda x, y: x * y, input_tensor_size)
                        print(input_tensor_size)
                        args += [input_tensor_size]

                    # If debugging
                    print(f"Your operator will be called with these arguments : ({', '.join(str(arg) for arg in args)})")
                    irb.emit(tir_call(irb, True, cls._EXTERNAL_FUNCTION_NAME, *args))
                    irb_result = irb.get()
                    return irb_result
                elif isinstance(op, tvm.tir.SeqStmt):
                    return op.seq[1]
                return op

            sch = tir.Schedule(func)

            if _has_block(cls._TVM_BLOCK_MATCH_NAME, func):
                # If debugging
                print(f"TVM find a tvm.tir.Block that match with '{cls._EXTERNAL_FUNCTION_NAME}()' function...")
                print(f"The tvm.tir.Block that match with {cls._EXTERNAL_FUNCTION_NAME}() is '{cls._TVM_BLOCK_MATCH_NAME}'...")
                                
                q_vanilla_accelerator_relu_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                rv_loops = sch.get_loops(q_vanilla_accelerator_relu_block)
                loops = dict(x = rv_loops[0]) # Extract from the loops each parameters

                _entry_node = sch.get(rv_loops[0])
                _loops = {k: sch.get(v) for k, v in loops.items()}
                _handles = func.buffer_map.items()

                x = tvm.tir.stmt_functor.ir_transform(func.body, 
                                                      None, 
                                                      _replace_q_vanilla_accelerator_relu, 
                                                      ["tir.For", "tir.SeqStmt"])
                print("Successfully replaced the TVM operator by your operator in TIR... ")
                return func.with_body(x)
            else:
                print(f"TVM didn't find a tvm.tir.Block that match with '{cls._EXTERNAL_FUNCTION_NAME}()' function...")
                return func

        r = _detect_and_replace_q_vanilla_accelerator_relu(func, mod, ctx)
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


@tvm.ir.transform.module_pass(opt_level=0)
class ConvertLayout:
    def transform_module(self, mod, ctx):
        # My pass functionality...
        desired_layouts = {'qnn.conv2d': ['NCHW', 'default'],
                           'qnn.dense': ['NHW', 'default'],
                           }
        # Convert the layout to NCHW
        # RemoveUnunsedFunctions is used to clean up the graph.
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

        return mod

@tvm.ir.transform.module_pass(opt_level=0)
class Canonicalize:
    def transform_module(self, mod, ctx):
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.qnn.transform.CanonicalizeOps()])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        
        return mod
        
        
