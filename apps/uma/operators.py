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
"""Registration of accelerator-specific operators of the custom accelerator in Relay level"""

import os
import yaml



#operator should contain acc name
#TODO:which libs are necessary?
#TODO add the array attrs
cpp_template =""" /*!
 * \\file src/relay/op/contrib/{acc_name}/{op_name}.cc
 * \\brief Operator definitions for the {acc_name} {op_name} ops.
 */
#include <tvm/relay/op.h>
#include "../../../qnn/utils.h"
#include "../../op_common.h"

namespace tvm {{
namespace relay {{
namespace op {{
namespace contrib {{
namespace {acc_name} {{

{attr_node}

{type_relation}

{relay_call_node}

{register_op}

}}  // namespace {acc_name}
}}  // namespace contrib
}}  // namespace op
}}  // namespace relay
}}  // namespace tvm

"""
##
attr_node_template = """
/*! \\brief Attributes used in {AccOperator_name} operator */
struct {AccOperator_name}Attrs : public tvm::AttrsNode<{AccOperator_name}Attrs> {{
  {attr_fields};
  TVM_DECLARE_ATTRS({AccOperator_name}Attrs, "relay.attrs.{AccOperator_name}Attrs") {{
    {attr_descriptions}
  }}
}};

TVM_REGISTER_NODE_TYPE({AccOperator_name}Attrs);"""

##
type_relation_template = """
bool {AccOperator_name}Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {{
    // types: 
    ICHECK_EQ(types.size(), {num_types}) << "Expects {num_types} types, {num_inps} for the input and another for the output";
    {inp_type_check}

    const auto* param = attrs.as<{AccOperator_name}Attrs>();
    ICHECK(param != nullptr) << "{AccOperator_name}Attrs cannot be nullptr.";
    {output_type_shape_assign}
    return true;
}}"""

##
relay_call_node_template = """
Expr Make{AccOperator_name}({inp_args}, {attr_args}) {{
    auto attrs = make_object<{AccOperator_name}Attrs>();
    {attrs_bind};
    static const Op& op = Op::Get("{operator_name_withDots}");
    return Call(op, {{{inpSet}}}, Attrs(attrs), {{}});
}}
TVM_REGISTER_GLOBAL("relay.op._make.{py_name}").set_body_typed(Make{AccOperator_name});"""
##

register_op_template = """
RELAY_REGISTER_OP("{operator_name_withDots}")
    .describe(
        R"doc({op_description})doc" TVM_ADD_FILELINE)
    .set_num_inputs({num_inps})
    {add_arg_methods}
    .set_support_level({op_support_level})
    .add_type_rel("{AccOperator_name}", {AccOperator_name}Rel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);"""

inp_check_type_template = """
    const auto* {inp_name} = types[{inp_idx}].as<TensorTypeNode>();
    if ({inp_name} == nullptr) return false;"""

ops_python_template = """
import tvm  # type: ignore
from tvm.relay.op import _make  # type: ignore

def {py_name}(
    {py_args_with_annotations}
) -> tvm.relay.Call:

    return _make.{py_name}(
    {all_args}
    )
"""

class UMAOperators(object):
    """
    Handling of new Relay operators in the Universal Modular Accelerator Interface (UMA)
    """

    def __init__(self, target_name: str, hw_path: str) -> None:
        self.target_name = target_name
        self.hw_path = hw_path
        self.accOpName = None
        self.OpName = None
        

    def _import_ops_config(
        self, yml_file_name: str
    ) -> None:
        """Exract operators specification to register in TVM Relay operators.

        Parameters
        ----------
        yml_file_name: str
            The YAML file that contains the data and is located in the accelerator backend directory.
        
        """
        if yml_file_name is not None:
            with open(yml_file_name, 'r') as f:
                operators_config = yaml.safe_load(f)
            for operator in operators_config['operators']:
                self._add_operator(operator)
        else:
            raise RuntimeError("There is no name for YAML file: if the accelerator supports different primitive operators, you should fill operators.yml out") #?? type of the error

    def _make_attr_node(self, attrs) -> str: # TODO: add asserts for make sure all elements were defined
        attr_node = ""
        attr_descs = ""
        attr_fields = (f"{attr['type']} {attr['name']}" for attr in attrs)
        for attr in attrs:
            if attr['default'] is not None:
                default_value = attr['default'] if attr['type'] != "String" else f"\"{attr['default']}\""
                attr_descs += f"TVM_ATTR_FIELD({attr['name']}).describe(\"{attr['description']}\").set_default({default_value});\n    "
            else:
                attr_descs += f"TVM_ATTR_FIELD({attr['name']}).describe(\"{attr['description']}\");\n    "
        attr_node = attr_node_template.format(
            AccOperator_name = self.accOpName, 
            attr_fields = ";\n  ".join(attr_fields),
            attr_descriptions = attr_descs
        )
        return attr_node

    def _make_type_rel(self, inputs, output) -> str:
        type_rel = ""
        num_inps = len(inputs)

        inp_type_checks = "\n  ".join(inp_check_type_template.format(
            inp_name=ins['name'], inp_idx=idx
        ) for idx, ins in enumerate(inputs))

        if output['dtype'] is not None:
            define_dtype = f"DataType out_dtype = DataType::{output['dtype']};\n    "
        else:
            raise RuntimeError("Please specify the output dtype")
        
        #TODO: think about how to apply type rules
        #TODO: add functions for handling rules for inputs like scale_bias in ETHOS
        #TODO: more important: what if there will be if-else for assigning output type and shape?
        out_shape_rule = output['shape_rule']
        assign_expr = f"reporter->Assign(types[{num_inps}], TensorType(out_shape, out_dtype));"
        type_rel = type_relation_template.format(
            AccOperator_name = self.accOpName,
            num_types = num_inps + 1, #TODO: always one output?
            num_inps = num_inps,
            inp_type_check = inp_type_checks,
            output_type_shape_assign = define_dtype + out_shape_rule + "\n    " + assign_expr
        )

        return type_rel


    def _make_relay_call_node(self, inputs, attrs) -> str:
        relay_call_node = ""
        attr_fields = (f"{attr['type']} {attr['name']}" for attr in attrs)

        tmp1 = ", ".join(f"Expr {ins['name']}" for ins in inputs)
        tmp2 = ", ".join(attr_fields)
        tmp3 = ";\n    ".join(f"attrs->{attr['name']} = {attr['name']}" for attr in attrs)
        tmp4 = ", ".join(f"{ins['name']}" for ins in inputs)

        relay_call_node = relay_call_node_template.format(
           AccOperator_name = self.accOpName,
           inp_args = tmp1,
           attr_args = tmp2,
           attrs_bind = tmp3,
           operator_name_withDots = "contrib." + self.target_name + "." + self.OpName,
           inpSet = tmp4, 
           py_name = self.target_name + "_" + self.OpName
        )

        return relay_call_node

    def _make_register_op(self, inputs, opDesc, opSupLe) -> str:
        register_op = ""
        tmp = "\n    ".join(f".add_argument(\"{ins['name']}\", \"{ins['type']}\", \"{ins['description']}\")" for ins in inputs)
        register_op = register_op_template.format(
            operator_name_withDots = "contrib." + self.target_name + "." + self.OpName,
            AccOperator_name = self.accOpName, 
            op_description = opDesc, 
            num_inps = len(inputs), 
            add_arg_methods = tmp,
            op_support_level = opSupLe
        )
        return register_op
    
    def _make_py_api_hook(self, inputs, attrs):

        py_hook = ""
        arg_names = []
        args = []
        for inp in inputs:
            if inp['type'] == "Tensor":
                inp_type = "tvm.relay.Expr"
            else:
                raise RuntimeError("unknown/unimplemented type")
            args.append(inp['name'] + ": " + inp_type)
            arg_names.append(inp['name'])
        for attr in attrs:
            if attr['type'] == "int":
                arg_annot = "int"
                if attr['default'] is not None:
                    arg_annot += f" = {attr['default']}"
            elif attr['type'] == "double":
                arg_annot = "float"
                if attr['default'] is not None:
                    arg_annot += f" = {attr['default']}"
            elif attr['type'] == "String":
                arg_annot = "str"
                if attr['default'] is not None:
                    arg_annot += f" = \"{attr['default']}\""
            else:
                raise RuntimeError("unknown/unimplemented type")
            
            args.append(attr['name'] + ": " + arg_annot)
            arg_names.append(attr['name'])

        tmp1 = ",\n    ".join(arg for arg in args)
        tmp2 = ",\n    ".join(arg for arg in arg_names)
        py_hook = ops_python_template.format(
            py_name = self.target_name + "_" + self.OpName, 
            py_args_with_annotations = tmp1, 
            all_args = tmp2
        )
        return py_hook

    def _add_operator(self, config) -> None:  #TODO:type of config? dict?
        """Registration of accelerator-specific operators in Relay level.

        Parameters
        ----------
        config : dict
           the operator specificatios which user has determined 
        """
        if config is not None:
            self.OpName = config['name']
            self.accOpName = self.target_name + self.OpName
            OpDescription = config['description']
            attributes = config['attributes']
            inputs = config['inputs']
            output = config['output']
            cpp_code = cpp_template.format(
                acc_name = self.target_name, 
                op_name = self.OpName,
                attr_node = self._make_attr_node(attributes), 
                type_relation = self._make_type_rel(inputs, output), 
                relay_call_node = self._make_relay_call_node(inputs, attributes),
                register_op = self._make_register_op(inputs, OpDescription, 11)
            )

            # add the operator cc file
            filename = f"{self.OpName}.cc"
            current_path = os.getcwd()
            cpp_path = os.path.abspath(os.path.join(current_path, f"../../src/relay/op/contrib/{self.target_name}"))
            if not os.path.exists(cpp_path):
                os.makedirs(cpp_path)
                print(f"Directory {cpp_path} created.")
            else:
                print(f"Directory {cpp_path} already exists.")
            
            op_Cpath = os.path.abspath(os.path.join(current_path, f"../../src/relay/op/contrib/{self.target_name}/{filename}"))
            with open(op_Cpath, 'w') as f:
                f.write(cpp_code)

            #TODO: make it in strategy file
            op_Pypath = os.path.abspath(os.path.join(self.hw_path, "strategies.py"))
            with open(op_Pypath, 'a') as f:
                f.write('\n')
                f.write(self._make_py_api_hook(inputs, attributes))