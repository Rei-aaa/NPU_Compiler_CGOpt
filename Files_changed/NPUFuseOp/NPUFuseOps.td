include "mlir/IR/OpBase.td"

// NPUFuseOp 接口
def NPUFuseOpCallableInterface : OpInterface<"NPUFuseOpCallableInterface"> {
  let description = [{
  Exposes NPU fused op calling conventions to lowering passes: provides
  external callee name and operand role indices to build a unified
  tensor -> memref -> func.call -> tensor lowering path.
  }];

  let methods = [
    InterfaceMethod<
      "Return the external NPU callee symbol name; each op must define it.",
      "::llvm::StringRef", "getNpuCalleeName", (ins)>,

    InterfaceMethod<
      "Return the operand index of the main input tensor, or -1 if absent.",
      "int", "getInputOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of the convolution filter, or -1 if absent.",
      "int", "getFilterOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of bias/addend, or -1 when this role is not present.",
      "int", "getBiasOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of batchnorm mean, or -1 when absent.",
      "int", "getBnMeanOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of batchnorm variance, or -1 when absent.",
      "int", "getBnVarianceOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of batchnorm scale (gamma), or -1 when absent.",
      "int", "getBnScaleOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of batchnorm offset (beta), or -1 when absent.",
      "int", "getBnOffsetOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the operand index of the outs/init tensor, or -1 if absent.",
      "int", "getOutsOperandIndex", (ins)>,

    InterfaceMethod<
      "Return the result index carrying the primary output, or -1 if absent.",
      "int", "getOutputResultIndex", (ins)>
  ];
}

// ranked tensor 或 memref
def AnyRankedTensorOrMemRefType : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;

def NPUFuseOp_Dialect : Dialect {
  let name = "npufuseop";
  let cppNamespace = "::mlir::iree::compiler::Dialect::NPUFuseOp";
}

class NPUFuseOp_Op<string mnemonic, list<Trait> traits = []> :
    Op<NPUFuseOp_Dialect, mnemonic, traits>;

// Fused conv + add + relu op 
def ConvAddReluOp : NPUFuseOp_Op<"conv_add_relu", [NPUFuseOpCallableInterface]> {
  let summary = "Fused convolution + add + relu";
  let description = [{
    Fuse a convolution, bias add, and relu into a single op. The op accepts
    ranked tensor operands for the input feature map, filter, bias (or folded
    element-wise add), and the destination tensor used as an "outs" operand.
  }];

  let arguments = (ins
    AnyRankedTensorOrMemRefType:$input,
    AnyRankedTensorOrMemRefType:$filter,
    AnyRankedTensorOrMemRefType:$bias,
    AnyRankedTensorOrMemRefType:$output_init
  );

  let results = (outs AnyRankedTensorOrMemRefType:$output);

  // getOutputResultIndex是按照输出的顺序
  let extraClassDeclaration = [{
    ::llvm::StringRef getNpuCalleeName() { return "npu_conv_add_relu"; }
    int getInputOperandIndex() { return 0; }
    int getFilterOperandIndex() { return 1; }
    int getBiasOperandIndex() { return 2; }
    int getBnMeanOperandIndex() { return -1; }
    int getBnVarianceOperandIndex() { return -1; }
    int getBnScaleOperandIndex() { return -1; }
    int getBnOffsetOperandIndex() { return -1; }
    int getOutsOperandIndex() { return 3; }
    int getOutputResultIndex() { return 0; }
  }];

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $input `,` $filter `,` $bias `:`
    type($input) `,` type($filter) `,` type($bias) `)`
    `outs` `(` $output_init `:` type($output_init) `)`
    `->` type($output)
  }];
}
 
// Fused conv + add + batchnorm + relu op
def ConvAddBnReluOp : NPUFuseOp_Op<"conv_add_bn_relu", [NPUFuseOpCallableInterface]> {
  let summary = "Fused convolution + add + batchnorm + relu";
  let description = [{
    Fuse a convolution, element-wise add/bias, batch normalization, and relu into
    a single op. The op accepts ranked tensor operands for the input feature map,
    filter, add/bias, batchnorm parameters (mean, variance, scale, offset), and
    the destination tensor used as an "outs" operand.
  }];

  let arguments = (ins
    AnyRankedTensorOrMemRefType:$input,
    AnyRankedTensorOrMemRefType:$filter,
    AnyRankedTensorOrMemRefType:$bias,
    AnyRankedTensorOrMemRefType:$bn_mean,
    AnyRankedTensorOrMemRefType:$bn_variance,
    AnyRankedTensorOrMemRefType:$bn_scale,
    AnyRankedTensorOrMemRefType:$bn_offset,
    AnyRankedTensorOrMemRefType:$output_init
  );

  let results = (outs AnyRankedTensorOrMemRefType:$output);

  let extraClassDeclaration = [{
    ::llvm::StringRef getNpuCalleeName() { return "npu_conv_add_bn_relu"; }
    int getInputOperandIndex() { return 0; }
    int getFilterOperandIndex() { return 1; }
    int getBiasOperandIndex() { return 2; }
    int getBnMeanOperandIndex() { return 3; }
    int getBnVarianceOperandIndex() { return 4; }
    int getBnScaleOperandIndex() { return 5; }
    int getBnOffsetOperandIndex() { return 6; }
    int getOutsOperandIndex() { return 7; }
    int getOutputResultIndex() { return 0; }
  }];

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $input `,` $filter `,` $bias `,` $bn_mean `,` $bn_variance `,`
    $bn_scale `,` $bn_offset `:`
    type($input) `,` type($filter) `,` type($bias) `,` type($bn_mean) `,`
    type($bn_variance) `,` type($bn_scale) `,` type($bn_offset) `)`
    `outs` `(` $output_init `:` type($output_init) `)`
    `->` type($output)
  }];
}
