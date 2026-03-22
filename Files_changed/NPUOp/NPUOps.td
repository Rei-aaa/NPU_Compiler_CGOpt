include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

// ranked tensor 或 memref
// Ranked tensor or memref.
def AnyRankedTensorOrMemRefType : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;
// ranked tensor, memref, or scalar float
def AnyRankedTensorOrMemRefOrFloatType : AnyTypeOf<[AnyRankedTensor, AnyMemRef, AnyFloat]>;

def NPUOp_Dialect : Dialect {
  let name = "npuop";
  let cppNamespace = "::mlir::iree::compiler::Dialect::NPUOp";
}

class NPUOp_Op<string mnemonic, list<Trait> traits = []> :
    Op<NPUOp_Dialect, mnemonic, traits>;


// Fused conv_2d with optional PE tensors
def Conv2DOp : NPUOp_Op<"conv_2d", [AttrSizedOperandSegments]> {
  let summary = "Convolution with optional PE tensors";
  let description = [{
    Convolution op with fixed input and filter operands, plus optional
    PE tensors for additional processing stages.
  }];

  let arguments = (ins
    AnyRankedTensorOrMemRefType:$input,
    AnyRankedTensorOrMemRefType:$filter,
    Optional<AnyRankedTensorOrMemRefType>:$pe1_a,
    Optional<AnyRankedTensorOrMemRefType>:$pe1_b,
    Optional<AnyRankedTensorOrMemRefType>:$pe2_a,
    Optional<AnyRankedTensorOrMemRefType>:$pe2_b,
    Optional<AnyRankedTensorOrMemRefType>:$pe3_a,
    Optional<AnyRankedTensorOrMemRefType>:$pe3_b
  );

  let results = (outs AnyRankedTensorOrMemRefType:$output);

  let assemblyFormat = [{
    $input `,` $filter
    (`,` $pe1_a^)?
    (`,` $pe1_b^)?
    (`,` $pe2_a^)?
    (`,` $pe2_b^)?
    (`,` $pe3_a^)?
    (`,` $pe3_b^)?
    attr-dict `:`
    type($input) `,` type($filter)
    (`,` type($pe1_a)^)?
    (`,` type($pe1_b)^)?
    (`,` type($pe2_a)^)?
    (`,` type($pe2_b)^)?
    (`,` type($pe3_a)^)?
    (`,` type($pe3_b)^)?
    `->` type($output)
  }];

}

// Activation ops used as intermediate fusion forms
def ReluOp : NPUOp_Op<"relu"> {
  let summary = "ReLU activation";
  let arguments = (ins AnyRankedTensorOrMemRefOrFloatType:$input);
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def LeakyReluOp : NPUOp_Op<"leakyrelu"> {
  let summary = "LeakyReLU activation";
  let arguments = (ins AnyRankedTensorOrMemRefOrFloatType:$input);
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def SiluOp : NPUOp_Op<"silu"> {
  let summary = "SiLU activation";
  let arguments = (ins AnyRankedTensorOrMemRefOrFloatType:$input);
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def SigmoidOp : NPUOp_Op<"sigmoid"> {
  let summary = "Sigmoid activation";
  let arguments = (ins AnyRankedTensorOrMemRefOrFloatType:$input);
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def GeluOp : NPUOp_Op<"gelu"> {
  let summary = "GELU activation";
  let arguments = (ins AnyRankedTensorOrMemRefOrFloatType:$input);
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def TanhOp : NPUOp_Op<"tanh"> {
  let summary = "Tanh activation";
  let arguments = (ins AnyRankedTensorOrMemRefOrFloatType:$input);
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def LayerNormOp : NPUOp_Op<"layer_norm", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>]> {
  let summary = "Layer normalization";
  let description = [{
    Layer normalization op with input tensor, optional weight/bias tensors,
    and attributes for normalized shape, epsilon, and optional cudnn-enable.
  }];

  let arguments = (ins
    AnyType:$input,
    Optional<AnyType>:$weight,
    Optional<AnyType>:$bias,
    AnyType:$output_init,
    DenseI64ArrayAttr:$normalized_shape,
    F64Attr:$eps,
    OptionalAttr<BoolAttr>:$cudnn_enable
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $input
    (`,` $weight^)?
    (`,` $bias^)?
    `,` $output_init
    attr-dict `:`
    type($input)
    (`,` type($weight)^)?
    (`,` type($bias)^)?
    `,` type($output_init)
    `->` type($output)
  }];
}

def SoftmaxOp : NPUOp_Op<"softmax", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>]> {
  let summary = "Softmax";
  let description = [{
    Softmax op with input tensor and attributes for reduction dimension and
    optional dtype.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output_init,
    I64Attr:$dim,
    OptionalAttr<I64Attr>:$dtype
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $input `,` $output_init attr-dict `:`
    type($input) `,` type($output_init) `->` type($output)
  }];
}

// PE ops used as intermediate fusion forms
def PE1AOp : NPUOp_Op<"PE1_a"> {
  let summary = "PE1_a multiply-like op";
  let arguments = (ins
    AnyRankedTensorOrMemRefOrFloatType:$lhs,
    AnyRankedTensorOrMemRefOrFloatType:$rhs
  );
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def PE1BOp : NPUOp_Op<"PE1_b"> {
  let summary = "PE1_b add-like op";
  let arguments = (ins
    AnyRankedTensorOrMemRefOrFloatType:$lhs,
    AnyRankedTensorOrMemRefOrFloatType:$rhs
  );
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def PE2AOp : NPUOp_Op<"PE2_a"> {
  let summary = "PE2_a multiply-like op";
  let arguments = (ins
    AnyRankedTensorOrMemRefOrFloatType:$lhs,
    AnyRankedTensorOrMemRefOrFloatType:$rhs
  );
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def PE2BOp : NPUOp_Op<"PE2_b"> {
  let summary = "PE2_b add-like op";
  let arguments = (ins
    AnyRankedTensorOrMemRefOrFloatType:$lhs,
    AnyRankedTensorOrMemRefOrFloatType:$rhs
  );
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def PE3AOp : NPUOp_Op<"PE3_a"> {
  let summary = "PE3_a multiply-like op";
  let arguments = (ins
    AnyRankedTensorOrMemRefOrFloatType:$lhs,
    AnyRankedTensorOrMemRefOrFloatType:$rhs
  );
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def PE3BOp : NPUOp_Op<"PE3_b"> {
  let summary = "PE3_b add-like op";
  let arguments = (ins
    AnyRankedTensorOrMemRefOrFloatType:$lhs,
    AnyRankedTensorOrMemRefOrFloatType:$rhs
  );
  let results = (outs AnyRankedTensorOrMemRefOrFloatType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

