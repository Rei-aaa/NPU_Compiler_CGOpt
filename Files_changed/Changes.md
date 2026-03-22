# NPU_Compiler中改动的部分

()括号中是改动后文件在Files_changed文件夹中的位置

## 1 方言和操作

### iree\compiler\src\iree\compiler\Dialect\NPUFuseOp

iree\compiler\src\iree\compiler\Dialect\ 新建NPUFuseOp文件夹，文件夹下包括：

（Files_changed/NPUFuseOps）

```
//文件夹下加入以下文件
BufferizableOpInterfaceImpl.cpp
CMakeLists.txt
NPUFuseDialect.cpp
NPUFuseDialect.h
NPUFuseOps.cpp
NPUFuseOps.h
NPUFuseOps.td
```

声明方言，操作，接口，以及提供注册函数以便在pipeline中注册方言

（这里有修改，名称也改为npuop）





改动了iree\compiler\src\iree\compiler\API\Internal\CompilerDriver.cpp

(Files_changed/CompilerDriver)

头文件处新加：NPUDialect.h（修改了名字）

```cpp
#include "iree/compiler/Dialect/NPUOp/NPUDialect.h"
```

406行左右，添加到MLIR上下文，Session: 

```cpp
Session::Session(GlobalInit &globalInit)
    : globalInit(globalInit), ownedContext(globalInit.createContext()),
      context(*ownedContext), binder(OptionsBinder::local()),
      pluginSession(globalInit.pluginManager, binder, pluginManagerOptions) {
  context.allowUnregisteredDialects();
  context.appendDialectRegistry(globalInit.registry);
          
  // 新加：        
   context.getOrLoadDialect<mlir::iree::compiler::Dialect::NPUOp::NPUDialect>();
	......
}
```





改动了iree\compiler\src\iree\compiler\API\CMakeLists.txt 新加一行 ：

(Files_changed/CMakeLists)

```cpp
list(APPEND _EXPORT_OBJECT_LIBS "iree_compiler_Dialect_NPUOp_IR.objects")
```





改动了iree\compiler\src\iree\compiler\Tools\init_dialects.h

(Files_changed/init_dialects.h)

在registerAllDialects中注册自定义方言

```cpp
// 新加头文件：
#include "iree/compiler/Dialect/NPUOp/NPUDialect.h"

namespace mlir::iree_compiler {

inline void registerAllDialects(DialectRegistry &registry) {
  registerMlirDialects(registry);
  registerIreeDialects(registry);

// 新加这一句：
mlir::iree::compiler::Dialect::NPUOp::registerNPUOpDialects(registry);

  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
}

}
```



## 2新加pass

在iree\compiler\src\iree\compiler\Codegen\LLVMCPU下新建了（后面已改动）

```
//LLVMCPU文件夹下加入以下文件
FuseConvVecOp.cpp
FuseConvVecOp.h
FuseConvVecFunc.cpp
FuseConvVecFunc.h
```

(Files_changed/LLVMCPU/...), 下同

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\BUILD.bazel

```cpp
    srcs = [
    	...
        "FuseConvVecFunc.cpp",
        "FuseConvVecOp.cpp",
        ...
        
    hdrs = [
    	...
        "FuseConvVecFunc.h",
        "FuseConvVecOp.h",
        ...
```





改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\CMakeLists.txt

```cpp
iree_cc_library(
  NAME
    LLVMCPU
  HDRS
    ...
    "FuseConvVecFunc.h"
    "FuseConvVecOp.h"
	...
  SRCS
    ...
    "FuseConvVecFunc.cpp"
    "FuseConvVecOp.cpp"
    ...
```





改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.td

在passes.td中声明pass

```cpp
def FuseConvVecOp :
    Pass<"fuse-conv-vec-op", "func::FuncOp"> {
  let summary = "Fuse linalg.conv_2d_nchw_fchw + element-wise op chains";
  let description = [{
    Detects a linalg.conv_2d_nchw_fchw followed by generics and
    replaces the chain with a npuop.conv_2d op that encodes the
    combined semantics for downstream lowering.
  }];
  let constructor = "mlir::iree_compiler::createFuseConvVecOpPass()";
}
```





改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.h

```cpp
// 新加头文件：
#include "iree/compiler/Codegen/LLVMCPU/FuseConvVecOp.h"
```





改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.cpp

```cpp
// 头文件
#include "iree/compiler/Codegen/LLVMCPU/FuseConvVecOp.h"

// 385 行：
        nestedModulePM.addNestedPass<func::FuncOp>(createFuseConvVecOpPass());
```

具体位置，新加pass目前加在LLVMCPUTileAndFuse pass之后：

```cpp

void addMultiTilingExpertPassPipeline(OpPassManager &passManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt) {
    ......
  // Apply tile and fuse to all the non-distribution fusable levels. Skip
  // distribution level as that level has been fused already.
  if (allFusableLevels.size() > 1) {
    llvm::SmallSetVector<int64_t, 4> fusableLevels(allFusableLevels.begin(),
                                                   allFusableLevels.end());
    for (int i = 0; i < tilingConfig.getNumTilingLevels(); ++i) {
      if (i == tilingConfig.getDistributionLevel())
        continue;
      if (fusableLevels.contains(i)) {
        nestedModulePM.addNestedPass<func::FuncOp>(
            createLLVMCPUTileAndFusePass(i));
          // 新加pass在这里：
        nestedModulePM.addNestedPass<func::FuncOp>(createFuseConvVecOpPass());
        nestedModulePM.addNestedPass<func::FuncOp>(
            createFuseTensorPadWithConsumerPass());
        nestedModulePM.addNestedPass<func::FuncOp>(
            createConcretizePadResultShapePass());
        continue;
      }

```





## 3 FuseOp -> func.call  ->llvm.call

一些改动点比较多的地方，主要是把原本的默认普通conv改成了支持更多操作数，以及需要继承属性

或许可以直接替换文件？



改动了iree\compiler\src\iree\compiler\Codegen\Common\GenericVectorization.cpp

(Files_changed/GenericVectorization)

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\ConvertToLLVM.cpp

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\ConvertNPUOps.cpp

(Files_changed/LLVMCPU/ConvertNPUOps.cpp)

## 4 CDP（LN、softmax相关）

iree\compiler\src\iree\compiler\Dialect\Flow\Transforms\FormDispatchRegions.cpp

```cpp
// 新加一个函数，例如240行左右
//===----------------------------------------------------------------------===//
// BEGIN NPU_DISPATCH_ROOT_BLOCK (local, minimal extension)
//===----------------------------------------------------------------------===//
static bool isNpuDispatchRootCandidate(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  if (opName != "npuop.layer_norm" && opName != "npuop.softmax") {
    return false;
  }

  if (op->getNumResults() == 0) {
    return false;
  }

  if (!llvm::all_of(op->getOperandTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); })) {
    return false;
  }
  if (!llvm::all_of(op->getResultTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); })) {
    return false;
  }

  return true;
}
//===----------------------------------------------------------------------===//
// END NPU_DISPATCH_ROOT_BLOCK
//===----------------------------------------------------------------------===//
```

```cpp
// 832行左右：
      if (!(isa<linalg::LinalgOp, tensor::PadOp, tensor::PackOp,
                IREE::LinalgExt::SetEncodingOp>(op) ||
            // 新加一句或：
            isNpuDispatchRootCandidate(&op)) ||
          
          isa<linalg::FillOp>(op) || isDequantizationLikeOp(&op)) {
        continue;
      }
```

新加iree\compiler\src\iree\compiler\Codegen\LLVMCPU\ConvertNPUUnaryOpsToCalls.cpp

需要新加此文件

iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.cpp

加入两个自定义pass，以及两个flags

```cpp
#include "iree/compiler/Codegen/LLVMCPU/FuseConvVecOp.h"
// 新加一个编译选项：是否允许在融合pass中启用fast-math
static llvm::cl::opt<bool> clEnableNpuFuseFastMath(
  "iree-llvmcpu-npufuse-fast-math",
  llvm::cl::desc("Enable fast-math for npufuseop fused conv chains"),
  llvm::cl::init(false));
static llvm::cl::opt<bool> clEnableNpuFuseAssumeConstParams(
  "iree-llvmcpu-npufuse-assume-const-params",
  llvm::cl::desc("Assume npufuseop conv params are constant for folding"),
  llvm::cl::init(false));
// ~400 行：
  nestedModulePM.addNestedPass<func::FuncOp>(
      createFuseConvVecOpPass(clEnableNpuFuseFastMath,
                              clEnableNpuFuseAssumeConstParams));
```

```cpp
// 660行左右：
static void addLowerToLLVMPasses(OpPassManager &passManager,
                                 bool enableAArch64SME) {
  // TODO: Remove the following pass and plumb support for #hal.descriptor_type
  // memory space through the stack.
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());

  // Lower `ukernel.*` ops to function calls
  passManager.addPass(createLowerUKernelOpsToCallsPass());

  // 新加：
  // Convert standalone NPU unary ops to func.call across this module,
  // including nested dispatch modules.
  passManager.addPass(createConvertNPUUnaryOpsToCallsPass());
```

iree\third_party\torch-mlir\lib\Dialect\Torch\Transforms\DecomposeComplexOps：

```cpp
// 头文件新加：
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

//新加：
static llvm::cl::opt<bool> clEnableNpuLnSoftmax(
    "iree-llvmcpu-enable-npu-ln-softmax",
    llvm::cl::desc("Rewrite aten.layer_norm/aten.softmax to npuop ops in "
                   "torch-decompose-complex-ops"),
    llvm::cl::init(false));

static Operation *createNpuOpWithSegmentedOperands(
    PatternRewriter &rewriter, Location loc, StringRef opName, Type resultType,
    ArrayRef<Value> operands, ArrayRef<NamedAttribute> attrs,
    ArrayRef<int32_t> segmentSizes) {
  OperationState state(loc, opName);
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttributes(attrs);
  if (!segmentSizes.empty()) {
    state.addAttribute("operand_segment_sizes",
                       rewriter.getDenseI32ArrayAttr(segmentSizes));
  }
  return rewriter.create(state);
}
static Value convertTorchValueTensorToBuiltin(PatternRewriter &rewriter,
                                              Location loc, Value value) {
  if (!value.getType().isa<Torch::ValueTensorType>())
    return Value();
  return rewriter.create<TorchConversion::ToBuiltinTensorOp>(loc, value);
}

static Value convertBuiltinTensorToTorchValue(PatternRewriter &rewriter,
                                              Location loc,
                                              Type torchValueTensorType,
                                              Value builtinTensor) {
  auto resultType = torchValueTensorType.dyn_cast<Torch::ValueTensorType>();
  if (!resultType)
    return Value();
  return rewriter.create<TorchConversion::FromBuiltinTensorOp>(
      loc, resultType, builtinTensor);
}
```

```cpp
（1279行左右）

// Decompose softmax into: exp(x) / sum(exp(x))
namespace {
class DecomposeAtenSoftmaxIntOp : public OpRewritePattern<AtenSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
// 新加：
    if (clEnableNpuLnSoftmax) {
      int64_t dimInt;
      if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
        return rewriter.notifyMatchFailure(op,
                                           "expected dim to be constant int");

      Value selfBuiltin =
          convertTorchValueTensorToBuiltin(rewriter, op.getLoc(), self);
      if (!selfBuiltin)
        return rewriter.notifyMatchFailure(
            op, "expected self to be !torch.vtensor for NPU softmax");

      auto resultType = op.getType().dyn_cast<Torch::ValueTensorType>();
      if (!resultType)
        return rewriter.notifyMatchFailure(
            op, "expected result type to be !torch.vtensor");
      Type builtinResultType = resultType.toBuiltinTensor();
      if (!builtinResultType)
        return rewriter.notifyMatchFailure(
            op, "failed to convert softmax result type to builtin tensor");

      SmallVector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("dim", rewriter.getI64IntegerAttr(dimInt)));

      if (!op.getDtype().getType().isa<Torch::NoneType>()) {
        int64_t dtypeInt;
        if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt))) {
          return rewriter.notifyMatchFailure(
              op, "expected dtype to be None or constant int");
        }
        attrs.push_back(rewriter.getNamedAttr("dtype",
                                              rewriter.getI64IntegerAttr(
                                                  dtypeInt)));
      }

        SmallVector<Value> operands = {selfBuiltin};
      Operation *npuSoftmaxOp = createNpuOpWithSegmentedOperands(
          rewriter, op.getLoc(), "npuop.softmax", builtinResultType, operands,
          attrs, {});

        Value torchResult = convertBuiltinTensorToTorchValue(
          rewriter, op.getLoc(), op.getType(), npuSoftmaxOp->getResult(0));
        if (!torchResult)
        return rewriter.notifyMatchFailure(
          op, "failed to convert NPU softmax result back to !torch.vtensor");

        rewriter.replaceOp(op, torchResult);
      return success();
    }
```



```cpp
（1390行左右）
namespace {
class DecomposeAten_SoftmaxOp : public OpRewritePattern<Aten_SoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");
    bool halfToFloat;
    if (!matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");
// 新加：
    if (clEnableNpuLnSoftmax && !halfToFloat) {
      int64_t dimInt;
      if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
        return rewriter.notifyMatchFailure(op,
                                           "expected dim to be constant int");

      SmallVector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("dim", rewriter.getI64IntegerAttr(dimInt)));

      SmallVector<Value> operands = {self};
      Operation *npuSoftmaxOp = createNpuOpWithSegmentedOperands(
          rewriter, op.getLoc(), "npuop.softmax", op.getType(), operands,
          attrs, {});
      rewriter.replaceOp(op, npuSoftmaxOp->getResults());
      return success();
    }

```

```cpp
（4148行左右）

class DecomposeAtenLayerNormOp : public OpRewritePattern<AtenLayerNormOp> {
  using OpRewritePattern<AtenLayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLayerNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
// 新加：
    if (clEnableNpuLnSoftmax) {
      SmallVector<Value> normalizedShapeElements;
      if (!getListConstructElements(op.getNormalizedShape(),
                                    normalizedShapeElements)) {
        return rewriter.notifyMatchFailure(
            op, "expected normalized_shape to be list construct");
      }

      SmallVector<int64_t> normalizedShapeInts;
      normalizedShapeInts.reserve(normalizedShapeElements.size());
      for (Value dimValue : normalizedShapeElements) {
        int64_t dimInt;
        if (!matchPattern(dimValue, m_TorchConstantInt(&dimInt))) {
          return rewriter.notifyMatchFailure(
              op, "expected normalized_shape elements to be constant int");
        }
        normalizedShapeInts.push_back(dimInt);
      }

      double epsValue;
      if (!matchPattern(op.getEps(), m_TorchConstantFloat(&epsValue))) {
        return rewriter.notifyMatchFailure(op,
                                           "expected eps to be constant float");
      }

      SmallVector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr(
          "normalized_shape",
          rewriter.getDenseI64ArrayAttr(normalizedShapeInts)));
      attrs.push_back(
          rewriter.getNamedAttr("eps", rewriter.getF64FloatAttr(epsValue)));

      if (!op.getCudnnEnable().getType().isa<Torch::NoneType>()) {
        bool cudnnEnable;
        if (!matchPattern(op.getCudnnEnable(),
                          m_TorchConstantBool(&cudnnEnable))) {
          return rewriter.notifyMatchFailure(
              op, "expected cudnn_enable to be None or constant bool");
        }
        attrs.push_back(rewriter.getNamedAttr(
            "cudnn_enable", rewriter.getBoolAttr(cudnnEnable)));
      }

      SmallVector<Value> operands = {op.getInput()};
      SmallVector<int32_t> operandSegmentSizes = {1, 0, 0};
      if (!op.getWeight().getType().isa<Torch::NoneType>()) {
        operands.push_back(op.getWeight());
        operandSegmentSizes[1] = 1;
      }
      if (!op.getBias().getType().isa<Torch::NoneType>()) {
        operands.push_back(op.getBias());
        operandSegmentSizes[2] = 1;
      }

      Operation *npuLayerNormOp = createNpuOpWithSegmentedOperands(
          rewriter, loc, "npuop.layer_norm", op.getType(), operands, attrs,
          operandSegmentSizes);
      rewriter.replaceOp(op, npuLayerNormOp->getResults());
      return success();
    }
```

