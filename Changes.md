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





改动了iree\compiler\src\iree\compiler\API\Internal\CompilerDriver.cpp

(Files_changed/CompilerDriver)

头文件处新加：NPUFuseDialect.h

```cpp
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseDialect.h"
```

406行左右，添加到上下文，Session: 

```cpp
Session::Session(GlobalInit &globalInit)
    : globalInit(globalInit), ownedContext(globalInit.createContext()),
      context(*ownedContext), binder(OptionsBinder::local()),
      pluginSession(globalInit.pluginManager, binder, pluginManagerOptions) {
  context.allowUnregisteredDialects();
  context.appendDialectRegistry(globalInit.registry);
          
  // 新加：        
  context.getOrLoadDialect<mlir::iree::compiler::Dialect::NPUFuseOp::NPUFuseDialect>();
	......
}
```





改动了iree\compiler\src\iree\compiler\API\CMakeLists.txt 新加一行 ：

(Files_changed/CMakeLists)

```cpp
list(APPEND _EXPORT_OBJECT_LIBS "iree_compiler_Dialect_NPUFuseOp_IR.objects")
```





改动了iree\compiler\src\iree\compiler\Tools\init_dialects.h

(Files_changed/init_dialects.h)

在registerAllDialects中注册自定义方言

```cpp
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseDialect.h"

inline void registerAllDialects(DialectRegistry &registry) {
  registerMlirDialects(registry);
  registerIreeDialects(registry);
  mlir::iree::compiler::Dialect::NPUFuseOp::registerNPUFuseOpDialects(
      registry);

  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
}
```



## 2新加pass

在iree\compiler\src\iree\compiler\Codegen\LLVMCPU下新建了

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
  let summary = "Fuse linalg.conv_2d_nchw_fchw + add + relu chains";
  let description = [{
    Detects a linalg.conv_2d_nchw_fchw followed by add/relu generics and
    replaces the chain with a npufuseop.conv_add_relu op that encodes the
    combined semantics for downstream lowering.
  }];
  let constructor = "mlir::iree_compiler::createFuseConvVecOpPass()";
}

```





改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.h

```
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

改动了iree\compiler\src\iree\compiler\Codegen\Common\GenericVectorization.cpp

(Files_changed/GenericVectorization)

改动较多，新加了两块：

一个是FuseHelper，统一处理操作数Tensor -> memref, 结果memref -> Tensor, 以及为新建的call op添加属性

一个是具体的rewrite，对支持的op重写（或者后续改成NPUFuseOp方言的op都统一重写）

两块新加的代码都标记了：

```cpp

///////////////////////////////////////////////////////////////
//                       NPU Fuse Helper                     //
//           Build external call + operand role attrs         //
///////////////////////////////////////////////////////////////

template <typename NpuOpT>
static void rewriteNpuFuseOpToCall(IRRewriter &rewriter, MLIRContext *context,
                                   NpuOpT npuOp)

......

    ///////////////////////////////////////////////////////////////
    //                       NPU Fuse OP                         //
    //                Call Accelerator CONV2D API                //
    ///////////////////////////////////////////////////////////////
    // 支持的fuse pattern
    if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddReluOp>(op)) {
      if (auto convOp = dyn_cast<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddReluOp>(op))
        rewriteNpuFuseOpToCall(rewriter, context, convOp);
      continue;
    }

    if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddBnReluOp>(op)) {
      if (auto convOp = dyn_cast<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddBnReluOp>(op))
        rewriteNpuFuseOpToCall(rewriter, context, convOp);
      continue;
    }
  };
```





改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\ConvertNPUOps.cpp

(Files_changed/LLVMCPU/ConvertNPUOps.cpp)

改动比较多，主要是把原本的默认普通conv改成了支持更多操作数，以及需要继承属性

或许可以直接替换文件？
