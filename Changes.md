# NPU_Compiler中改动的部分

()括号中是改动后文件在Files_changed文件夹中的位置

## 1 方言和操作

### iree\compiler\src\iree\compiler\Dialect\NPUFuseOp

iree\compiler\src\iree\compiler\Dialect\ 新建NPUFuseOp文件夹，文件夹下包括：

（Files_changed/NPUFuseOps）

```
BufferizableOpInterfaceImpl.cpp
CMakeLists.txt
NPUFuseDialect.cpp
NPUFuseDialect.h
NPUFuseOps.cpp
NPUFuseOps.h
NPUFuseOps.td
```

改动了iree\compiler\src\iree\compiler\API\Internal\CompilerDriver.cpp

(Files_changed/CompilerDriver)

改动了iree\compiler\src\iree\compiler\API\CMakeLists.txt 新加一行 ：

(Files_changed/CMakeLists)

```cpp
list(APPEND _EXPORT_OBJECT_LIBS "iree_compiler_Dialect_NPUFuseOp_IR.objects")
```

改动了iree\compiler\src\iree\compiler\Tools\init_dialects.h

(Files_changed/init_dialects.h)



## 2新加pass

在iree\compiler\src\iree\compiler\Codegen\LLVMCPU下新建了

```
FuseConvVecOp.cpp
FuseConvVecOp.h
FuseConvVecFunc.cpp
FuseConvVecFunc.h
```

(Files_changed/LLVMCPU/...), 下同

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\BUILD.bazel

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\CMakeLists.txt

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.td

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.h

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\Passes.cpp

改动了iree\compiler\src\iree\compiler\Codegen\LLVMCPU\ConvertNPUOps.cpp



## 3 FuseOp -> func.call  ->llvm.call

改动了iree\compiler\src\iree\compiler\Codegen\Common\GenericVectorization.cpp

(Files_changed/GenericVectorization)


