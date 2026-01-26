//===- FuseConvVecOp.h ----------------------------------------*- C++ -*-===//
//
// Declarations for the FuseConvVec pass.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECOP_H
#define IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECOP_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::iree_compiler {

/// 创建用于融合卷积后续 Add/Relu 链的 Pass，
/// 将 "linalg.conv_2d_nchw_fchw" 重写为 "npufuseop.conv_add_relu"。
std::unique_ptr<OperationPass<func::FuncOp>> createFuseConvVecOpPass();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECOP_H
