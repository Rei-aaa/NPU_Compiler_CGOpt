#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECFUNC_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECFUNC_H_

#include <string>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler {

// 融合子算子信息：记录子算子和mlir中构成它的操作名称列表
struct FusedSubOpInfo {
  std::string role;
  llvm::SmallVector<std::string, 6> opNames;
};

// 额外操作数信息：记录额外操作数的值和角色名称
struct ExtraOperandInfo {
  Value value;
  std::string role;
};

// 融合模式信息：记录匹配到的融合算子名、子算子信息、额外操作数
struct FusionPatternInfo {
  std::string fusedOpName;
  llvm::SmallVector<FusedSubOpInfo, 6> fusedSubOps;
  llvm::SmallVector<ExtraOperandInfo, 6> extraOperands;
};

// 逐元素操作链信息，记录generic中按顺序出现的逐元素op
using ElementwiseChain = llvm::SmallVector<Operation *, 12>;

// 向下收集可能参与融合的算子并匹配受支持的融合模式，返回命中的模式
bool collectFusableOps(Operation *convOp,
  Operation *&fusedGenericOp,
  ElementwiseChain &elementwiseOps,
  FusionPatternInfo &pattern);

// 根据匹配信息创建新的融合算子（默认 npufuseop.conv_add_relu）
Operation *rewriteWithFusedOp(Operation *convOp,
        Operation *fusedGenericOp,
        const ElementwiseChain &elementwiseOps,
        const FusionPatternInfo &pattern,
                              PatternRewriter &rewriter);

// 将链尾结果的用户重定向到新融合操作
void redirectFusedChain(Operation *convOp,
  Operation *fusedGenericOp,
  Operation *newOp);

// 删除链条中的 generic，并最终移除原始 conv
void eraseFusedOps(Operation *convOp,
     Operation *fusedGenericOp,
     PatternRewriter &rewriter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECFUNC_H_