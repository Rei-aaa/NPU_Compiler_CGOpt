#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECFUNC_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECFUNC_H_

#include <string>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

// 激活信息：记录激活类型与它发生在逐元素链中的位置
struct ActivationInfo {
  std::string type;
  int afterOpIndex = -1;
};

// 融合模式信息：记录匹配到的融合算子名、子算子信息、额外操作数
struct FusionPatternInfo {
  std::string fusedOpName;
  llvm::SmallVector<FusedSubOpInfo, 6> fusedSubOps;
  llvm::SmallVector<ExtraOperandInfo, 6> extraOperands;
  llvm::SmallVector<ActivationInfo, 6> activations;
  // 记录匹配到的逐元素操作数量，以及该部分最后一个结果
  int matchedOpCount = 0;
  Value matchedValue;
};

// 逐元素操作链信息，记录generic中按顺序出现的逐元素op
using ElementwiseChain = llvm::SmallVector<Operation *, 12>;

// 向下收集可能参与融合的算子
bool collectFusableOps(Operation *convOp,
  Operation *&fusedGenericOp,
  ElementwiseChain &elementwiseOps);

// Convert elementwise chains to npuop PE/activation ops in-order.
void convertToNpuOp(linalg::GenericOp genericOp,
  ElementwiseChain &elementwiseOps,
  PatternRewriter &rewriter);

// Fold constant-only arithmetic ops inside the matched elementwise chain.
void foldConstantElementwiseOps(linalg::GenericOp genericOp,
  ElementwiseChain &elementwiseOps,
  PatternRewriter &rewriter,
  FusionPatternInfo &pattern,
  bool enableFastMath);

// 根据匹配信息创建新的融合算子（默认 npuop.conv_add）
Operation *rewriteWithFusedOp(Operation *convOp,
  Operation *fusedGenericOp,
  const ElementwiseChain &elementwiseOps,
  const FusionPatternInfo &pattern,
            PatternRewriter &rewriter,
            bool enableFastMath);

// 将链尾结果的用户重定向到新融合操作
void redirectFusedChain(Operation *convOp,
  Operation *fusedGenericOp,
  Operation *newOp,
  const FusionPatternInfo &pattern,
  const ElementwiseChain &elementwiseOps);

// 删除链条中的 generic，并最终移除原始 conv
void eraseFusedOps(Operation *convOp,
     Operation *fusedGenericOp,
     PatternRewriter &rewriter,
     const FusionPatternInfo &pattern,
     const ElementwiseChain &elementwiseOps);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_FUSECONVVECFUNC_H_