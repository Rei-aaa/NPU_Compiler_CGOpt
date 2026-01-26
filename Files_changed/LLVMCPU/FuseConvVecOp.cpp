#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/FuseConvVecOp.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "FuseConvVecFunc.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::iree_compiler {

namespace {

struct FuseConvVecPattern : public RewritePattern {
  explicit FuseConvVecPattern(MLIRContext *ctx)
      : RewritePattern("linalg.conv_2d_nchw_fchw", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // 目标卷积算子 linalg.conv_2d_nchw_fchw 操作

    Operation *convOp = op;

    // 向下收集可融合的操作列表（支持的可融合逐元素操作）并匹配可融合模式
    Operation *fusedGenericOp = nullptr; // 当前只记录generic块
    ElementwiseChain elementwiseOps; // 记录generic内部逐元素操作链
    FusionPatternInfo pattern;
    if (!collectFusableOps(convOp, fusedGenericOp, elementwiseOps, pattern))
      return failure();

    // 根据匹配的情况rewrite操作
    Operation *newOp = rewriteWithFusedOp(convOp, fusedGenericOp, elementwiseOps, pattern, rewriter);
    if (!newOp)
      return failure();

    // 重定向外部用户到新融合操作
    redirectFusedChain(convOp, fusedGenericOp, newOp);

    // 删除多余的已融合操作
    eraseFusedOps(convOp, fusedGenericOp, rewriter);

    return success();
  }
};

// Pass 实现：
struct FuseConvVecOpPass
    : public PassWrapper<FuseConvVecOpPass, OperationPass<func::FuncOp>> {

  StringRef getArgument() const final { return "iree-llvmcpu-fuse-conv-vec"; }
  StringRef getDescription() const final {
    return "Fuses tiled linalg.conv_2d_nchw_fchw ops with supported elementwise chains.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<FuseConvVecPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createFuseConvVecOpPass() {
  return std::make_unique<FuseConvVecOpPass>();
}

} // namespace mlir::iree_compiler