#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/FuseConvVecOp.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "FuseConvVecFunc.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::iree_compiler {

namespace {

struct FuseConvVecPattern : public RewritePattern {
  explicit FuseConvVecPattern(MLIRContext *ctx, bool enableFastMath)
      : RewritePattern("linalg.conv_2d_nchw_fchw", /*benefit=*/1, ctx),
        enableFastMath(enableFastMath) {}

  bool enableFastMath;
  mutable bool printedBeforeFolding = false;
  mutable bool printedAfterFolding = false;
  mutable bool printedAfterConvert = false;
  mutable bool printedFusedOpCreated = false;
  
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // Target the linalg.conv_2d_nchw_fchw convolution op.

    Operation *convOp = op;

    // 向下收集可融合的操作列表（支持的可融合逐元素操作）
    // Collect downstream fusible ops, including supported elementwise ops.
    Operation *fusedGenericOp = nullptr;
    // Only track the generic op for now.
    ElementwiseChain elementwiseOps;
    // Track the elementwise op chain inside generic.
    FusionPatternInfo pattern;
    if (!collectFusableOps(convOp, fusedGenericOp, elementwiseOps))
      return failure();
    auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                    : linalg::GenericOp();
    if (!genericOp)
      return failure();

    // 常量折叠(开启fastmath时的折叠)
    // Apply constant folding, more aggressively when fastmath is enabled.
    if (enableFastMath && !printedBeforeFolding) {
      llvm::dbgs() << "[FuseConvVecOp] Before constant folding:\n";
      genericOp.dump();
      printedBeforeFolding = true;
    }

    foldConstantElementwiseOps(genericOp, elementwiseOps, rewriter, pattern,
                   enableFastMath);

    if (enableFastMath && !printedAfterFolding) {
      llvm::dbgs() << "[FuseConvVecOp] After constant folding:\n";
      genericOp.dump();
      printedAfterFolding = true;
    }

    // 可融合的操作转化为中间状态 npuop PE/activation op
    // Convert fusible ops into intermediate NPU PE/activation ops.
    convertToNpuOp(genericOp, elementwiseOps, rewriter);

    if (!printedAfterConvert) {
      llvm::dbgs() << "[FuseConvVecOp] After convertToNpuOp:\n";
      genericOp.dump();
      printedAfterConvert = true;
    }
    
    // rewrite -> fusedConv
    Operation *newOp = rewriteWithFusedOp(convOp, fusedGenericOp, elementwiseOps,
                                          pattern, rewriter, enableFastMath);
    if (!newOp)
      return failure();

    if (!printedFusedOpCreated) {
      llvm::dbgs() << "[FuseConvVecOp] Successfully created fused op: "
                   << newOp->getName() << "\n";
      printedFusedOpCreated = true;
    }

    // 重定向外部用户到新融合操作
    // Redirect external users to the new fused op
    redirectFusedChain(convOp, fusedGenericOp, newOp, pattern, elementwiseOps);

    // 删除多余的已融合操作
    // Remove redundant operations that were already fused
    eraseFusedOps(convOp, fusedGenericOp, rewriter, pattern, elementwiseOps);

    return success();
  }
};

// Pass implementation
struct FuseConvVecOpPass
  : public PassWrapper<FuseConvVecOpPass, OperationPass<func::FuncOp>> {
  FuseConvVecOpPass() = default;
  explicit FuseConvVecOpPass(bool enableFastMath)
    : enableFastMath(enableFastMath) {}

  StringRef getArgument() const final { return "iree-llvmcpu-fuse-conv-vec"; }
  StringRef getDescription() const final {
    return "Fuses tiled linalg.conv_2d_nchw_fchw ops with supported elementwise chains.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<FuseConvVecPattern>(ctx, enableFastMath);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
  bool enableFastMath = false;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createFuseConvVecOpPass(bool enableFastMath) {
  return std::make_unique<FuseConvVecOpPass>(enableFastMath);
}

} // namespace mlir::iree_compiler