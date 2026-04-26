#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/FuseConvVecOp.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "FuseConvVecFunc.h"
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"
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
    ElementwiseChain mainDataChain;
    // Track the elementwise op chain inside generic.
    FusionPatternInfo pattern;
    if (!collectFusableOps(convOp, fusedGenericOp, elementwiseOps,
                           mainDataChain))
      return failure();
    auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                    : linalg::GenericOp();
    if (!genericOp)
      return failure();

    // 常量折叠(开启fastmath时的折叠)
    // Apply constant folding, more aggressively when fastmath is enabled.
    if (enableFastMath && !printedBeforeFolding) {
      llvm::dbgs() << "[FuseConvVecOp] Generic block before constant folding:\n";
      genericOp.dump();
      llvm::dbgs() << "[FuseConvVecOp] Main data chain from conv result:\n";
      if (mainDataChain.empty()) {
        llvm::dbgs() << "  (empty)\n";
      } else {
        for (auto it : llvm::enumerate(mainDataChain)) {
          llvm::dbgs() << "  [" << it.index() << "] ";
          it.value()->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
      }
      printedBeforeFolding = true;
    }

    // Restrict rewrite stages to the main stream chain only.
    ElementwiseChain mainChainOps = mainDataChain;

    foldConstantElementwiseOps(genericOp, mainChainOps, rewriter, pattern,
                   enableFastMath);

    // Re-collect after folding so converted/fused stages see updated stream chain.
    elementwiseOps.clear();
    mainDataChain.clear();
    if (!collectFusableOps(convOp, fusedGenericOp, elementwiseOps,
                           mainDataChain))
      return failure();
    genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                               : linalg::GenericOp();
    if (!genericOp)
      return failure();
    mainChainOps = mainDataChain;

    if (enableFastMath && !printedAfterFolding) {
      llvm::dbgs() << "[FuseConvVecOp] Generic block after constant folding:\n";
      genericOp.dump();
      printedAfterFolding = true;
    }

    // 可融合的操作转化为中间状态 npufuseop PE/activation op
    // Convert fusible ops into intermediate NPU PE/activation ops.
    convertToNpuOp(convOp, genericOp, mainChainOps, rewriter);

    // Re-collect after conversion because old chain pointers may have been
    // erased/replaced by npufuseop ops.
    elementwiseOps.clear();
    mainDataChain.clear();
    if (!collectFusableOps(convOp, fusedGenericOp, elementwiseOps,
                           mainDataChain))
      return failure();
    genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                               : linalg::GenericOp();
    if (!genericOp)
      return failure();
    mainChainOps = mainDataChain;

    if (!printedAfterConvert) {
      llvm::dbgs() << "[FuseConvVecOp] Generic block after convertToNpuOp:\n";
      genericOp.dump();
      printedAfterConvert = true;
    }
    
    // rewrite -> fusedConv
    Operation *newOp = rewriteWithFusedOp(convOp, fusedGenericOp, mainChainOps,
                                          pattern, rewriter, enableFastMath);
    if (!newOp)
      return failure();

    if (!printedFusedOpCreated) {
      llvm::dbgs() << "[FuseConvVecOp] Successfully created fused op: "
                   << newOp->getName() << "\n";
      llvm::dbgs() << "[FuseConvVecOp] New fused op IR:\n";
      newOp->print(llvm::dbgs());
      llvm::dbgs() << "\n";
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

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    Operation *firstResidualPE = nullptr;
    func.walk([&](Operation *op) {
      if (firstResidualPE)
        return;
                  if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::PE1AOp,
                    mlir::iree::compiler::Dialect::NPUFuseOp::PE1BOp,
                    mlir::iree::compiler::Dialect::NPUFuseOp::PE2AOp,
                    mlir::iree::compiler::Dialect::NPUFuseOp::PE2BOp,
                    mlir::iree::compiler::Dialect::NPUFuseOp::PE3AOp,
                    mlir::iree::compiler::Dialect::NPUFuseOp::PE3BOp>(op))
        firstResidualPE = op;
    });

    if (firstResidualPE) {
      func.emitError()
          << "FuseConvVecOpPass found residual npufuseop PE intermediate "
             "ops after rewrite";
      firstResidualPE->emitRemark() << "first residual PE op";
      signalPassFailure();
      return;
    }
  }
  bool enableFastMath = false;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createFuseConvVecOpPass(bool enableFastMath) {
  return std::make_unique<FuseConvVecOpPass>(enableFastMath);
}

} // namespace mlir::iree_compiler