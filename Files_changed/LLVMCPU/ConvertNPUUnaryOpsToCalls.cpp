#include "iree/compiler/Codegen/LLVMCPU/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {
namespace {

static FailureOr<func::FuncOp>
getOrCreateCallee(PatternRewriter &rewriter, ModuleOp module, Location loc,
                  StringRef calleeName, FunctionType calleeType) {
  // Reuse an existing declaration only if the signature matches exactly.
  // This pass intentionally keeps one canonical symbol per callee name.
  if (auto callee = module.lookupSymbol<func::FuncOp>(calleeName)) {
    if (callee.getFunctionType() != calleeType) {
      return failure();
    }
    return callee;
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto newCallee = rewriter.create<func::FuncOp>(loc, calleeName, calleeType);
  auto linkageAttr =
      LLVM::LinkageAttr::get(rewriter.getContext(), LLVM::Linkage::External);
  newCallee->setAttr("llvm.linkage", linkageAttr);
  // Keep the symbol local to this module while still modeling an external
  // function declaration for lowering/import.
  newCallee.setPrivate();
  return newCallee;
}

static void copyAttrsToCall(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getAttrs()) {
    // Segment size metadata belongs to variadic custom ops, not func.call.
    if (attr.getName().getValue() == "operandSegmentSizes")
      continue;
    to->setAttr(attr.getName(), attr.getValue());
  }
}

static bool isSupportedCallBoundaryType(Type type) {
  return isa<BaseMemRefType, RankedTensorType>(type);
}

class ConvertNPUUnaryToCallPattern : public RewritePattern {
public:
  ConvertNPUUnaryToCallPattern(MLIRContext *context, StringRef rootOpName,
                               StringRef calleeName)
      : RewritePattern(rootOpName, /*benefit=*/1, context),
        calleeName(calleeName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Rewrite both tensor and memref forms so this pass can run before DPS.
    if (!llvm::all_of(op->getOperandTypes(), isSupportedCallBoundaryType) ||
      !llvm::all_of(op->getResultTypes(), isSupportedCallBoundaryType) ||
      op->getNumResults() != 1 || op->getNumOperands() == 0) {
      return failure();
    }

    // Require destination-style shape/type compatibility between the trailing
    // output init operand and the op result.
    Value outputInit = op->getOperand(op->getNumOperands() - 1);
    if (outputInit.getType() != op->getResult(0).getType()) {
      return failure();
    }

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Type> callOperandTypes;
    SmallVector<Type> callResultTypes;
    SmallVector<Value> callOperands;
    callOperands.assign(op->operand_begin(), op->operand_end());
    callOperandTypes.assign(op->operand_type_begin(), op->operand_type_end());
    callResultTypes.assign(op->result_type_begin(), op->result_type_end());

    // Build/find an external declaration and replace the custom op with
    // a result-returning func.call of identical signature.
    auto calleeType = rewriter.getFunctionType(callOperandTypes, callResultTypes);
    FailureOr<func::FuncOp> maybeCallee =
        getOrCreateCallee(rewriter, module, op->getLoc(), calleeName,
                          calleeType);
    if (failed(maybeCallee)) {
      return rewriter.notifyMatchFailure(
          op,
          "failed to create/find callee with compatible function type");
    }

    auto call = rewriter.create<func::CallOp>(op->getLoc(), maybeCallee->getName(),
                                              callResultTypes, callOperands);
    copyAttrsToCall(op, call.getOperation());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  StringRef calleeName;
};

struct ConvertNPUUnaryOpsToCallsPass
    : public PassWrapper<ConvertNPUUnaryOpsToCallsPass,
             OperationPass<ModuleOp>> {
  StringRef getArgument() const override {
    return "iree-convert-npu-unary-ops-to-calls";
  }

  StringRef getDescription() const override {
      return "Converts npufuseop.softmax/layer_norm to result-returning "
        "func.call";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Lower standalone unary NPU ops to runtime-call form.
    patterns.add<ConvertNPUUnaryToCallPattern>(
      &getContext(), "npufuseop.layer_norm", "npu_layer_norm");
    patterns.add<ConvertNPUUnaryToCallPattern>(
      &getContext(), "npufuseop.softmax", "npu_softmax");

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createConvertNPUUnaryOpsToCallsPass() {
  return std::make_unique<ConvertNPUUnaryOpsToCallsPass>();
}

} // namespace mlir::iree_compiler
