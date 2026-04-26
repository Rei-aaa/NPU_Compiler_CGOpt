//------------- BufferizableOpInterfaceImpl.cpp -------------------------
// Bufferization hooks for the NPUFuseOp dialect.
//-----------------------------------------------------------------------

#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseDialect.h"
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::iree::compiler::Dialect::NPUFuseOp;

// Defined with linalg naming conventions, so index maps are derived by convention.
// Computed maps are attached to linalg.memoized_indexing_maps.
// This auto-generated field represents the current op indexing maps.
// If indexing changes later, corresponding passes will rewrite it.

namespace {

// NPU fused ops have conv-like bufferization behavior.
// - Inputs are read-only, with additional input operands.
// - Output behavior is consistent with conv.
template <typename OpTy>
struct ConvLikeExternalModel
  : public BufferizableOpInterface::ExternalModel<ConvLikeExternalModel<OpTy>,
                          OpTy> {
  // Check whether an operand is read from memory.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &) const {
    //最后一个是 outs，其余都是只读输入
    // The last operand is outs; all others are read-only inputs.
    return opOperand.getOperandNumber() != getOutsOperandIndex(op);
  }

  // Check whether an operand writes to memory.
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &) const {
    return opOperand.getOperandNumber() == getOutsOperandIndex(op);
  }

  // 获取操作数别名信息，如果操作数是 outs，则操作结果与其是等价的别名关系
  // For outs, result 0 is an equivalent alias of that operand.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    Operation *owner = opOperand.getOwner();
    if (opOperand.getOperandNumber() == getOutsOperandIndex(owner))
      return {{owner->getResult(0), BufferRelation::Equivalent}};
    return {};
  }

  BufferRelation bufferRelation(OpResult, const AnalysisState &) const {
    // 结果与 outs 缓冲区等价
    // The result is equivalent to the outs buffer.
    return BufferRelation::Equivalent;
  }

  bool bufferizesToElementwiseAccess(Operation *, const AnalysisState &,
                                     ArrayRef<OpOperand *>) const {
    // Return false because this is not an elementwise op.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // 认为最后一个是 outs，其余都是只读输入
    // Keep last-operand-as-outs to tolerate future operand growth.
    unsigned outsIdx = getOutsOperandIndex(op);
    if (outsIdx >= op->getNumOperands())
      return failure();

    // Get a buffer value for every operand.
    SmallVector<Value> buffers;
    buffers.reserve(op->getNumOperands());
    for (OpOperand &operand : op->getOpOperands()) {
      FailureOr<Value> buf = getBuffer(rewriter, operand.get(), options);
      if (failed(buf))
        return failure();
      buffers.push_back(*buf);
    }

    // Rebuild the same op with memref IO and replace tensor results.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    OperationState state(op->getLoc(), op->getName().getIdentifier());
    state.addTypes(buffers[outsIdx].getType());
    state.addOperands(buffers);
    state.addAttributes(op->getAttrs());
    Operation *newOp = Operation::create(state);
    rewriter.insert(newOp);
    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
    return success();
  }

private:
  static unsigned getOutsOperandIndex(Operation *op) {
    // Current fused ops have one outs operand at the end.
    return op->getNumOperands() - 1;
  }
};

// 非 DPS 的一元 NPU 操作（如 layer_norm/softmax）：
// Non-DPS unary NPU ops (e.g., layer_norm/softmax):
// - 输入只读
// - Input is read-only.
// - 结果不与任何输入别名
// - Result does not alias any input.
// - bufferize 时将 tensor 输入/结果统一替换为 memref 版本的同名算子
// - Bufferization rewrites tensor IO to memref on the same op.
template <typename OpTy>
struct UnaryLikeExternalModel
    : public BufferizableOpInterface::ExternalModel<
          UnaryLikeExternalModel<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &opOperand,
                              const AnalysisState &) const {
    return isa<BaseMemRefType, TensorType>(opOperand.get().getType());
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  bool bufferizesToElementwiseAccess(Operation *, const AnalysisState &,
                                     ArrayRef<OpOperand *>) const {
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    if (op->getNumResults() != 1 ||
        !isa<TensorType>(op->getResult(0).getType())) {
      return failure();
    }

    SmallVector<Value> bufferOperands;
    bufferOperands.reserve(op->getNumOperands());
    for (OpOperand &operand : op->getOpOperands()) {
      if (!isa<TensorType>(operand.get().getType())) {
        bufferOperands.push_back(operand.get());
        continue;
      }
      FailureOr<Value> maybeBuffer = getBuffer(rewriter, operand.get(), options);
      if (failed(maybeBuffer))
        return failure();
      bufferOperands.push_back(*maybeBuffer);
    }

    BaseMemRefType resultType = getMemRefType(op->getResult(0), options);

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    OperationState state(op->getLoc(), op->getName().getIdentifier());
    state.addTypes(resultType);
    state.addOperands(bufferOperands);
    state.addAttributes(op->getAttrs());
    Operation *newOp = Operation::create(state);
    rewriter.insert(newOp);
    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
    return success();
  }
};

} // namespace

void mlir::iree::compiler::Dialect::NPUFuseOp::
    registerNPUFuseOpBufferizableOpInterfaceExternalModels(
        DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, NPUFuseDialect *dialect) {
    (void)dialect; // Unused.
    Conv2DOp::attachInterface<UnaryLikeExternalModel<Conv2DOp>>(*ctx);
    LayerNormOp::attachInterface<ConvLikeExternalModel<LayerNormOp>>(*ctx);
    SoftmaxOp::attachInterface<ConvLikeExternalModel<SoftmaxOp>>(*ctx);
  });
}
