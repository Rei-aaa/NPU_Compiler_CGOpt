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

// 按照linalg命名算子模式定义，会按照约定计算出索引映射
// 会把计算结果挂在 linalg.memoized_indexing_maps
// 他是自动生成的，表示当前算子的索引映射
// 后续如果有变化，对应的pass会重写更新它

namespace {

// NPU 融合操作具有和 原conv 类似的 bufferization 行为：
// - 输入均为只读，但有额外的输入数
// - 输出和conv一致
template <typename OpTy>
struct ConvLikeExternalModel
  : public BufferizableOpInterface::ExternalModel<ConvLikeExternalModel<OpTy>,
                          OpTy> {
  // 判断操作数是否需要内存读取                          
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &) const {
    //最后一个是 outs，其余都是只读输入。
    return opOperand.getOperandNumber() != getOutsOperandIndex(op);
  }

  // 判断操作数是否需要内存写入
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &) const {
    return opOperand.getOperandNumber() == getOutsOperandIndex(op);
  }

  // 获取操作数别名信息，如果操作数是 outs，则操作结果与其是等价的别名关系
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    Operation *owner = opOperand.getOwner();
    if (opOperand.getOperandNumber() == getOutsOperandIndex(owner))
      return {{owner->getResult(0), BufferRelation::Equivalent}};
    return {};
  }

  BufferRelation bufferRelation(OpResult, const AnalysisState &) const {
    // 结果与 outs 缓冲区等价
    return BufferRelation::Equivalent;
  }

  bool bufferizesToElementwiseAccess(Operation *, const AnalysisState &,
                                     ArrayRef<OpOperand *>) const {
    // 返回 false 表示这是非逐元素操作
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // 支持未来增加操作数的场景：认为最后一个是 outs，其余都是只读输入。
    unsigned outsIdx = getOutsOperandIndex(op);
    if (outsIdx >= op->getNumOperands())
      return failure();

    // 为每个操作数获取对应的 buffer 值
    SmallVector<Value> buffers;
    buffers.reserve(op->getNumOperands());
    for (OpOperand &operand : op->getOpOperands()) {
      FailureOr<Value> buf = getBuffer(rewriter, operand.get(), options);
      if (failed(buf))
        return failure();
      buffers.push_back(*buf);
    }

    // 用 memref 操作数/结果重建同名算子，属性保持一致，再替换原 tensor 结果。
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
    // 所有当前融合算子都只有一个 outs，且位于操作数末尾。
    return op->getNumOperands() - 1;
  }
};

} // namespace

void mlir::iree::compiler::Dialect::NPUFuseOp::
    registerNPUFuseOpBufferizableOpInterfaceExternalModels(
        DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, NPUFuseDialect *dialect) {
    (void)dialect; // Unused.
    ConvAddReluOp::attachInterface<ConvLikeExternalModel<ConvAddReluOp>>(*ctx);
    ConvAddBnReluOp::attachInterface<ConvLikeExternalModel<ConvAddBnReluOp>>(*ctx);
  });
}
