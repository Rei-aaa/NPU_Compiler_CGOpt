// This file includes the TableGen-generated op implementations.
#include "iree/compiler/Dialect/NPUOp/NPUOps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include <utility>

namespace mlir::iree::compiler::Dialect::NPUOp {

MutableOperandRange LayerNormOp::getDpsInitsMutable() {
  return MutableOperandRange(getOutputInitMutable());
}

MutableOperandRange SoftmaxOp::getDpsInitsMutable() {
  return MutableOperandRange(getOutputInitMutable());
}

} // namespace mlir::iree::compiler::Dialect::NPUOp

// clang-format off
#include "iree/compiler/Dialect/NPUOp/NPUOpsInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/NPUOp/NPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on
