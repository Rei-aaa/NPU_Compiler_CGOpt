#include "iree/compiler/Dialect/NPUOp/NPUOps.h"
#include "iree/compiler/Dialect/NPUOp/NPUDialect.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace iree {
namespace compiler {
namespace Dialect {
namespace NPUOp {

// Implement the simple methods declared in the header.
llvm::StringRef NPUDialect::getDialectNamespace() {
  return "npuop";
}

NPUDialect::NPUDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<NPUDialect>()) {
  addOperations<
      #define GET_OP_LIST
      #include "iree/compiler/Dialect/NPUOp/NPUOps.cpp.inc"
      //::mlir::iree::compiler::Dialect::NPUOp::ConvAddReluOp
  >();
}

void registerNPUDialect(mlir::MLIRContext *context) {
  context->getOrLoadDialect<mlir::iree::compiler::Dialect::NPUOp::NPUDialect>();
}

} // namespace NPUOp
} // namespace Dialect
} // namespace compiler
} // namespace iree
} // namespace mlir
