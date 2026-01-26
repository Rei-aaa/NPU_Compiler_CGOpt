#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseDialect.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace iree {
namespace compiler {
namespace Dialect {
namespace NPUFuseOp {

// Implement the simple methods declared in the header.
llvm::StringRef NPUFuseDialect::getDialectNamespace() {
  return "npufuseop";
}

NPUFuseDialect::NPUFuseDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<NPUFuseDialect>()) {
  addOperations<
      #define GET_OP_LIST
      #include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.cpp.inc"
      //::mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddReluOp
  >();
}

void registerNPUFuseDialect(mlir::MLIRContext *context) {
  context->getOrLoadDialect<mlir::iree::compiler::Dialect::NPUFuseOp::NPUFuseDialect>();
}

} // namespace NPUFuseOp
} // namespace Dialect
} // namespace compiler
} // namespace iree
} // namespace mlir
