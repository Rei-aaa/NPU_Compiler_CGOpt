// Header exposing the NPUFuse dialect and a helper to register it into a
// DialectRegistry. This avoids needing special link-time wiring; callers can
// just include this header and call `registerNPUFuseOpDialects(registry)` to
// ensure the dialect is available.

#ifndef IREE_COMPILER_DIALECT_NPU_NPUFUSEDIALECT_H_
#define IREE_COMPILER_DIALECT_NPU_NPUFUSEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace iree {
namespace compiler {
namespace Dialect {
namespace NPUFuseOp {

class NPUFuseDialect : public mlir::Dialect {
public:
  static llvm::StringRef getDialectNamespace();
  explicit NPUFuseDialect(MLIRContext *context);
};

// Registers BufferizableOpInterface external models for NPUFuseOp dialect.
void registerNPUFuseOpBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

inline void registerNPUFuseOpDialects(mlir::DialectRegistry &registry) {
  registry.insert<NPUFuseDialect>();
  registerNPUFuseOpBufferizableOpInterfaceExternalModels(registry);
}

// Convenience helper to register the dialect into a context.
void registerNPUFuseDialect(mlir::MLIRContext *context);

} // namespace NPUFuseOp
} // namespace Dialect
} // namespace compiler
} // namespace iree
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_NPU_NPUFUSEDIALECT_H_
