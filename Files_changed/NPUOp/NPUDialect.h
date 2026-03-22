// Header exposing the NPU dialect and a helper to register it into a
// DialectRegistry. This avoids needing special link-time wiring; callers can
// just include this header and call `registerNPUOpDialects(registry)` to
// ensure the dialect is available.

#ifndef IREE_COMPILER_DIALECT_NPU_NPUDIALECT_H_
#define IREE_COMPILER_DIALECT_NPU_NPUDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace iree {
namespace compiler {
namespace Dialect {
namespace NPUOp {

class NPUDialect : public mlir::Dialect {
public:
  static llvm::StringRef getDialectNamespace();
  explicit NPUDialect(MLIRContext *context);
};

// Registers BufferizableOpInterface external models for NPUOp dialect.
void registerNPUOpBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

inline void registerNPUOpDialects(mlir::DialectRegistry &registry) {
  registry.insert<NPUDialect>();
  registerNPUOpBufferizableOpInterfaceExternalModels(registry);
}

// Convenience helper to register the dialect into a context.
void registerNPUDialect(mlir::MLIRContext *context);

} // namespace NPUOp
} // namespace Dialect
} // namespace compiler
} // namespace iree
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_NPU_NPUDIALECT_H_
