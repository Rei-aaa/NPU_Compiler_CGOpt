// This file includes the TableGen-generated op implementations.
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include <utility>

// clang-format off
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOpsInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.cpp.inc" // IWYU pragma: keep
// clang-format on
