#ifndef IREE_COMPILER_DIALECT_NPUFUSEOP_IR_NPUFUSEOPS_H_
#define IREE_COMPILER_DIALECT_NPUFUSEOP_IR_NPUFUSEOPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOpsInterfaces.h.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h.inc"

#endif  // IREE_COMPILER_DIALECT_NPUFUSEOP_IR_NPUFUSEOPS_H_
