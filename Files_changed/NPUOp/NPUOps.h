#ifndef IREE_COMPILER_DIALECT_NPU_IR_NPUOPS_H_
#define IREE_COMPILER_DIALECT_NPU_IR_NPUOPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "iree/compiler/Dialect/NPUOp/NPUOpsInterfaces.h.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/NPUOp/NPUOps.h.inc"

#endif  // IREE_COMPILER_DIALECT_NPU_IR_NPUOPS_H_
