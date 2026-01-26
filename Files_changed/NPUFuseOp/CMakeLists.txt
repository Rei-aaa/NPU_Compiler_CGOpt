################################################################################
# CMake for npufuseop dialect
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    IR
  HDRS
    "NPUFuseDialect.h"
    "NPUFuseOps.h"
  TEXTUAL_HDRS
    "NPUFuseOps.h.inc"
    "NPUFuseOps.cpp.inc"
    "NPUFuseOpsInterfaces.h.inc"
    "NPUFuseOpsInterfaces.cpp.inc"
  SRCS
    "NPUFuseDialect.cpp"
    "BufferizableOpInterfaceImpl.cpp"
    "NPUFuseOps.cpp"
  DEPS
    MLIRIR
    MLIRSupport
    iree::compiler::Utils
  PUBLIC
)

iree_tablegen_library(
  NAME
    NPUFuseOpsGen
  TD_FILE
    "NPUFuseOps.td"
  OUTS
    --gen-op-decls NPUFuseOps.h.inc
    --gen-op-defs NPUFuseOps.cpp.inc
    --gen-op-interface-decls NPUFuseOpsInterfaces.h.inc
    --gen-op-interface-defs NPUFuseOpsInterfaces.cpp.inc
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
