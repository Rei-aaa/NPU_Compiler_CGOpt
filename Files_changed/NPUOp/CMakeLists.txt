################################################################################
# CMake for npuop dialect
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    IR
  HDRS
    "NPUDialect.h"
    "NPUOps.h"
  TEXTUAL_HDRS
    "NPUOps.h.inc"
    "NPUOps.cpp.inc"
    "NPUOpsInterfaces.h.inc"
    "NPUOpsInterfaces.cpp.inc"
  SRCS
    "NPUDialect.cpp"
    "BufferizableOpInterfaceImpl.cpp"
    "NPUOps.cpp"
  DEPS
    MLIRIR
    MLIRSupport
    iree::compiler::Utils
  PUBLIC
)

iree_tablegen_library(
  NAME
    NPUOpsGen
  TD_FILE
    "NPUOps.td"
  OUTS
    --gen-op-decls NPUOps.h.inc
    --gen-op-defs NPUOps.cpp.inc
    --gen-op-interface-decls NPUOpsInterfaces.h.inc
    --gen-op-interface-defs NPUOpsInterfaces.cpp.inc
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
