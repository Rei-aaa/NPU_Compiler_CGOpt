// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseDialect.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler {

namespace {

///////////////////////////////////////////////////////////////
//                       NPU Fuse Helper                     //
//           Build external call + operand role attrs         //
///////////////////////////////////////////////////////////////

template <typename NpuOpT>
static void rewriteNpuFuseOpToCall(IRRewriter &rewriter, MLIRContext *context,
                                   NpuOpT npuOp) {
  SmallVector<Type> callArgumentTypes;
  SmallVector<Type> callReturnTypes;
  SmallVector<Value> operandsRange;

  auto loc = npuOp.getLoc();
  rewriter.setInsertionPoint(npuOp);

  // 处理操作数，TensorType -> MemRefType
  auto operands = npuOp->getOperands();
  for (int i = 0; i < static_cast<int>(operands.size()) - 1; i++) {
    auto operand = operands[i];
    auto tensorType = llvm::cast<TensorType>(operand.getType());
    auto memrefType = MemRefType::get(tensorType.getShape(),
                                      tensorType.getElementType());
    auto newOperand =
        rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, operand);
    operandsRange.push_back(newOperand);
    callArgumentTypes.push_back(memrefType);
  }

  // 处理结果，TensorType -> MemRefType
  for (Type returnType : npuOp->getResultTypes()) {
    auto tensorType = llvm::cast<TensorType>(returnType);
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    callReturnTypes.push_back(memrefType);
  }

  // 函数类型function op type
  auto functionType =
      rewriter.getFunctionType(callArgumentTypes, callReturnTypes);

  // 调用 npu 外部函数（统一到 npu_conv_1d），并在属性中标记每个操作数的角色与位置
  StringRef fnName = "npu_conv_1d";
  auto makeRoleAttr = [&](StringRef role, unsigned idx) {
    return rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("role", rewriter.getStringAttr(role)),
        rewriter.getNamedAttr("index", rewriter.getI32IntegerAttr(idx)),
    });
  };
  auto addRoleIfValid = [&](SmallVectorImpl<Attribute> &attrs, StringRef role,
                            int idx) {
    if (idx >= 0 && static_cast<size_t>(idx) < operands.size())
      attrs.push_back(makeRoleAttr(role, static_cast<unsigned>(idx)));
  };

  SmallVector<Attribute> operandRoleAttrs;
  addRoleIfValid(operandRoleAttrs, "input", npuOp.getInputOperandIndex());
  addRoleIfValid(operandRoleAttrs, "filter", npuOp.getFilterOperandIndex());
  addRoleIfValid(operandRoleAttrs, "bias", npuOp.getBiasOperandIndex());
  addRoleIfValid(operandRoleAttrs, "bn_mean", npuOp.getBnMeanOperandIndex());
  addRoleIfValid(operandRoleAttrs, "bn_variance",
                 npuOp.getBnVarianceOperandIndex());
  addRoleIfValid(operandRoleAttrs, "bn_scale", npuOp.getBnScaleOperandIndex());
  addRoleIfValid(operandRoleAttrs, "bn_offset",
                 npuOp.getBnOffsetOperandIndex());
  addRoleIfValid(operandRoleAttrs, "outs", npuOp.getOutsOperandIndex());

  SmallVector<NamedAttribute> callAttrs;
  if (!operandRoleAttrs.empty())
    callAttrs.push_back(rewriter.getNamedAttr(
        "npu.operands", rewriter.getArrayAttr(operandRoleAttrs)));

  // 定位或创建外部被调用函数
  auto moduleOp = SymbolTable::getNearestSymbolTable(npuOp);
  auto fnDecl = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, fnName));
  if (!fnDecl) {
    // 插入点设置到module开头
    rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());
    // 外部链接
    auto linkageAttr =
        LLVM::LinkageAttr::get(rewriter.getContext(), LLVM::Linkage::External);
    SmallVector<NamedAttribute> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr("llvm.linkage", linkageAttr));
    // 创建函数
    fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType, funcAttrs);
    SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
  }

  // 恢复插入点，调用创建函数
  rewriter.setInsertionPoint(npuOp);
  auto symbolRef = mlir::SymbolRefAttr::get(context, fnName);
  auto newOp = rewriter.create<func::CallOp>(loc, callReturnTypes, symbolRef,
                                             operandsRange);
  if (!callAttrs.empty())
    newOp->setAttrs(callAttrs);
  // memref类型转换回Tensor类型
  Value outputOperand = newOp.getResult(0);
  auto newOutput =
      rewriter.create<bufferization::ToTensorOp>(loc, outputOperand, true);
  rewriter.replaceOp(npuOp, newOutput);
}
////////////////////////////////////////////////////////////////////////////

/// Tries to infer the vector sizes from an IR using ValueBounds analysis.
/// Returns failure if vector sizes can't be inferred.
static FailureOr<SmallVector<int64_t>>
inferVectorSizesFromIR(linalg::LinalgOp linalgOp) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring vector sizes for:\n" << linalgOp << "\n");

  SmallVector<int64_t> vectorSizes;
  unsigned numDims = linalgOp.getNumLoops();

  for (int dim = 0; dim < numDims; ++dim) {
    // Map dimension `dim` to an operand dimension that we will use to
    // traverse the U-D chain to get `dim` vector size information.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return failure();
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    if (!ShapedType::isDynamic(dimSize)) {
      vectorSizes.push_back(dimSize);
      LLVM_DEBUG(VEC_DBGS() << "Inferred vector size '" << dimSize
                            << "' for dimension '" << dim << "'\n");
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<int64_t> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand, operandDim,
          /*stopCondition=*/nullptr, /*closedUB=*/true);

      if (succeeded(maybeDimBound)) {
        break;
      }
    }

    if (failed(maybeDimBound)) {
      return failure();
    }

    dimSize = maybeDimBound.value();
    vectorSizes.push_back(dimSize);
    LLVM_DEBUG(VEC_DBGS() << "Inferred vector size '" << dimSize
                          << "' for dimension '" << dim << "'\n");
  }

  return vectorSizes;
}

// Return the vector sizes from the local lowering config or try to infer them
// from the tensor shapes and tiled loops in the IR.
static FailureOr<SizesAndScalableFlags>
getVectorSizes(linalg::LinalgOp linalgOp, bool useConfiguredVectorSizes) {
  // Get vector sizes from the lowering config, if available in the op itself.
  IREE::Codegen::LoweringConfigAttr loweringConfig =
      getLoweringConfig(linalgOp);
  if (useConfiguredVectorSizes && loweringConfig) {
    TilingConfig tilingConfig(loweringConfig);
    auto [vectorSizes, scalableFlags] = tilingConfig.getVectorTileSizes();
    // Replace zeros in canonical vector shape to turn it into a valid shape.
    std::replace(vectorSizes.begin(), vectorSizes.end(), 0, 1);
    return std::make_pair(vectorSizes, scalableFlags);
  }

  // Try to infer the vector sizes from the IR.
  auto vectorSizes = inferVectorSizesFromIR(linalgOp);
  if (succeeded(vectorSizes)) {
    // This can't identify scalable flags, so pad them with `false`.
    return std::make_pair(*vectorSizes,
                          SmallVector<bool>(vectorSizes->size(), false));
  }
  return failure();
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp linalgOp,
                                             int64_t maxVectorSize) {
  int64_t maxFlatVecSize = 1;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto type = llvm::dyn_cast<ShapedType>(operand.get().getType());
    if (!type)
      continue;
    if (!type.hasStaticShape())
      return failure();
    maxFlatVecSize = std::max(maxFlatVecSize, type.getNumElements());
  }
  return success(maxFlatVecSize < maxVectorSize);
}



class GenericVectorizationPass
    : public GenericVectorizationBase<GenericVectorizationPass> {
public:
  using GenericVectorizationBase::GenericVectorizationBase;
  GenericVectorizationPass(const GenericVectorizationPassOptions &options) {
    this->enableVectorMasking.setValue(options.enableVectorMasking);
    this->useConfiguredVectorSizes.setValue(options.useConfiguredVectorSizes);
    this->vectorizePadding.setValue(options.vectorizePadding);
    this->vectorizeGatherAccesses.setValue(options.vectorizeGatherAccesses);
    this->enableCleanup.setValue(options.enableCleanup);
    this->generateContract.setValue(options.generateContract);
    this->foldCastIntoContract.setValue(options.foldCastIntoContract);
    this->maxVectorSize.setValue(options.maxVectorSize);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect,
                    mlir::iree::compiler::Dialect::NPUFuseOp::NPUFuseDialect,
                    bufferization::BufferizationDialect,
                    memref::MemRefDialect,
                    func::FuncDialect,
                    LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  // Get linalg op and pad op
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op))
    // NPUFuseOps
      candidates.push_back(op);
    if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddReluOp>(op))
      candidates.push_back(op);
    if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddBnReluOp>(op))
      candidates.push_back(op);
    if (vectorizePadding && enableVectorMasking && isa<tensor::PadOp>(op))
      candidates.push_back(op);
  });

  // Deal the candidate op
  for (Operation *op : candidates) {
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableVecDims;

    if ((isa<linalg::Mmt4DOp>(op))){
      llvm::dbgs() << "\nThe op is related to mmt4d\n";

      auto mmt4DOp = dyn_cast<linalg::Mmt4DOp>(op);
      auto input1Type = mmt4DOp.getInputs()[0].getType();
      llvm::dbgs() << "Input 1 Type: " << input1Type << "\n";
      auto input2Type = mmt4DOp.getInputs()[1].getType();
      llvm::dbgs() << "Input 2 Type: " << input2Type << "\n";
      auto outputType = mmt4DOp.getOutputs()[0].getType();
      llvm::dbgs() << "Output Type: " << outputType << "\n";

      SmallVector<Type> callArgumentTypes;
      SmallVector<Type> callReturnTypes;
      SmallVector<Value> operandsRange;

      auto loc = op->getLoc();
      rewriter.setInsertionPoint(op);
      auto operands = op->getOperands();

      for (int i = 0; i < operands.size() - 1; i++) {
        auto operand = operands[i];
        TensorType tensorType = operand.getType().cast<TensorType>();
        MemRefType memrefType = MemRefType::get(  
            tensorType.getShape(),
            tensorType.getElementType()
        );
        llvm::dbgs() << "\nThe operand memrefType of this op is :\n";
        memrefType.dump();

        auto newOperand = rewriter.create<bufferization::ToMemrefOp>(
          loc, memrefType, operand);
        operandsRange.push_back(newOperand);
        callArgumentTypes.push_back(memrefType);
        llvm::dbgs() << "\nThe newOperand of this op is :\n";
        newOperand.dump();
      
      }

      llvm::dbgs() << "mmt4d operandsRange is :";
      for (auto operand : operandsRange) {
        operand.dump();
      }

      llvm::dbgs() << "mmt4d callArgumentTypes is :";
      for (auto type : callArgumentTypes) {
        type.dump();
      }

      for (auto returnType : op->getResultTypes()) {
        TensorType tensorType = returnType.cast<TensorType>();
        MemRefType memrefType = MemRefType::get(  
            tensorType.getShape(),
            tensorType.getElementType()
        );
        llvm::dbgs() << "\nThe return memrefType of this op is :\n";
        memrefType.dump();
        callReturnTypes.push_back(memrefType);
      }
      llvm::dbgs() << "mmt4d Output types: ";
      for (auto type : callReturnTypes) {
        type.dump();
      }

      // Obtain desired function op type
      auto functionType = rewriter.getFunctionType(callArgumentTypes, callReturnTypes);
      llvm::dbgs() << "mmt4d Function type: ";
      functionType.dump();

      // Create mmt4d op
      mlir::StringRef fnName = mlir::StringRef("npu_mmt4d");
      auto moduleOp = SymbolTable::getNearestSymbolTable(op);
      auto fnDecl = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(moduleOp, fnName));

      if (!fnDecl) {
        // Set correct insertion point for function op
        rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());

        // Create attributes
        auto linkageAttr = LLVM::LinkageAttr::get(rewriter.getContext(), LLVM::Linkage::External);
        SmallVector<NamedAttribute> funcAttrs;
        funcAttrs.push_back(rewriter.getNamedAttr("llvm.linkage", linkageAttr));

        fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType, funcAttrs);
        SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
      } else {
        llvm::dbgs() << "\nDid not create<func::FuncOp>!\n";
      }

      llvm::dbgs() << "\nHere finish creating the attribute of npu_mmt4d and set it to llvm.linkage to external api.\n";

      // Restore insertion point
      rewriter.setInsertionPoint(op);

      // Create call op
      auto symbolRef = mlir::SymbolRefAttr::get(context, fnName);
      auto newOp = rewriter.create<func::CallOp>(loc, symbolRef, callReturnTypes, operandsRange);
      llvm::dbgs() << "\nThe created newOp is :\n";
      newOp.dump();

      // Replace previous call op with new call op while keeping data type unchanged
      Value outputOperand = newOp.getResult(0);
      llvm::dbgs() << "\nOutput Operand before ToTensorOp:\n";
      llvm::dbgs() << "Output Operand: " << outputOperand << "\n";
      llvm::dbgs() << "Output Operand Type: " << outputOperand.getType() << "\n";

      auto newOutput = rewriter.create<bufferization::ToTensorOp>(loc, outputOperand, true);
      llvm::dbgs() << "\nNew Output after ToTensorOp:\n";
      newOutput.dump();
      llvm::dbgs() << "New Output Type: " << newOutput.getType() << "\n";

      // rewriter.replaceOp(op, newOutput);

      // Test part
      // Value newRead = rewriter.create<vector::TransferReadOp>(
      //   op->getLoc(), newReadType, op.getSource(), op.getIndices(),
      //   AffineMapAttr::get(newMap), op->getPadding(), op.getMask(),
      //   newInBoundsAttr);

      // OpBuilder builder(op);
      // Value output = newOutput.getResult();
      // auto transferReadOp = builder.create<vector::TransferReadOp>(loc, inputTy, output, indices, inBounds);
      // auto transferReadOp = rewriter.create<vector::TransferReadOp>(loc, inputTy, newOutput.getResult(0), indices, inBounds);

      // llvm::dbgs() << "Created vector.transfer_read:\n";
      // transferReadOp.dump();    

      // llvm::dbgs() << "\nCurrent operation:\n";
      // op->dump();

      // scf::ForOp forOp = op->getParentOfType<scf::ForOp>();
      // llvm::dbgs() << "\nforOp:\n";
      // forOp.dump();

      // Value loopArg3 = forOp.getRegionIterArgs()[0];
      // llvm::dbgs() << "\nloopArg3 is :\n";
      // loopArg3.dump();

      // SmallVector<Value, 4> indexVals = { c0, c0, c0, c0 };
      // SmallVector<Value, 8> opOperands;
      // opOperands.push_back(transferReadOp);
      // opOperands.push_back(loopArg3);
      // for (auto idx : indexVals)
      //   opOperands.push_back(idx); 
      // SmallVector<Type, 1> resultTypes = { loopArg3.getType() };
      // auto inBoundsAttr = rewriter.getBoolArrayAttr({true, true, true, true});
      // SmallVector<NamedAttribute, 1> attributes;
      // attributes.push_back(rewriter.getNamedAttr("in_bounds", inBoundsAttr));
      // auto transferWriteOp = rewriter.create<vector::TransferWriteOp>(
      //   loc, resultTypes, opOperands, attributes
      // );
      // llvm::dbgs() << "Created vector.transfer_write:\n";
      // transferWriteOp.dump();

      // rewriter.setInsertionPointToEnd(forOp.getBody());
      // rewriter.replaceOpWithNewOp<scf::YieldOp>(
      //     forOp.getBody()->getTerminator(),
      //     transferWriteOp.getResult()
      // );
      // llvm::dbgs() << "\nUpdated scf.yield operation with transfer_write result.\n";

      // continue;

    }
    
    ///////////////////////////////////////////////////////////////
    //                 Skip original CONV2D OP                   //
    //                Call Accelerator CONV2D API                //
    ///////////////////////////////////////////////////////////////

    if ((isa<linalg::Conv1DOp>(op))
        || (isa<linalg::Conv1DNcwFcwOp>(op))
        || (isa<linalg::Conv1DNwcWcfOp>(op))
        || (isa<linalg::Conv2DNhwcHwcfOp>(op))
        ) {
      // break;
      SmallVector<Type> callArgumentTypes;
      SmallVector<Type> callReturnTypes;
      SmallVector<Value> operandsRange;

      auto loc = op->getLoc();
      
      rewriter.setInsertionPoint(op);

      // Creating corresponding memref input type and new memref input
      // May add layout to memrefType later
      auto operands = op->getOperands();
      for (int i = 0; i < operands.size() - 1; i++) {
        auto operand = operands[i];
        TensorType tensorType = operand.getType().cast<TensorType>();
        MemRefType memrefType = MemRefType::get(  
            tensorType.getShape(),
            tensorType.getElementType()
        );
        auto newOperand = rewriter.create<bufferization::ToMemrefOp>(
          loc, memrefType, operand);
        operandsRange.push_back(newOperand);
        callArgumentTypes.push_back(memrefType);
        newOperand.dump();
        memrefType.dump();
      }

      llvm::dbgs() << "Input operands: ";
      for (auto operand : operandsRange) {
        operand.dump();
      }

      llvm::dbgs() << "Input types: ";
      for (auto type : callArgumentTypes) {
        type.dump();
      }

      // Creating corresponding memref return type
      // May add layout to memrefType later
      for (auto returnType : op->getResultTypes()) {
        TensorType tensorType = returnType.cast<TensorType>();
        MemRefType memrefType = MemRefType::get(  
            tensorType.getShape(),
            tensorType.getElementType()
        );
        memrefType.dump();
        callReturnTypes.push_back(memrefType);
      }
      llvm::dbgs() << "Output types: ";
      for (auto type : callReturnTypes) {
        type.dump();
      }

      // Obtain desired function op type
      auto functionType = rewriter.getFunctionType(callArgumentTypes, callReturnTypes);
      llvm::dbgs() << "Function type: ";
      functionType.dump();

      // Find desired function op
      mlir::StringRef fnName = mlir::StringRef("npu_conv_1d");
      auto moduleOp = SymbolTable::getNearestSymbolTable(op);
      auto fnDecl = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(moduleOp, fnName));

      // Create Function Op if not found
      if (!fnDecl) {
        // Set correct insertion point for function op
        rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());

        // Create attributes
        auto linkageAttr = LLVM::LinkageAttr::get(rewriter.getContext(), LLVM::Linkage::External);
        SmallVector<NamedAttribute> funcAttrs;
        funcAttrs.push_back(rewriter.getNamedAttr("llvm.linkage", linkageAttr));

        fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType, funcAttrs);
        SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
      } else {
        llvm::dbgs() << "\nDid not create<func::FuncOp>!\n";
      }

      // Restore insertion point
      rewriter.setInsertionPoint(op);

      // Create call op
      auto symbolRef = mlir::SymbolRefAttr::get(context, fnName);
      auto newOp = rewriter.create<func::CallOp>(loc, symbolRef, callReturnTypes, operandsRange);

      // Replace previous call op with new call op while keeping data type unchanged
      Value outputOperand = newOp.getResult(0);
      auto newOutput = rewriter.create<bufferization::ToTensorOp>(loc, outputOperand, true);
      rewriter.replaceOp(op, newOutput);

      continue;
    }
    
    if ((isa<linalg::Conv2DOp>(op))
        || (isa<linalg::Conv2DNchwFchwOp>(op))
        || (isa<linalg::Conv2DNgchwFgchwOp>(op))
        || (isa<linalg::Conv2DNgchwGfchwOp>(op))
        //|| (isa<linalg::Conv2DNgchwGfchwQOp>(op))
        || (isa<linalg::Conv2DNhwcFhwcOp>(op))
        || (isa<linalg::Conv2DNhwcFhwcQOp>(op))
        || (isa<linalg::Conv2DNhwcHwcfOp>(op))
        || (isa<linalg::Conv2DNhwcHwcfQOp>(op))
        ) {
      //llvm::dbgs() << "\n\n<linalg::Conv1D>!\n\n";
      continue;
    }
    if (isa<linalg::Conv3DOp>(op)
        || (isa<linalg::Conv3DNcdhwFcdhwOp>(op))
        || (isa<linalg::Conv3DNdhwcDhwcfOp>(op))
        || (isa<linalg::Conv3DNdhwcDhwcfQOp>(op))
        ) {
      //llvm::dbgs() << "\n\n<linalg::Conv3D>!\n\n";
      continue;
    }
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not vectorize the op if the vector size is greater than or equal
      // to limit.
      if (enableVectorMasking) {
        auto vectorSizesAndScalableDims =
            getVectorSizes(linalgOp, useConfiguredVectorSizes);
        if (succeeded(vectorSizesAndScalableDims)) {
          auto [sizes, scalableDims] = *vectorSizesAndScalableDims;
          vectorSizes.append(sizes.begin(), sizes.end());
          scalableVecDims.append(scalableDims.begin(), scalableDims.end());
        }
        if (std::accumulate(vectorSizes.begin(), vectorSizes.end(), 1,
                            std::multiplies<int64_t>()) >= maxVectorSize)
          continue;
      } else {
        if (failed(isWithinVectorSizeLimit(linalgOp, maxVectorSize)))
          continue;
      }
    } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      auto ty = padOp.getResultType();
      // TODO(hanchung): Infer the vector sizes for pad op after
      // maskedVectorize method allows dynamic result shapes.
      if (!ty.hasStaticShape())
        continue;
      vectorSizes.append(ty.getShape().begin(), ty.getShape().end());
    }
    // Pad scalable dims with `false` to match the vector sizes.
    scalableVecDims.resize(vectorSizes.size());
    (void)linalg::vectorize(rewriter, op, vectorSizes, scalableVecDims,
                            vectorizeGatherAccesses);
    
    ///////////////////////////////////////////////////////////////
    //                       NPU Fuse OP                         //
    //                Call Accelerator CONV2D API                //
    ///////////////////////////////////////////////////////////////
    // 支持的fuse pattern
    if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddReluOp>(op)) {
      if (auto convOp = dyn_cast<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddReluOp>(op))
        rewriteNpuFuseOpToCall(rewriter, context, convOp);
      continue;
    }

    if (isa<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddBnReluOp>(op)) {
      if (auto convOp = dyn_cast<mlir::iree::compiler::Dialect::NPUFuseOp::ConvAddBnReluOp>(op))
        rewriteNpuFuseOpToCall(rewriter, context, convOp);
      continue;
    }
  };

  {
    // Canonicalize mask related ops before we lower them.
    RewritePatternSet maskCanonPatterns(funcOp.getContext());
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                      funcOp.getContext());
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                        funcOp.getContext());
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                funcOp.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(maskCanonPatterns)))) {
      return signalPassFailure();
    }
  }

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  if (generateContract) {
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  }
  if (foldCastIntoContract) {
    vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  }
  if (enableVectorMasking) {
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
  }

  if (enableCleanup) {
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
  }
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  // Apply the pad tensor op vectorization separately to avoid running the
  // GenericPadOpVectorizationPattern too early.
  // TODO: Improve once we have better infrastructure to control pattern
  // application.
  if (vectorizePadding) {
    RewritePatternSet patterns(funcOp.getContext());
    linalg::populatePadOpVectorizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGenericVectorizationPass() {
  return std::make_unique<GenericVectorizationPass>();
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGenericVectorizationPass(const GenericVectorizationPassOptions &options) {
  return std::make_unique<GenericVectorizationPass>(options);
}

} // namespace mlir::iree_compiler
