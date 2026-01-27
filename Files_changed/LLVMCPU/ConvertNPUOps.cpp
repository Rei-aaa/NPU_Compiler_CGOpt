#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h" 
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"


namespace mlir::iree_compiler{
namespace {

///////////////////////////////////////////////////////////////
//                  NPU Call Op Conversion                   //
///////////////////////////////////////////////////////////////

class ConvertNPUCallOpPattern : public OpRewritePattern<func::CallOp> {
    public:
    using OpRewritePattern::OpRewritePattern;

    private:
    LogicalResult matchAndRewrite(func::CallOp callOp,
            PatternRewriter &rewriter) const override {
        llvm::dbgs() << "ConvertNPUCallOpPattern: matchAndRewrite\n";

        if (!callOp.getCallee().contains("npu_")) {
            llvm::dbgs() << "Not an NPU call, skipping\n";
            return failure();
        }

        auto context = callOp.getContext();
        Location loc = callOp.getLoc();
        auto module = callOp.getOperation()->getParentOfType<ModuleOp>();

        SmallVector<Value> operands;
        SmallVector<Value> newOperands;
        for (auto operand : callOp.getOperands()) {
            operands.push_back(operand);
        }

        module.dump();
        llvm::dbgs() << "\n\n";

        // convert memref to llvm ptr as new operands
        Type llvmptrType = LLVM::LLVMPointerType::get(context);
        for (Value operand : operands) {
            llvm::dbgs() << "operand:\n";
            operand.dump();
            Value structOperand = operand.getDefiningOp()->getOperand(0);
            structOperand.dump();
            Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(
                loc, llvmptrType, structOperand, 1);
            newOperands.push_back(alignedPtr);
            operand.replaceAllUsesWith(alignedPtr);
            alignedPtr.dump();
        }

        llvm::dbgs() << "operands converted!\n\n";

        module.dump();

        // CallOp Creation
        auto newCallOp = rewriter.create<LLVM::CallOp>(
            loc, llvmptrType, callOp.getCallee(), newOperands);

        // 继承原attributes，后续需要npuoprands
        SmallVector<NamedAttribute> callAttrs;
        for (auto attr : callOp->getAttrs()) {
            if (attr.getName() != callOp.getCalleeAttrName())
                callAttrs.push_back(attr);
        }
        newCallOp->setAttrs(callAttrs);

        Value returnedPtr = newCallOp.getResult();
        Type callOpType = callOp.getResult(0).getType();
        MemRefType memrefType = callOpType.cast<MemRefType>();

        unsigned rank = memrefType.getRank();
        Type i64Type = IntegerType::get(context, 64);
        SmallVector<Type, 5> structElements{
            llvmptrType,     // Allocated pointer
            llvmptrType,     // Aligned pointer
            i64Type,         // Offset
            LLVM::LLVMArrayType::get(i64Type, rank),  // Sizes array
            LLVM::LLVMArrayType::get(i64Type, rank)   // Strides array
        };
        auto structType = LLVM::LLVMStructType::getLiteral(context, structElements);

        Value zero = rewriter.create<LLVM::ConstantOp>(loc, i64Type, 0);

        llvm::dbgs() << "UndefOp dump!\n";
        Value structValue = rewriter.create<LLVM::UndefOp>(loc, structType);
        structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, returnedPtr,
            rewriter.getDenseI64ArrayAttr({0}));
        structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, returnedPtr,
            rewriter.getDenseI64ArrayAttr({1}));
        structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, zero,
            rewriter.getDenseI64ArrayAttr({2}));

        structValue.dump();

        int64_t stride = 1;
        ArrayRef<int64_t> shapes = memrefType.getShape();
        for (int i = shapes.size() - 1; i >= 0; i--) {
            Value dimSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, shapes[i]);
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, dimSize,
                rewriter.getDenseI64ArrayAttr({3, i}));

            Value strideVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type, stride);
            structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, strideVal,
                rewriter.getDenseI64ArrayAttr({4, i}));

            if (i > 0) {
                stride *= shapes[i];
            }
        }

        callOp.getResult(0).replaceAllUsesWith(structValue);
        rewriter.eraseOp(callOp);

        structValue.dump();
        module.dump();

        for (Value operand : operands) {
            rewriter.eraseOp(operand.getDefiningOp());
        }

        return success();
    }
};

///////////////////////////////////////////////////////////////
//                 NPU Func Op Declaration                   //
///////////////////////////////////////////////////////////////

class ConvertNPUFuncOpPattern : public OpRewritePattern<func::FuncOp> {
public:
    using OpRewritePattern::OpRewritePattern; 
    private:
    LogicalResult matchAndRewrite(func::FuncOp op, 
            PatternRewriter &rewriter) const override {
        if (!op.getSymName().contains("npu_")) {
            return failure();
        }

        auto llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        // Create LLVM function type with the same number of operands as the original FuncOp.
        SmallVector<Type> argTypes(op.getFunctionType().getNumInputs(), llvmPtrType);
        auto llvmFuncType = LLVM::LLVMFunctionType::get(llvmPtrType, argTypes, false);

        rewriter.setInsertionPoint(op);
        auto newFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(),
            op.getName(), llvmFuncType, LLVM::Linkage::External);

        // Preserve non-signature attributes (e.g., npu metadata) on the LLVM function.
        for (auto attr : op->getAttrs()) {
            if (attr.getName() == SymbolTable::getSymbolAttrName() ||
            attr.getName() == op.getFunctionTypeAttrName() ||
            attr.getName() == "llvm.linkage") {
            continue;
            }
            newFunc->setAttr(attr.getName(), attr.getValue());
        }
        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

///////////////////////////////////////////////////////////////
//                     Pattern Registration                  //
///////////////////////////////////////////////////////////////

void populateNPUOpsPatterns(RewritePatternSet &patterns) {
    patterns.add<ConvertNPUFuncOpPattern,
        ConvertNPUCallOpPattern>(patterns.getContext());
}

///////////////////////////////////////////////////////////////
//                     Convert NPU Pass                      //
///////////////////////////////////////////////////////////////

struct ConvertNPUOpsPass
        : public PassWrapper<ConvertNPUOpsPass, OperationPass<ModuleOp>> {

    StringRef getArgument() const override { return "convert-npu-ops"; }
    StringRef getDescription() const override { 
        return "Convert NPU ops to LLVM dialect"; } 
    
    void runOnOperation() override {
        llvm::dbgs() << "\nConvertNPUOpsPass:runOnOperation!\n";
        auto module = getOperation();
        MLIRContext *context = &getContext();

        // Converting NPUCallOp and NPUFuncOp
        RewritePatternSet new_patterns(context);
        new_patterns.add<ConvertNPUFuncOpPattern, ConvertNPUCallOpPattern>(context);
            
        if (failed(applyPatternsAndFoldGreedily(module, std::move(new_patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<OperationPass<ModuleOp>>
createConvertNPUOpsPass() {
    return std::make_unique<ConvertNPUOpsPass>();
}

} // namespace mlir::iree_compiler