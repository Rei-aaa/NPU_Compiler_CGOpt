#include "FuseConvVecFunc.h"                
#include <functional>
#include <string>
#include <utility>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
 
namespace mlir::iree_compiler {

static bool combineAddSubPair(linalg::GenericOp genericOp,
                              Operation *firstOp,
                              Operation *secondOp,
                              PatternRewriter &rewriter,
                                        FusionPatternInfo &pattern,
                              Value &newResult,
                                        bool enableFastMath);
static bool combineMulPair(linalg::GenericOp genericOp,
                           Operation *firstOp,
                           Operation *secondOp,
                           PatternRewriter &rewriter,
                                    FusionPatternInfo &pattern,
                           Value &newResult,
                                    bool enableFastMath);
static Operation *createFusedOpFromPattern(StringRef fusedName,
                                           Operation *convOp,
                                           const FusionPatternInfo &pattern,
                                           ArrayRef<Value> operands,
                                           PatternRewriter &rewriter);
static Value mapBlockArgToGenericValue(linalg::GenericOp genericOp, Value v);
static bool isZeroConstant(Value v);
static bool isFloatConstant(Value v, double expectedValue);
static Value remapValue(Value v,
                        const llvm::DenseMap<Value, Value> &valueRemap);
static Value chooseActivationInput(Value lhs, Value rhs,
                                   ArrayRef<Operation *> filteredOps,
                                   const llvm::DenseMap<Value, Value> &valueRemap);
static bool getGenericOperandIndex(linalg::GenericOp genericOp, Value v,
                                   int &operandIndex);
static Value getGenericInputValue(linalg::GenericOp genericOp,
                                  int operandIndex);
static bool foldSingleSubToAdd(linalg::GenericOp genericOp,
                               arith::SubFOp subOp,
                               PatternRewriter &rewriter,
                               FusionPatternInfo &pattern,
                               Value &newResult);
static bool getGenericInputOperandIndex(linalg::GenericOp genericOp, Value v,
                                        int &operandIndex);
enum class BinaryKind { Add, Sub, Mul };
static Value createSplatTensor(PatternRewriter &rewriter, Location loc,
                               ShapedType type, Value like, double value);
static Value createElementwiseBinaryTensorOp(PatternRewriter &rewriter,
                                             Location loc, Value lhs, Value rhs,
                                             BinaryKind kind);

struct ActivationMatch {
    StringRef type;
    Value input;
    Value output;
    size_t consumed = 0;
    llvm::SmallVector<Operation *, 6> ops;
};

static bool matchActivation(size_t index,
                              ArrayRef<Operation *> elementwiseOps,
                              ActivationMatch &match);
static bool hasNonTerminatorBodyOps(linalg::GenericOp genericOp);
static bool genericUsesOperandValue(linalg::GenericOp genericOp, Value v);
static bool cleanupResidualGenericAfterFusion(
    linalg::GenericOp genericOp, ArrayRef<Operation *> consumedOps,
    PatternRewriter &rewriter, linalg::GenericOp &updatedGenericOp,
    int forcePreserveInputIndex = -1);

/////////////////////////////////////////////////////////////////////
//                main functions used in rewrite                   //
/////////////////////////////////////////////////////////////////////

// 收集逐元素操作 collect fusible elementwise ops connected to (convOp single-use chain)
bool collectFusableOps(Operation *convOp,
                       Operation *&fusedGenericOp,
                       ElementwiseChain &elementwiseOps,
                       ElementwiseChain &mainDataChain) {
    fusedGenericOp = nullptr;
    elementwiseOps.clear();
    mainDataChain.clear();

    while (!convOp->getResult(0).use_empty()) {
        if (!convOp->getResult(0).hasOneUse())
            break;
        Operation *user = *convOp->getResult(0).getUsers().begin();
        if (!user || user->getBlock() != convOp->getBlock())
            break;
        auto genericOp = dyn_cast<linalg::GenericOp>(user);
        if (!genericOp)
            break;
        Block *bodyBlock = genericOp.getBody();
        for (Operation &inner : *bodyBlock) {
            if (inner.hasTrait<OpTrait::IsTerminator>())
                continue;
            elementwiseOps.push_back(&inner);
        }

        int streamBlockArgIndex = -1;
        for (auto it : llvm::enumerate(genericOp.getDpsInputs())) {
            if (it.value() == convOp->getResult(0)) {
                streamBlockArgIndex = static_cast<int>(it.index());
                break;
            }
        }
        if (streamBlockArgIndex < 0) {
            for (auto it : llvm::enumerate(genericOp.getDpsInits())) {
                if (it.value() == convOp->getResult(0)) {
                    streamBlockArgIndex =
                        static_cast<int>(genericOp.getNumDpsInputs() +
                                         it.index());
                    break;
                }
            }
        }

        if (streamBlockArgIndex >= 0) {
            Value current =
                genericOp.getBody()->getArgument(streamBlockArgIndex);
            size_t scanFrom = 0;
            while (current) {
                size_t firstUseIndex = elementwiseOps.size();
                for (size_t i = scanFrom; i < elementwiseOps.size(); ++i) {
                    Operation *op = elementwiseOps[i];
                    if (!op)
                        continue;
                    for (Value operand : op->getOperands()) {
                        if (operand == current) {
                            firstUseIndex = i;
                            break;
                        }
                    }
                    if (firstUseIndex != elementwiseOps.size())
                        break;
                }

                if (firstUseIndex == elementwiseOps.size())
                    break;

                ActivationMatch activation;
                if (matchActivation(firstUseIndex, elementwiseOps, activation) &&
                    activation.input == current) {
                    mainDataChain.append(activation.ops.begin(),
                                         activation.ops.end());
                    current = activation.output;
                    scanFrom = firstUseIndex + activation.consumed;
                    continue;
                }

                Operation *next = elementwiseOps[firstUseIndex];
                mainDataChain.push_back(next);
                if (next->getNumResults() == 0)
                    break;
                Value nextResult = next->getResult(0);
                if (nextResult == current)
                    break;
                current = nextResult;
                scanFrom = firstUseIndex + 1;
            }
        }

        fusedGenericOp = user;
        break;
    }
    return fusedGenericOp != nullptr;
}

// 折叠&预处理 fold and preprocess
void foldConstantElementwiseOps(linalg::GenericOp genericOp,
                                ElementwiseChain &elementwiseOps,
                                PatternRewriter &rewriter,
                                FusionPatternInfo &pattern,
                                bool enableFastMath) {
    (void)enableFastMath;
    (void)pattern;
    if (!genericOp || elementwiseOps.empty())
        return;

    OpBuilder::InsertionGuard guard(rewriter);
    llvm::DenseSet<Operation *> mainOpSet(elementwiseOps.begin(),
                                          elementwiseOps.end());

    enum class ExprKind { Input, Const, Add, Sub, Mul };
    struct ExprNode {
        ExprKind kind = ExprKind::Input;
        int inputIndex = -1;
        int lhs = -1;
        int rhs = -1;
        double constValue = 0.0;
    };

    llvm::SmallVector<ExprNode, 16> exprs;
    llvm::DenseMap<int, int> inputExprByOperand;
    llvm::DenseMap<int, int> updatedExprByOperand;

    auto addInputExpr = [&](int operandIndex) -> int {
        auto it = inputExprByOperand.find(operandIndex);
        if (it != inputExprByOperand.end())
            return it->second;
        int id = static_cast<int>(exprs.size());
        ExprNode node;
        node.kind = ExprKind::Input;
        node.inputIndex = operandIndex;
        exprs.push_back(node);
        inputExprByOperand[operandIndex] = id;
        return id;
    };

    auto getCurrentExpr = [&](int operandIndex) -> int {
        auto it = updatedExprByOperand.find(operandIndex);
        if (it != updatedExprByOperand.end())
            return it->second;
        return addInputExpr(operandIndex);
    };

    auto addConstExpr = [&](double value) -> int {
        int id = static_cast<int>(exprs.size());
        ExprNode node;
        node.kind = ExprKind::Const;
        node.constValue = value;
        exprs.push_back(node);
        return id;
    };

    auto addBinaryExpr = [&](ExprKind kind, int lhs, int rhs) -> int {
        int id = static_cast<int>(exprs.size());
        ExprNode node;
        node.kind = kind;
        node.lhs = lhs;
        node.rhs = rhs;
        exprs.push_back(node);
        return id;
    };

    auto assignUpdatedExpr = [&](int operandIndex, int exprId) {
        updatedExprByOperand[operandIndex] = exprId;
    };

    llvm::SmallVector<Operation *, 12> eraseOps;
    eraseOps.reserve(elementwiseOps.size());
    llvm::SmallVector<arith::SubFOp, 8> pendingSubToAdd;

    for (size_t i = 0; i < elementwiseOps.size();) {
        Operation *first = elementwiseOps[i];
        if (!first || first->getBlock() != genericOp.getBody()) {
            ++i;
            continue;
        }

        Value newResult;
        bool folded = false;

        Operation *second = nullptr;
        if (first->hasOneUse()) {
            second = *first->getResult(0).getUsers().begin();
            if (!llvm::is_contained(elementwiseOps, second))
                second = nullptr;
        }
        if (second && second->getBlock() == genericOp.getBody()) {
                if (auto addOp = dyn_cast<arith::AddFOp>(first)) {
                    if (first->hasOneUse() &&
                        *first->getResult(0).getUsers().begin() == second) {
                        Value secondSideArg;
                        ExprKind combineKind = ExprKind::Add;
                        if (auto secondAdd = dyn_cast<arith::AddFOp>(second)) {
                            if (secondAdd.getLhs() == addOp.getResult()) {
                                secondSideArg = secondAdd.getRhs();
                                combineKind = ExprKind::Add;
                            } else if (secondAdd.getRhs() == addOp.getResult()) {
                                secondSideArg = secondAdd.getLhs();
                                combineKind = ExprKind::Add;
                            }
                        } else if (auto secondSub = dyn_cast<arith::SubFOp>(second)) {
                            if (secondSub.getLhs() == addOp.getResult()) {
                                secondSideArg = secondSub.getRhs();
                                combineKind = ExprKind::Sub;
                            }
                        }

                        if (secondSideArg) {
                            int biasIndex = -1;
                            int secondSideIndex = -1;
                            if (getGenericInputOperandIndex(genericOp,
                                                            secondSideArg,
                                                            secondSideIndex)) {
                                if (addOp.getLhs() != secondSideArg &&
                                    getGenericInputOperandIndex(genericOp,
                                                                addOp.getLhs(),
                                                                biasIndex)) {
                                    // use lhs
                                } else if (addOp.getRhs() != secondSideArg &&
                                           getGenericInputOperandIndex(
                                               genericOp, addOp.getRhs(),
                                               biasIndex)) {
                                    // use rhs
                                }
                            }

                            if (biasIndex >= 0 && secondSideIndex >= 0) {
                                Value biasTensor = getGenericInputValue(genericOp,
                                                                        biasIndex);
                                Value sideTensor = getGenericInputValue(
                                    genericOp, secondSideIndex);
                                if (biasTensor && sideTensor &&
                                    biasTensor.getType() == sideTensor.getType()) {
                                    int lhsExpr = getCurrentExpr(biasIndex);
                                    int rhsExpr = getCurrentExpr(secondSideIndex);
                                    int newBiasExpr = addBinaryExpr(combineKind,
                                                                    lhsExpr,
                                                                    rhsExpr);
                                    int zeroExpr = addConstExpr(0.0);
                                    assignUpdatedExpr(biasIndex, newBiasExpr);
                                    assignUpdatedExpr(secondSideIndex, zeroExpr);

                                    second->getResult(0).replaceAllUsesWith(
                                        addOp.getResult());
                                    eraseOps.push_back(second);
                                    // Keep the current chain head and keep
                                    // folding with the next op, enabling
                                    // 3+ connected add/sub chains.
                                    auto secondIt = llvm::find(elementwiseOps,
                                                               second);
                                    if (secondIt != elementwiseOps.end())
                                        elementwiseOps.erase(secondIt);
                                    folded = true;
                                }
                            }
                        }
                    }
                }

                if (!folded) {
                    if (auto firstMul = dyn_cast<arith::MulFOp>(first)) {
                        auto secondMul = dyn_cast<arith::MulFOp>(second);
                        if (first->hasOneUse() && secondMul &&
                            *first->getResult(0).getUsers().begin() == second &&
                            (secondMul.getLhs() == firstMul.getResult() ||
                             secondMul.getRhs() == firstMul.getResult())) {
                            int varianceIndex = -1;
                            int scaleIndex = -1;
                            if (!getGenericInputOperandIndex(genericOp,
                                                             firstMul.getLhs(),
                                                             varianceIndex)) {
                                getGenericInputOperandIndex(genericOp,
                                                            firstMul.getRhs(),
                                                            varianceIndex);
                            }
                            Value scaleArg = secondMul.getLhs() ==
                                                     firstMul.getResult()
                                                 ? secondMul.getRhs()
                                                 : secondMul.getLhs();
                            getGenericInputOperandIndex(genericOp, scaleArg,
                                                        scaleIndex);

                            if (varianceIndex >= 0 && scaleIndex >= 0) {
                                Value varianceTensor = getGenericInputValue(
                                    genericOp, varianceIndex);
                                Value scaleTensor = getGenericInputValue(
                                    genericOp, scaleIndex);
                                if (varianceTensor && scaleTensor &&
                                    varianceTensor.getType() ==
                                        scaleTensor.getType()) {
                                    int lhsExpr = getCurrentExpr(varianceIndex);
                                    int rhsExpr = getCurrentExpr(scaleIndex);
                                    int newScaleExpr = addBinaryExpr(
                                        ExprKind::Mul, lhsExpr, rhsExpr);
                                    int oneExpr = addConstExpr(1.0);
                                    assignUpdatedExpr(varianceIndex,
                                                      newScaleExpr);
                                    assignUpdatedExpr(scaleIndex, oneExpr);

                                    secondMul.getResult().replaceAllUsesWith(
                                        firstMul.getResult());
                                    eraseOps.push_back(secondMul.getOperation());
                                    // Keep the current chain head and keep
                                    // folding with the next op, enabling
                                    // 3+ connected mul chains.
                                    auto secondIt = llvm::find(
                                        elementwiseOps,
                                        secondMul.getOperation());
                                    if (secondIt != elementwiseOps.end())
                                        elementwiseOps.erase(secondIt);
                                    folded = true;
                                }
                            }
                        }
                    }
                }
        }

        if (folded)
            continue;

        if (auto subOp = dyn_cast<arith::SubFOp>(first)) {
            int rhsIndex = -1;
            if (getGenericInputOperandIndex(genericOp, subOp.getRhs(), rhsIndex) &&
                rhsIndex >= 0) {
                Value rhsTensor = getGenericInputValue(genericOp, rhsIndex);
                if (rhsTensor && rhsTensor.getType().isa<RankedTensorType>()) {
                    int zeroExpr = addConstExpr(0.0);
                    int rhsExpr = getCurrentExpr(rhsIndex);
                    int negExpr = addBinaryExpr(ExprKind::Sub, zeroExpr, rhsExpr);
                    assignUpdatedExpr(rhsIndex, negExpr);
                    pendingSubToAdd.push_back(subOp);
                    ++i;
                    continue;
                }
            }
        }
        ++i;
    }

    bool updatedGenericOperands = false;
    if (!updatedExprByOperand.empty()) {
        llvm::SmallVector<int, 8> targetOperands;
        targetOperands.reserve(updatedExprByOperand.size());
        for (auto it : updatedExprByOperand)
            targetOperands.push_back(it.first);
        llvm::sort(targetOperands);

        llvm::SmallVector<int, 8> inputOperands;
        llvm::DenseSet<int> inputVisited;

        std::function<void(int)> collectInputs = [&](int exprId) {
            const ExprNode &node = exprs[exprId];
            if (node.kind == ExprKind::Input) {
                if (!inputVisited.contains(node.inputIndex)) {
                    inputVisited.insert(node.inputIndex);
                    inputOperands.push_back(node.inputIndex);
                }
                return;
            }
            if (node.kind == ExprKind::Const)
                return;
            collectInputs(node.lhs);
            collectInputs(node.rhs);
        };

        for (int operandIndex : targetOperands)
            collectInputs(updatedExprByOperand.lookup(operandIndex));
        llvm::sort(inputOperands);

        auto firstType = getGenericInputValue(genericOp, targetOperands.front())
                             .getType()
                             .dyn_cast<RankedTensorType>();
        if (firstType) {
            bool allCompatible = true;
            for (int operandIndex : targetOperands) {
                auto t = getGenericInputValue(genericOp, operandIndex)
                             .getType()
                             .dyn_cast<RankedTensorType>();
                if (!t || t != firstType) {
                    allCompatible = false;
                    break;
                }
            }

            if (allCompatible) {
                SmallVector<Value, 8> ins;
                SmallVector<AffineMap, 16> maps;
                llvm::DenseMap<int, int> inputArgPos;
                unsigned numInputs = genericOp.getNumDpsInputs();
                llvm::SmallVector<int, 8> requiredInputIndices = inputOperands;
                ins.reserve(requiredInputIndices.size());
                maps.reserve(requiredInputIndices.size() + targetOperands.size());
                auto identity =
                    rewriter.getMultiDimIdentityMap(firstType.getRank());

                llvm::SmallVector<Operation *, 12> supportOps;
                llvm::DenseSet<Operation *> supportSet;
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (Operation &op : genericOp.getBody()->getOperations()) {
                        if (op.hasTrait<OpTrait::IsTerminator>())
                            continue;
                        if (mainOpSet.contains(&op) || supportSet.contains(&op))
                            continue;

                        bool neededByMain = false;
                        for (Operation *mainOp : mainOpSet) {
                            if (!mainOp)
                                continue;
                            for (Value operand : mainOp->getOperands()) {
                                if (operand.getDefiningOp() == &op) {
                                    neededByMain = true;
                                    break;
                                }
                            }
                            if (neededByMain)
                                break;
                        }

                        bool neededBySupport = false;
                        if (!neededByMain) {
                            for (Operation *supportOp : supportOps) {
                                for (Value operand : supportOp->getOperands()) {
                                    if (operand.getDefiningOp() == &op) {
                                        neededBySupport = true;
                                        break;
                                    }
                                }
                                if (neededBySupport)
                                    break;
                            }
                        }

                        if (!neededByMain && !neededBySupport)
                            continue;

                        bool mappable = true;
                        for (Value operand : op.getOperands()) {
                            if (auto arg = operand.dyn_cast<BlockArgument>()) {
                                if (arg.getOwner() != genericOp.getBody() ||
                                    arg.getArgNumber() >= numInputs) {
                                    mappable = false;
                                    break;
                                }
                                int idx = static_cast<int>(arg.getArgNumber());
                                if (!llvm::is_contained(requiredInputIndices,
                                                        idx)) {
                                    requiredInputIndices.push_back(idx);
                                }
                                continue;
                            }
                            Operation *def = operand.getDefiningOp();
                            if (!def || !supportSet.contains(def)) {
                                mappable = false;
                                break;
                            }
                        }
                        if (!mappable)
                            continue;

                        supportOps.push_back(&op);
                        supportSet.insert(&op);
                        changed = true;
                    }
                }

                llvm::sort(requiredInputIndices);
                requiredInputIndices.erase(
                    std::unique(requiredInputIndices.begin(),
                                requiredInputIndices.end()),
                    requiredInputIndices.end());

                for (auto it : llvm::enumerate(requiredInputIndices)) {
                    int idx = it.value();
                    ins.push_back(getGenericInputValue(genericOp, idx));
                    maps.push_back(identity);
                    inputArgPos[idx] = static_cast<int>(it.index());
                }

                SmallVector<Value, 8> outs;
                SmallVector<Type, 8> resultTypes;
                outs.reserve(targetOperands.size());
                resultTypes.reserve(targetOperands.size());
                for (int operandIndex : targetOperands) {
                    auto t = getGenericInputValue(genericOp, operandIndex)
                                 .getType()
                                 .cast<RankedTensorType>();
                    outs.push_back(rewriter.create<tensor::EmptyOp>(
                        genericOp.getLoc(), t.getShape(), t.getElementType()));
                    resultTypes.push_back(t);
                    maps.push_back(identity);
                }

                SmallVector<utils::IteratorType, 4> iterators(
                    firstType.getRank(), utils::IteratorType::parallel);

                rewriter.setInsertionPoint(genericOp);
                auto fusedGeneric = rewriter.create<linalg::GenericOp>(
                    genericOp.getLoc(), resultTypes, ins, outs, maps, iterators,
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                        IRMapping supportMapper;
                        for (auto it : llvm::enumerate(requiredInputIndices)) {
                            int idx = it.value();
                            supportMapper.map(
                                genericOp.getBody()->getArgument(
                                    static_cast<unsigned>(idx)),
                                args[it.index()]);
                        }
                        for (Operation *supportOp : supportOps)
                            b.clone(*supportOp, supportMapper);

                        std::function<Value(int, Type)> emitExpr =
                            [&](int exprId, Type elemType) -> Value {
                            const ExprNode &node = exprs[exprId];
                            switch (node.kind) {
                            case ExprKind::Input:
                                return args[inputArgPos.lookup(node.inputIndex)];
                            case ExprKind::Const:
                                return b.create<arith::ConstantOp>(
                                            loc,
                                            b.getFloatAttr(elemType.cast<FloatType>(),
                                                           node.constValue))
                                    .getResult();
                            case ExprKind::Add: {
                                Value lhs = emitExpr(node.lhs, elemType);
                                Value rhs = emitExpr(node.rhs, elemType);
                                return b.create<arith::AddFOp>(loc, lhs, rhs)
                                    .getResult();
                            }
                            case ExprKind::Sub: {
                                Value lhs = emitExpr(node.lhs, elemType);
                                Value rhs = emitExpr(node.rhs, elemType);
                                return b.create<arith::SubFOp>(loc, lhs, rhs)
                                    .getResult();
                            }
                            case ExprKind::Mul: {
                                Value lhs = emitExpr(node.lhs, elemType);
                                Value rhs = emitExpr(node.rhs, elemType);
                                return b.create<arith::MulFOp>(loc, lhs, rhs)
                                    .getResult();
                            }
                            }
                            return Value();
                        };

                        SmallVector<Value, 8> yielded;
                        yielded.reserve(targetOperands.size());
                        for (int operandIndex : targetOperands) {
                            auto elemType = getGenericInputValue(genericOp,
                                                                 operandIndex)
                                                .getType()
                                                .cast<RankedTensorType>()
                                                .getElementType();
                            yielded.push_back(
                                emitExpr(updatedExprByOperand.lookup(operandIndex),
                                         elemType));
                        }
                        b.create<linalg::YieldOp>(loc, yielded);
                    });

                for (auto it : llvm::enumerate(targetOperands)) {
                    int operandIndex = it.value();
                    Value newV = fusedGeneric->getResult(it.index());
                    genericOp.setOperand(operandIndex, newV);
                }
                updatedGenericOperands = true;
            }
        }
    }

    if (updatedGenericOperands) {
        for (arith::SubFOp subOp : pendingSubToAdd) {
            if (!subOp)
                continue;
            rewriter.setInsertionPoint(subOp);
            auto addOp = rewriter.create<arith::AddFOp>(
                subOp.getLoc(), subOp.getLhs(), subOp.getRhs());
            subOp.getResult().replaceAllUsesWith(addOp.getResult());
            eraseOps.push_back(subOp.getOperation());
        }
    }

    for (Operation *op : llvm::reverse(eraseOps)) {
        if (op && op->use_empty())
            rewriter.eraseOp(op);
    }

    elementwiseOps.clear();
    for (Operation &op : genericOp.getBody()->getOperations()) {
        if (!op.hasTrait<OpTrait::IsTerminator>())
            elementwiseOps.push_back(&op);
    }
}

void convertToNpuOp(Operation *convOp,
                        linalg::GenericOp genericOp,
                        ElementwiseChain &elementwiseOps,
                        PatternRewriter &rewriter) {
    if (!convOp || !genericOp || elementwiseOps.empty())
        return;
    using namespace mlir::iree::compiler::Dialect::NPUFuseOp;

    int streamBlockArgIndex = -1;
    for (auto it : llvm::enumerate(genericOp.getDpsInputs())) {
        if (it.value() == convOp->getResult(0)) {
            streamBlockArgIndex = static_cast<int>(it.index());
            break;
        }
    }
    if (streamBlockArgIndex < 0) {
        for (auto it : llvm::enumerate(genericOp.getDpsInits())) {
            if (it.value() == convOp->getResult(0)) {
                streamBlockArgIndex =
                    static_cast<int>(genericOp.getNumDpsInputs() + it.index());
                break;
            }
        }
    }
    if (streamBlockArgIndex < 0)
        return;

    Value streamBlockArg =
        genericOp.getBody()->getArgument(streamBlockArgIndex);
    Value currentStream = streamBlockArg;

// 根据硬件特性从近匹配可融合的逐元素操作 -> npufuseop
// Convert near-matched fusible elementwise ops to NPU ops.
    enum class StageKind { PeA, PeB, Act };
    struct StageSlot {
        StageKind kind;
        int index;
    };

    llvm::SmallVector<StageSlot, 9> stageSlots = {
        {StageKind::PeA, 1}, {StageKind::PeB, 1}, {StageKind::Act, 1},
        {StageKind::PeA, 2}, {StageKind::PeB, 2}, {StageKind::Act, 2},
        {StageKind::PeA, 3}, {StageKind::PeB, 3}, {StageKind::Act, 3}};
    // Track the currently available fusion slot.
    size_t slotCursor = 0;
    bool startedPrefix = false;
    llvm::SmallVector<Operation *, 12> convertedOps;

    // Find the next slot that matches the required stage.
    auto consumeNextSlot = [&](StageKind kind, int &slotIndex) -> bool {
        for (size_t i = slotCursor; i < stageSlots.size(); ++i) {
            if (stageSlots[i].kind == kind) {
                slotIndex = stageSlots[i].index;
                slotCursor = i + 1;
                return true;
            }
        }
        return false;
    };

    auto canResolveSideOperand = [&](Value sideOperand) -> bool {
        if (!sideOperand)
            return false;
        if (mapBlockArgToGenericValue(genericOp, sideOperand))
            return true;

        Operation *rootDef = sideOperand.getDefiningOp();
        if (!rootDef || rootDef->getBlock() != genericOp.getBody())
            return false;

        unsigned numInputs = genericOp.getNumDpsInputs();
        llvm::DenseSet<Operation *> visited;
        llvm::DenseSet<Operation *> visiting;

        std::function<bool(Value)> collectFromValue = [&](Value v) -> bool {
            if (!v)
                return false;

            if (v == currentStream || v == streamBlockArg)
                return false;

            if (auto arg = v.dyn_cast<BlockArgument>()) {
                if (arg.getOwner() != genericOp.getBody())
                    return false;
                return arg.getArgNumber() < numInputs;
            }

            Operation *def = v.getDefiningOp();
            if (!def)
                return false;
            if (def->getBlock() != genericOp.getBody())
                return false;
            if (def->hasTrait<OpTrait::IsTerminator>())
                return false;

            if (visited.contains(def))
                return true;
            if (!visiting.insert(def).second)
                return false;

            for (Value operand : def->getOperands()) {
                if (!collectFromValue(operand)) {
                    visiting.erase(def);
                    return false;
                }
            }

            visiting.erase(def);
            visited.insert(def);
            return true;
        };

        return collectFromValue(sideOperand);
    };

    // Convert the maximal contiguous prefix connected to the conv stream.
    // Leading unrelated ops are preserved in generic.
    for (size_t i = 0; i < elementwiseOps.size();) {
        // 需要先匹配激活函数
        // Try to match activation patterns first.
        ActivationMatch activation;
        if (matchActivation(i, elementwiseOps, activation)) {
            if (activation.input != currentStream) {
                if (!startedPrefix) {
                    ++i;
                    continue;
                }
                break;
            }
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::Act, slotIndex))
                break;
            startedPrefix = true;
            rewriter.setInsertionPoint(activation.ops.back());
            auto loc = activation.ops.back()->getLoc();
            auto resultType = activation.output.getType();
            Operation *actOp = nullptr;
            if (activation.type == "relu")
                actOp = rewriter.create<ReluOp>(loc, resultType, activation.input);
            else if (activation.type == "leakyrelu")
                actOp = rewriter.create<LeakyReluOp>(loc, resultType, activation.input);
            else if (activation.type == "silu")
                actOp = rewriter.create<SiluOp>(loc, resultType, activation.input);
            else if (activation.type == "sigmoid")
                actOp = rewriter.create<SigmoidOp>(loc, resultType, activation.input);
            else if (activation.type == "gelu")
                actOp = rewriter.create<GeluOp>(loc, resultType, activation.input);
            else if (activation.type == "tanh")
                actOp = rewriter.create<TanhOp>(loc, resultType, activation.input);

            if (actOp) {
                activation.output.replaceAllUsesWith(actOp->getResult(0));
                currentStream = actOp->getResult(0);
                convertedOps.append(activation.ops.begin(), activation.ops.end());
            }
            i += activation.consumed;
            continue;
        }
        
        // mul : PE_a
        if (auto mulOp = dyn_cast<arith::MulFOp>(elementwiseOps[i])) {
            bool streamOnLhs = mulOp.getLhs() == currentStream;
            bool streamOnRhs = mulOp.getRhs() == currentStream;
            if (streamOnLhs == streamOnRhs) {
                if (!startedPrefix) {
                    ++i;
                    continue;
                }
                break;
            }
            Value sideOperand = streamOnLhs ? mulOp.getRhs() : mulOp.getLhs();
            if (!canResolveSideOperand(sideOperand)) {
                if (!startedPrefix) {
                    ++i;
                    continue;
                }
                break;
            }
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::PeA, slotIndex))
                break;
            startedPrefix = true;
            rewriter.setInsertionPoint(mulOp);
            auto loc = mulOp.getLoc();
            auto resultType = mulOp.getResult().getType();
            Operation *peOp = nullptr;
            if (slotIndex == 1)
                peOp = rewriter.create<PE1AOp>(loc, resultType, mulOp.getLhs(), mulOp.getRhs());
            else if (slotIndex == 2)
                peOp = rewriter.create<PE2AOp>(loc, resultType, mulOp.getLhs(), mulOp.getRhs());
            else
                peOp = rewriter.create<PE3AOp>(loc, resultType, mulOp.getLhs(), mulOp.getRhs());
            mulOp.getResult().replaceAllUsesWith(peOp->getResult(0));
            currentStream = peOp->getResult(0);
            convertedOps.push_back(mulOp.getOperation());
            ++i;
            continue;
        }

        // add : PE_b
        if (auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[i])) {
            bool streamOnLhs = addOp.getLhs() == currentStream;
            bool streamOnRhs = addOp.getRhs() == currentStream;
            if (streamOnLhs == streamOnRhs) {
                if (!startedPrefix) {
                    ++i;
                    continue;
                }
                break;
            }
            Value sideOperand = streamOnLhs ? addOp.getRhs() : addOp.getLhs();
            if (!canResolveSideOperand(sideOperand)) {
                if (!startedPrefix) {
                    ++i;
                    continue;
                }
                break;
            }
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::PeB, slotIndex))
                break;
            startedPrefix = true;
            rewriter.setInsertionPoint(addOp);
            auto loc = addOp.getLoc();
            auto resultType = addOp.getResult().getType();
            Operation *peOp = nullptr;
            if (slotIndex == 1)
                peOp = rewriter.create<PE1BOp>(loc, resultType, addOp.getLhs(), addOp.getRhs());
            else if (slotIndex == 2)
                peOp = rewriter.create<PE2BOp>(loc, resultType, addOp.getLhs(), addOp.getRhs());
            else
                peOp = rewriter.create<PE3BOp>(loc, resultType, addOp.getLhs(), addOp.getRhs());
            addOp.getResult().replaceAllUsesWith(peOp->getResult(0));
            currentStream = peOp->getResult(0);
            convertedOps.push_back(addOp.getOperation());
            ++i;
            continue;
        }

        if (!startedPrefix) {
            ++i;
            continue;
        }
        break;
    }

    // 清理已经被 npufuseop 结果替换后的旧逐元素算子
    // Erase old elementwise ops whose uses were replaced by NPU ops.
    for (Operation *op : llvm::reverse(convertedOps)) {
        if (op && op->use_empty())
            rewriter.eraseOp(op);
    }
}

// rewrite convOp -> fusedConvOp
Operation *rewriteWithFusedOp(Operation *convOp,
                              Operation *&fusedGenericOp,
                              const ElementwiseChain &elementwiseOps,
                              const FusionPatternInfo &pattern,
                              PatternRewriter &rewriter,
                              bool enableFastMath) {
    OpBuilder::InsertionGuard guard(rewriter);

    using namespace mlir::iree::compiler::Dialect::NPUFuseOp;

    auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                    : linalg::GenericOp();
    if (!genericOp)
        return nullptr;
    (void)elementwiseOps;
    (void)pattern;
    rewriter.setInsertionPoint(fusedGenericOp);

    bool hasNpuOps = false;
    if (genericOp) {
        for (Operation &op : genericOp.getBody()->getOperations()) {
            if (isa<ReluOp, LeakyReluOp, SiluOp, SigmoidOp, GeluOp, TanhOp,
                    PE1AOp, PE1BOp, PE2AOp, PE2BOp, PE3AOp, PE3BOp>(&op)) {
                hasNpuOps = true;
                break;
            }
        }
    }

    if (hasNpuOps) {
        Value input = convOp->getOperand(0);
        Value filter = convOp->getOperand(1);

        // conv操作结果在generic操作数的位置
        // Locate where the conv result appears in generic operands.
        int streamBlockArgIndex = -1;
        for (auto it : llvm::enumerate(genericOp.getDpsInputs())) {
            if (it.value() == convOp->getResult(0)) {
                streamBlockArgIndex = static_cast<int>(it.index());
                break;
            }
        }
        if (streamBlockArgIndex < 0) {
            for (auto it : llvm::enumerate(genericOp.getDpsInits())) {
                if (it.value() == convOp->getResult(0)) {
                    streamBlockArgIndex =
                        static_cast<int>(genericOp.getNumDpsInputs() + it.index());
                    break;
                }
            }
        }
        if (streamBlockArgIndex < 0) {
            convOp->emitOpError(
                "expected conv result to be used by linalg.generic as a dps "
                "input or output init, but it was not found in dps "
                "inputs/inits; refusing fusion to avoid leaving unfused npufuseop "
                "ops for later LLVM lowering");
            return nullptr;
        }

        // Current stream value represented as a generic block argument.
        Value streamBlockArg =
            genericOp.getBody()->getArgument(streamBlockArgIndex);
        Value currentStream = streamBlockArg;
        Value pe1_a;
        Value pe1_b;
        Value pe2_a;
        Value pe2_b;
        Value pe3_a;
        Value pe3_b;

        SmallVector<ActivationInfo, 3> activationInfo;
        SmallVector<Operation *, 8> consumedOps;
        bool startedFusionPrefix = false;
        bool reachedResidualOps = false;

        // 激活位置，数字指前面出现的槽位顺序，所以开始是2
        // Activation position is keyed by prior slot order, so it starts with 2
        int activationAfter = 2;

        // Resolve additional side operands needed by fused conv.
        auto materializeSideTensorFromInternalChain =
            [&](Value sideValue) -> Value {
            if (!sideValue)
                return Value();

            // Fast-path: side value already maps to a generic operand.
            if (Value mapped = mapBlockArgToGenericValue(genericOp, sideValue))
                return mapped;

            Operation *rootDef = sideValue.getDefiningOp();
            if (!rootDef || rootDef->getBlock() != genericOp.getBody())
                return Value();

            unsigned numInputs = genericOp.getNumDpsInputs();
            llvm::SmallVector<unsigned, 8> requiredInputIndices;
            llvm::DenseSet<unsigned> requiredInputSet;
            llvm::SmallVector<Operation *, 12> topoOps;
            llvm::DenseSet<Operation *> visited;
            llvm::DenseSet<Operation *> visiting;

            std::function<bool(Value)> collectFromValue = [&](Value v) -> bool {
                if (!v)
                    return false;

                // Side tensor must not depend on the main stream value.
                if (v == currentStream || v == streamBlockArg)
                    return false;

                if (auto arg = v.dyn_cast<BlockArgument>()) {
                    if (arg.getOwner() != genericOp.getBody())
                        return false;
                    unsigned idx = arg.getArgNumber();
                    if (idx >= numInputs)
                        return false;
                    if (requiredInputSet.insert(idx).second)
                        requiredInputIndices.push_back(idx);
                    return true;
                }

                Operation *def = v.getDefiningOp();
                if (!def)
                    return false;
                if (def->getBlock() != genericOp.getBody())
                    return false;
                if (def->hasTrait<OpTrait::IsTerminator>())
                    return false;

                if (visited.contains(def))
                    return true;
                if (!visiting.insert(def).second)
                    return false;

                for (Value operand : def->getOperands()) {
                    if (!collectFromValue(operand)) {
                        visiting.erase(def);
                        return false;
                    }
                }

                visiting.erase(def);
                visited.insert(def);
                topoOps.push_back(def);
                return true;
            };

            if (!collectFromValue(sideValue) || topoOps.empty())
                return Value();

            if (requiredInputIndices.empty())
                return Value();

            auto streamType = mapBlockArgToGenericValue(genericOp, streamBlockArg)
                                  .getType()
                                  .dyn_cast<RankedTensorType>();
            if (!streamType)
                return Value();

            llvm::sort(requiredInputIndices);

            SmallVector<Value, 8> ins;
            SmallVector<AffineMap, 8> maps;
            SmallVector<Value, 1> outs;
            SmallVector<AffineMap, 1> outMaps;
            ins.reserve(requiredInputIndices.size());
            maps.reserve(requiredInputIndices.size());

            auto indexingMaps = genericOp.getIndexingMapsArray();
            for (unsigned idx : requiredInputIndices) {
                ins.push_back(getGenericInputValue(genericOp, idx));
                maps.push_back(indexingMaps[idx]);
            }

            auto outTensor = rewriter.create<tensor::EmptyOp>(
                convOp->getLoc(), streamType.getShape(),
                streamType.getElementType());
            outs.push_back(outTensor);

            // Use conv-stream indexing map for produced side tensor shape.
            maps.push_back(indexingMaps[streamBlockArgIndex]);

            int64_t loopRank = maps.back().getNumDims();
            size_t iteratorCount =
                loopRank > 0 ? static_cast<size_t>(loopRank) : 0;
            SmallVector<utils::IteratorType, 8> iterators(
                iteratorCount,
                utils::IteratorType::parallel);

            OpBuilder::InsertionGuard sideInsertionGuard(rewriter);
            rewriter.setInsertionPoint(genericOp);
            auto sideGeneric = rewriter.create<linalg::GenericOp>(
                convOp->getLoc(),
                TypeRange{outTensor.getResult().getType()},
                ins, outs, maps, iterators,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                    IRMapping mapper;
                    for (auto it : llvm::enumerate(requiredInputIndices)) {
                        mapper.map(genericOp.getBody()->getArgument(it.value()),
                                   args[it.index()]);
                    }

                    for (Operation *op : topoOps)
                        b.clone(*op, mapper);

                    Value yielded = mapper.lookupOrDefault(sideValue);
                    b.create<linalg::YieldOp>(loc, yielded);
                });

            return sideGeneric->getResult(0);
        };

        auto resolveSideOperand = [&](Value lhs, Value rhs,
                                      Value &sideTensor) -> bool {
            Value sideValue;
            if (lhs == currentStream && rhs != currentStream) {
                sideValue = rhs;
            } else if (rhs == currentStream && lhs != currentStream) {
                sideValue = lhs;
            } else {
                return false;
            }
            sideTensor = mapBlockArgToGenericValue(genericOp, sideValue);
            if (!sideTensor)
                sideTensor = materializeSideTensorFromInternalChain(sideValue);
            return static_cast<bool>(sideTensor);
        };

        for (Operation *chainOp : elementwiseOps) {
            if (!chainOp)
                continue;

            if (reachedResidualOps)
                break;

            if (auto peOp = dyn_cast<PE1AOp>(chainOp)) {
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                if (pe1_a)
                    reachedResidualOps = true;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe1_a))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                activationAfter = 2;
                // after 2：PE1    after 5：PE2    after 8：PE3
                currentStream = peOp.getResult();
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto peOp = dyn_cast<PE1BOp>(chainOp)) {
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                if (pe1_b)
                    reachedResidualOps = true;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe1_b))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                activationAfter = 2;
                currentStream = peOp.getResult();
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto peOp = dyn_cast<PE2AOp>(chainOp)) {
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                if (pe2_a)
                    reachedResidualOps = true;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe2_a))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                activationAfter = 5;
                currentStream = peOp.getResult();
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto peOp = dyn_cast<PE2BOp>(chainOp)) {
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                if (pe2_b)
                    reachedResidualOps = true;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe2_b))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                activationAfter = 5;
                currentStream = peOp.getResult();
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto peOp = dyn_cast<PE3AOp>(chainOp)) {
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                if (pe3_a)
                    reachedResidualOps = true;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe3_a))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                activationAfter = 8;
                currentStream = peOp.getResult();
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto peOp = dyn_cast<PE3BOp>(chainOp)) {
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                if (pe3_b)
                    reachedResidualOps = true;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe3_b))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                activationAfter = 8;
                currentStream = peOp.getResult();
                consumedOps.push_back(chainOp);
                continue;
            }

            auto handleActivation = [&](StringRef type, Value input,
                                        Value output) -> bool {
                if (input != currentStream)
                    return false;
                if (!startedFusionPrefix)
                    startedFusionPrefix = true;
                ActivationInfo info;
                info.type = type.str();
                info.afterOpIndex = activationAfter;
                activationInfo.push_back(std::move(info));
                currentStream = output;
                return true;
            };

            if (auto act = dyn_cast<ReluOp>(chainOp)) {
                if (!handleActivation("relu", act.getInput(), act.getOutput()))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto act = dyn_cast<LeakyReluOp>(chainOp)) {
                if (!handleActivation("leakyrelu", act.getInput(), act.getOutput()))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto act = dyn_cast<SiluOp>(chainOp)) {
                if (!handleActivation("silu", act.getInput(), act.getOutput()))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto act = dyn_cast<SigmoidOp>(chainOp)) {
                if (!handleActivation("sigmoid", act.getInput(), act.getOutput()))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto act = dyn_cast<GeluOp>(chainOp)) {
                if (!handleActivation("gelu", act.getInput(), act.getOutput()))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                consumedOps.push_back(chainOp);
                continue;
            }
            if (auto act = dyn_cast<TanhOp>(chainOp)) {
                if (!handleActivation("tanh", act.getInput(), act.getOutput()))
                    reachedResidualOps = true;
                if (reachedResidualOps)
                    continue;
                consumedOps.push_back(chainOp);
                continue;
            }

            if (startedFusionPrefix)
                reachedResidualOps = true;
        }

        if (consumedOps.empty()) {
            return nullptr;
        }

        // new fused conv op
        auto fused = rewriter.create<Conv2DOp>(
            convOp->getLoc(), convOp->getResult(0).getType(), input, filter,
            pe1_a, pe1_b, pe2_a, pe2_b, pe3_a, pe3_b);

        for (NamedAttribute attr : convOp->getAttrs())
            fused->setAttr(attr.getName(), attr.getValue());

        if (pe1_a || pe1_b || pe2_a || pe2_b || pe3_a || pe3_b) {
            SmallVector<Attribute, 6> fusedOpInfoAttrs;
            int64_t operandIndex = 2;
            auto appendRole = [&](Value v, StringRef role) {
                if (!v)
                    return;
                SmallVector<NamedAttribute, 2> dict;
                dict.emplace_back(rewriter.getStringAttr("operand_index"),
                                  rewriter.getI64IntegerAttr(operandIndex));
                dict.emplace_back(rewriter.getStringAttr("role"),
                                  rewriter.getStringAttr(role));
                fusedOpInfoAttrs.push_back(rewriter.getDictionaryAttr(dict));
                ++operandIndex;
            };
            appendRole(pe1_a, "PE1_a");
            appendRole(pe1_b, "PE1_b");
            appendRole(pe2_a, "PE2_a");
            appendRole(pe2_b, "PE2_b");
            appendRole(pe3_a, "PE3_a");
            appendRole(pe3_b, "PE3_b");
            fused->setAttr("fused_op_info", rewriter.getArrayAttr(fusedOpInfoAttrs));
        }

        if (!activationInfo.empty()) {
            SmallVector<Attribute, 6> activationAttrs;
            activationAttrs.reserve(activationInfo.size());
            for (const ActivationInfo &activation : activationInfo) {
                SmallVector<NamedAttribute, 2> dict;
                dict.emplace_back(rewriter.getStringAttr("type"),
                                  rewriter.getStringAttr(activation.type));
                dict.emplace_back(rewriter.getStringAttr("after"),
                                  rewriter.getI64IntegerAttr(activation.afterOpIndex));
                activationAttrs.push_back(rewriter.getDictionaryAttr(dict));
            }
            fused->setAttr("npu.activation", rewriter.getArrayAttr(activationAttrs));
        }

        if (enableFastMath)
            fused->setAttr("npu.fastmath", rewriter.getBoolAttr(true));

        // 将 generic 的 conv 输入流重定向为 fused conv 输出。
        // Redirect generic conv input stream to fused conv output.
        genericOp->setOperand(streamBlockArgIndex, fused->getResult(0));

        // 保留剩余elementwise，仅移除已经融合进 conv 的 npufuseop
        // Keep residual elementwise ops and remove only consumed npufuseops.
        if (currentStream != streamBlockArg)
            currentStream.replaceAllUsesWith(streamBlockArg);

        linalg::GenericOp updatedGeneric;
        int forcePreserveInputIndex =
            streamBlockArgIndex < static_cast<int>(genericOp.getNumDpsInputs())
                ? streamBlockArgIndex
                : -1;
        if (!cleanupResidualGenericAfterFusion(genericOp, consumedOps, rewriter,
                                               updatedGeneric,
                                               forcePreserveInputIndex))
            return nullptr;
        fusedGenericOp = updatedGeneric ? updatedGeneric.getOperation()
                                        : fusedGenericOp;

        return fused.getOperation();
    }

    // Skip fusion when no converted npufuseop/activation prefix exists.
    return nullptr;
}

// 将链尾结果的用户重定向为新融合操作的结果
// Redirect users of the chain tail to the fused result.
void redirectFusedChain(Operation *convOp,
                        Operation *fusedGenericOp,
                        Operation *newOp,
                        const FusionPatternInfo &pattern,
                        const ElementwiseChain &elementwiseOps) {
    (void)pattern;
    (void)elementwiseOps;
    auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                    : linalg::GenericOp();

    // If residual body ops remain, generic must stay as result carrier.
    if (genericOp && hasNonTerminatorBodyOps(genericOp))
        return;

    Operation *tailOp = fusedGenericOp ? fusedGenericOp : convOp;
    tailOp->getResult(0).replaceAllUsesWith(newOp->getResult(0));
}

void eraseFusedOps(Operation *convOp,
                   Operation *fusedGenericOp,
                   PatternRewriter &rewriter,
                   const FusionPatternInfo &pattern,
                   const ElementwiseChain &elementwiseOps) {
    (void)pattern;
    (void)elementwiseOps;

    // If the post-conv generic body is empty, it is no longer needed.
    if (auto genericOp =
            fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                           : linalg::GenericOp()) {
        if (!hasNonTerminatorBodyOps(genericOp) && genericOp->use_empty())
            rewriter.eraseOp(genericOp);
    }

    // Remove the original conv once all users were redirected.
    if (convOp && convOp->use_empty())
        rewriter.eraseOp(convOp);
}

// generic中其它操作
// Helper checks for other ops remaining in generic.
static bool hasNonTerminatorBodyOps(linalg::GenericOp genericOp) {
    if (!genericOp)
        return false;
    for (Operation &op : genericOp.getBody()->getOperations()) {
        if (!op.hasTrait<OpTrait::IsTerminator>())
            return true;
    }
    return false;
}

static bool genericUsesOperandValue(linalg::GenericOp genericOp, Value v) {
    if (!genericOp || !v)
        return false;
    for (Value operand : genericOp->getOperands()) {
        if (operand == v)
            return true;
    }
    return false;
}

// Drop fused ops from generic and prune unused dps input arguments.
static bool cleanupResidualGenericAfterFusion(
    linalg::GenericOp genericOp, ArrayRef<Operation *> consumedOps,
    PatternRewriter &rewriter, linalg::GenericOp &updatedGenericOp,
    int forcePreserveInputIndex) {
    updatedGenericOp = genericOp;
    if (!genericOp || consumedOps.empty())
        return true;

    llvm::DenseSet<Operation *> consumedSet(consumedOps.begin(),
                                            consumedOps.end());
    Block *body = genericOp.getBody();

    llvm::SmallVector<Operation *, 12> keptOps;
    keptOps.reserve(body->getOperations().size());
    for (Operation &op : body->getOperations()) {
        if (op.hasTrait<OpTrait::IsTerminator>())
            continue;
        if (!consumedSet.contains(&op))
            keptOps.push_back(&op);
    }

    // Keep only ops that are live from generic yield; dead side-chain ops
    // (previously used only by consumed PE ops) are pruned here.
    llvm::DenseSet<Operation *> keptSet(keptOps.begin(), keptOps.end());
    llvm::DenseSet<Operation *> liveSet;
    std::function<void(Value)> markLiveFromValue = [&](Value v) {
        Operation *def = v.getDefiningOp();
        if (!def || !keptSet.contains(def) || liveSet.contains(def))
            return;
        liveSet.insert(def);
        for (Value operand : def->getOperands())
            markLiveFromValue(operand);
    };

    auto *yieldOp = cast<linalg::YieldOp>(body->getTerminator()).getOperation();
    for (Value yielded : yieldOp->getOperands())
        markLiveFromValue(yielded);

    llvm::SmallVector<Operation *, 12> deadOps;
    deadOps.reserve(keptOps.size());
    llvm::SmallVector<Operation *, 12> liveKeptOps;
    liveKeptOps.reserve(keptOps.size());
    for (Operation *op : keptOps) {
        if (liveSet.contains(op))
            liveKeptOps.push_back(op);
        else
            deadOps.push_back(op);
    }
    keptOps.swap(liveKeptOps);

    auto operandFromConsumedSet = [&](Value v) {
        Operation *def = v.getDefiningOp();
        return def && consumedSet.contains(def);
    };

    for (Operation *kept : keptOps) {
        for (Value operand : kept->getOperands()) {
            if (operandFromConsumedSet(operand)) {
                genericOp.emitOpError(
                    "residual generic op still depends on fused op result");
                return false;
            }
        }
    }
    for (Value yielded : yieldOp->getOperands()) {
        if (operandFromConsumedSet(yielded)) {
            genericOp.emitOpError(
                "generic yield still depends on fused op result");
            return false;
        }
    }

    auto eraseDeadAndConsumed = [&]() {
        llvm::SmallVector<Operation *, 16> opsInBlock;
        for (Operation &op : llvm::reverse(body->getOperations())) {
            opsInBlock.push_back(&op);
        }
        llvm::SmallDenseSet<Operation *, 16> eraseSet;
        eraseSet.insert(consumedOps.begin(), consumedOps.end());
        for (Operation *dead : deadOps) {
            if (dead)
                eraseSet.insert(dead);
        }
        for (Operation *op : opsInBlock) {
            if (eraseSet.contains(op) && op->use_empty()) {
                rewriter.eraseOp(op);
            }
        }
    };

    if (keptOps.empty()) {
        eraseDeadAndConsumed();
        return true;
    }

    unsigned numInputs = genericOp.getNumDpsInputs();
    unsigned numInits = genericOp.getNumDpsInits();
    llvm::SmallDenseSet<unsigned, 8> usedInputIndices;

    auto recordUsedBlockArg = [&](Value v) {
        auto arg = v.dyn_cast<BlockArgument>();
        if (!arg || arg.getOwner() != body)
            return;
        unsigned argNum = arg.getArgNumber();
        if (argNum < numInputs)
            usedInputIndices.insert(argNum);
    };

    for (Operation *kept : keptOps)
        for (Value operand : kept->getOperands())
            recordUsedBlockArg(operand);
    for (Value yielded : yieldOp->getOperands())
        recordUsedBlockArg(yielded);

    // Keep the stream input that was rewired to the new fused conv result,
    // so the new conv inherits dataflow before old ops are erased.
    if (forcePreserveInputIndex >= 0 &&
        forcePreserveInputIndex < static_cast<int>(numInputs)) {
        usedInputIndices.insert(static_cast<unsigned>(forcePreserveInputIndex));
    }

    if (usedInputIndices.size() == numInputs) {
        eraseDeadAndConsumed();
        return true;
    }

    llvm::SmallVector<unsigned, 8> keptInputIndices;
    keptInputIndices.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; ++i) {
        if (usedInputIndices.contains(i))
            keptInputIndices.push_back(i);
    }

    auto indexingMaps = genericOp.getIndexingMapsArray();
    llvm::SmallVector<AffineMap, 8> newMaps;
    llvm::SmallVector<Value, 8> newInputs;
    newMaps.reserve(keptInputIndices.size() + numInits);
    newInputs.reserve(keptInputIndices.size());
    for (unsigned idx : keptInputIndices) {
        newInputs.push_back(genericOp.getDpsInputs()[idx]);
        newMaps.push_back(indexingMaps[idx]);
    }

    llvm::SmallVector<Value, 4> newInits(genericOp.getDpsInits().begin(),
                                         genericOp.getDpsInits().end());
    for (unsigned i = 0; i < numInits; ++i)
        newMaps.push_back(indexingMaps[numInputs + i]);

    llvm::SmallVector<int, 12> oldArgToNew(numInputs + numInits, -1);
    for (auto it : llvm::enumerate(keptInputIndices))
        oldArgToNew[it.value()] = static_cast<int>(it.index());
    for (unsigned i = 0; i < numInits; ++i)
        oldArgToNew[numInputs + i] =
            static_cast<int>(keptInputIndices.size() + i);

    rewriter.setInsertionPoint(genericOp);
    auto newGeneric = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), genericOp.getResultTypes(), newInputs, newInits,
        newMaps, genericOp.getIteratorTypesArray(),
        [&](OpBuilder &b, Location loc, ValueRange args) {
            IRMapping mapper;
            for (unsigned oldArg = 0; oldArg < oldArgToNew.size(); ++oldArg) {
                int mapped = oldArgToNew[oldArg];
                if (mapped < 0)
                    continue;
                mapper.map(body->getArgument(oldArg), args[mapped]);
            }

            for (Operation *kept : keptOps)
                b.clone(*kept, mapper);

            auto oldYield = cast<linalg::YieldOp>(body->getTerminator());
            llvm::SmallVector<Value, 4> newYieldedValues;
            newYieldedValues.reserve(oldYield.getNumOperands());
            for (Value yielded : oldYield.getValues()) {
                Value mapped = mapper.lookupOrDefault(yielded);
                newYieldedValues.push_back(mapped);
            }
            b.create<linalg::YieldOp>(loc, newYieldedValues);
        });

    genericOp->replaceAllUsesWith(newGeneric->getResults());
    rewriter.eraseOp(genericOp);
    updatedGenericOp = newGeneric;
    return true;
}

static Operation *createFusedOpFromPattern(StringRef fusedName,
                                           Operation *convOp,
                                           const FusionPatternInfo &pattern,
                                           ArrayRef<Value> operands,
                                           PatternRewriter &rewriter) {
    Location loc = convOp->getLoc();

    // 通用创建方式
    // Generic op construction path.
    OperationState state(loc, fusedName);
    state.addOperands(operands);
    state.addTypes(convOp->getResultTypes());
    return rewriter.create(state);
}

////////////////////////////////////////////////////////////////////
//                       Folding logic                           //
///////////////////////////////////////////////////////////////////
static bool combineAddSubPair(linalg::GenericOp genericOp,
                              Operation *firstOp,
                              Operation *secondOp,
                              PatternRewriter &rewriter,
                              FusionPatternInfo &pattern,
                              Value &newResult,
                              bool enableFastMath) {
    (void)enableFastMath;
    (void)pattern;
    // Require two directly chained ops.
    if (!firstOp->hasOneUse())
        return false;
    if (*firstOp->getResult(0).getUsers().begin() != secondOp)
        return false;

    auto addOp = dyn_cast<arith::AddFOp>(firstOp);
    if (!addOp)
        return false;

    Value secondSideArg;
    BinaryKind combineKind = BinaryKind::Add;
    if (auto secondAdd = dyn_cast<arith::AddFOp>(secondOp)) {
        if (secondAdd.getLhs() == addOp.getResult()) {
            secondSideArg = secondAdd.getRhs();
            combineKind = BinaryKind::Add;
        } else if (secondAdd.getRhs() == addOp.getResult()) {
            secondSideArg = secondAdd.getLhs();
            combineKind = BinaryKind::Add;
        } else {
            return false;
        }
    } else if (auto secondSub = dyn_cast<arith::SubFOp>(secondOp)) {
        if (secondSub.getLhs() != addOp.getResult())
            return false;
        secondSideArg = secondSub.getRhs();
        combineKind = BinaryKind::Sub;
    } else {
        return false;
    }

    // Both add/sub operands must be generic block arguments.
    int biasIndex = -1;
    int secondSideIndex = -1;
    if (!getGenericInputOperandIndex(genericOp, secondSideArg, secondSideIndex))
        return false;

    if (addOp.getLhs() != secondSideArg &&
        getGenericInputOperandIndex(genericOp, addOp.getLhs(), biasIndex)) {
        // Prefer lhs when it is an input block arg.
    } else if (addOp.getRhs() != secondSideArg &&
               getGenericInputOperandIndex(genericOp, addOp.getRhs(),
                                           biasIndex)) {
        // Fallback to rhs when lhs is not a valid input block arg.
    } else {
        return false;
    }

    if (biasIndex < 0 || secondSideIndex < 0)
        return false;

    // Resolve the corresponding input tensors.
    Value biasTensor = getGenericInputValue(genericOp, biasIndex);
    Value secondSideTensor = getGenericInputValue(genericOp, secondSideIndex);
    if (!biasTensor || !secondSideTensor)
        return false;
    if (biasTensor.getType() != secondSideTensor.getType())
        return false;

    rewriter.setInsertionPoint(genericOp);
    Value newBias = createElementwiseBinaryTensorOp(
        rewriter, genericOp.getLoc(), biasTensor, secondSideTensor, combineKind);
    if (!newBias)
        return false;
    auto secondSideType = secondSideTensor.getType().cast<ShapedType>();
    Value zeroSecondSide = createSplatTensor(rewriter, genericOp.getLoc(),
                                             secondSideType, secondSideTensor,
                                             0.0);
    if (!zeroSecondSide)
        return false;

    genericOp.setOperand(biasIndex, newBias);
    genericOp.setOperand(secondSideIndex, zeroSecondSide);

    secondOp->getResult(0).replaceAllUsesWith(addOp.getResult());
    newResult = addOp.getResult();
    return true;
}

static bool combineMulPair(linalg::GenericOp genericOp,
                           Operation *firstOp,
                           Operation *secondOp,
                           PatternRewriter &rewriter,
                           FusionPatternInfo &pattern,
                           Value &newResult,
                           bool enableFastMath) {
    (void)enableFastMath;
    (void)pattern;
    if (!firstOp->hasOneUse())
        return false;
    if (*firstOp->getResult(0).getUsers().begin() != secondOp)
        return false;

    auto firstMul = dyn_cast<arith::MulFOp>(firstOp);
    auto secondMul = dyn_cast<arith::MulFOp>(secondOp);
    if (!firstMul || !secondMul)
        return false;
    if (secondMul.getLhs() != firstMul.getResult() &&
        secondMul.getRhs() != firstMul.getResult())
        return false;

    int varianceIndex = -1;
    int scaleIndex = -1;
    if (!getGenericInputOperandIndex(genericOp, firstMul.getLhs(),
                                     varianceIndex))
        getGenericInputOperandIndex(genericOp, firstMul.getRhs(),
                                    varianceIndex);
    Value scaleArg = secondMul.getLhs() == firstMul.getResult()
                         ? secondMul.getRhs()
                         : secondMul.getLhs();
    if (!getGenericInputOperandIndex(genericOp, scaleArg, scaleIndex))
        return false;
    if (varianceIndex < 0 || scaleIndex < 0)
        return false;

    Value varianceTensor = getGenericInputValue(genericOp, varianceIndex);
    Value scaleTensor = getGenericInputValue(genericOp, scaleIndex);
    if (!varianceTensor || !scaleTensor)
        return false;
    if (varianceTensor.getType() != scaleTensor.getType())
        return false;

    rewriter.setInsertionPoint(genericOp);
    Value newScale = createElementwiseBinaryTensorOp(
        rewriter, genericOp.getLoc(), varianceTensor, scaleTensor,
        BinaryKind::Mul);
    if (!newScale)
        return false;
    auto scaleType = scaleTensor.getType().cast<ShapedType>();
    Value ones = createSplatTensor(rewriter, genericOp.getLoc(), scaleType,
                                   scaleTensor, 1.0);
    if (!ones)
        return false;

    genericOp.setOperand(varianceIndex, newScale);
    genericOp.setOperand(scaleIndex, ones);

    secondMul.getResult().replaceAllUsesWith(firstMul.getResult());
    newResult = firstMul.getResult();
    return true;
}

////////////////////////////////////////////////////////////////////
//                        其它辅助函数实现                          //
//                     Other helper implementations                 //
////////////////////////////////////////////////////////////////////
// 尝试匹配常见的激活函数模式（如ReLU、LeakyReLU、Tanh），并将匹配结果记录在match中
// Match common activation patterns and store results in `match`.
static bool matchActivation(size_t index,
                              ArrayRef<Operation *> elementwiseOps,
                              ActivationMatch &match) {

    if (index + 1 < elementwiseOps.size()) {
        auto cmpOp = dyn_cast<arith::CmpFOp>(elementwiseOps[index]);
        auto selectOp = dyn_cast<arith::SelectOp>(elementwiseOps[index + 1]);
        if (cmpOp && selectOp) {
            bool cmpInputOnLhs = isZeroConstant(cmpOp.getRhs());
            bool cmpInputOnRhs = isZeroConstant(cmpOp.getLhs());
            if (cmpInputOnLhs || cmpInputOnRhs) {
                Value inputValue = cmpInputOnLhs ? cmpOp.getLhs() : cmpOp.getRhs();
                using Pred = arith::CmpFPredicate;
                bool matched = false;
                if (cmpInputOnLhs) {
                    matched = (cmpOp.getPredicate() == Pred::OGT ||
                               cmpOp.getPredicate() == Pred::UGT) &&
                              selectOp.getCondition() == cmpOp.getResult() &&
                              selectOp.getTrueValue() == inputValue &&
                              isZeroConstant(selectOp.getFalseValue());
                } else {
                    matched = (cmpOp.getPredicate() == Pred::OLT ||
                               cmpOp.getPredicate() == Pred::ULT) &&
                              selectOp.getCondition() == cmpOp.getResult() &&
                              isZeroConstant(selectOp.getTrueValue()) &&
                              selectOp.getFalseValue() == inputValue;
                }

                if (matched) {
                    match.type = "relu";
                    match.input = inputValue;
                    match.output = selectOp.getResult();
                    match.consumed = 2;
                    match.ops = {cmpOp.getOperation(), selectOp.getOperation()};
                    return true;
                }
            }
        }
    }

    if (index + 5 < elementwiseOps.size()) {
        auto cmpOp0 = dyn_cast<arith::CmpFOp>(elementwiseOps[index]);
        auto selectOp0 = dyn_cast<arith::SelectOp>(elementwiseOps[index + 1]);
        auto mulOp = dyn_cast<arith::MulFOp>(elementwiseOps[index + 2]);
        auto cmpOp1 = dyn_cast<arith::CmpFOp>(elementwiseOps[index + 3]);
        auto selectOp1 = dyn_cast<arith::SelectOp>(elementwiseOps[index + 4]);
        auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[index + 5]);

        if (cmpOp0 && selectOp0 && mulOp && cmpOp1 && selectOp1 && addOp) {
            using Pred = arith::CmpFPredicate;
            bool cmp0IsMin = cmpOp0.getPredicate() == Pred::OLT ||
                             cmpOp0.getPredicate() == Pred::ULT;
            bool cmp1IsMax = cmpOp1.getPredicate() == Pred::OGT ||
                             cmpOp1.getPredicate() == Pred::UGT;
            Value lhs0 = cmpOp0.getLhs();
            Value rhs0 = cmpOp0.getRhs();
            Value lhs1 = cmpOp1.getLhs();
            Value rhs1 = cmpOp1.getRhs();
            bool sameInputs = (lhs0 == lhs1 && rhs0 == rhs1) ||
                              (lhs0 == rhs1 && rhs0 == lhs1);
            bool select0IsMin = selectOp0.getCondition() == cmpOp0.getResult() &&
                                selectOp0.getTrueValue() == lhs0 &&
                                selectOp0.getFalseValue() == rhs0;
            bool select1IsMax = selectOp1.getCondition() == cmpOp1.getResult() &&
                                selectOp1.getTrueValue() == lhs1 &&
                                selectOp1.getFalseValue() == rhs1;
            bool mulIsMinScaled =
                (mulOp.getLhs() == selectOp0.getResult() &&
                 isa<arith::ConstantOp>(mulOp.getRhs().getDefiningOp())) ||
                (mulOp.getRhs() == selectOp0.getResult() &&
                 isa<arith::ConstantOp>(mulOp.getLhs().getDefiningOp()));
            bool addUsesMax = (addOp.getLhs() == selectOp1.getResult() &&
                               addOp.getRhs() == mulOp.getResult()) ||
                              (addOp.getRhs() == selectOp1.getResult() &&
                               addOp.getLhs() == mulOp.getResult());

            if (cmp0IsMin && cmp1IsMax && sameInputs && select0IsMin &&
                select1IsMax && mulIsMinScaled && addUsesMax) {
                Value inputValue = chooseActivationInput(
                    lhs0, rhs0, elementwiseOps, llvm::DenseMap<Value, Value>());
                match.type = "leakyrelu";
                match.input = inputValue;
                match.output = addOp.getResult();
                match.consumed = 6;
                match.ops = {cmpOp0.getOperation(), selectOp0.getOperation(),
                             mulOp.getOperation(), cmpOp1.getOperation(),
                             selectOp1.getOperation(), addOp.getOperation()};
                return true;
            }
        }
    }

    if (index + 4 < elementwiseOps.size()) {
        auto negOp = dyn_cast<arith::NegFOp>(elementwiseOps[index]);
        auto expOp = dyn_cast<math::ExpOp>(elementwiseOps[index + 1]);
        auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[index + 2]);
        auto divOp = dyn_cast<arith::DivFOp>(elementwiseOps[index + 3]);
        auto mulOp = dyn_cast<arith::MulFOp>(elementwiseOps[index + 4]);
        if (negOp && expOp && addOp && divOp && mulOp) {
            bool expUsesNeg = expOp.getOperand() == negOp.getResult();
            bool addUsesExp = (addOp.getLhs() == expOp.getResult() &&
                               isFloatConstant(addOp.getRhs(), 1.0)) ||
                              (addOp.getRhs() == expOp.getResult() &&
                               isFloatConstant(addOp.getLhs(), 1.0));
            bool divIsSigmoid = (divOp.getLhs() != divOp.getRhs()) &&
                                isFloatConstant(divOp.getLhs(), 1.0) &&
                                divOp.getRhs() == addOp.getResult();
            bool mulUsesSigmoid = (mulOp.getLhs() == divOp.getResult() &&
                                   mulOp.getRhs() == negOp.getOperand()) ||
                                  (mulOp.getRhs() == divOp.getResult() &&
                                   mulOp.getLhs() == negOp.getOperand());
            if (expUsesNeg && addUsesExp && divIsSigmoid && mulUsesSigmoid) {
                match.type = "silu";
                match.input = negOp.getOperand();
                match.output = mulOp.getResult();
                match.consumed = 5;
                match.ops = {negOp.getOperation(), expOp.getOperation(),
                             addOp.getOperation(), divOp.getOperation(),
                             mulOp.getOperation()};
                return true;
            }
        }
    }

    if (index + 3 < elementwiseOps.size()) {
        auto negOp = dyn_cast<arith::NegFOp>(elementwiseOps[index]);
        auto expOp = dyn_cast<math::ExpOp>(elementwiseOps[index + 1]);
        auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[index + 2]);
        auto divOp = dyn_cast<arith::DivFOp>(elementwiseOps[index + 3]);
        if (negOp && expOp && addOp && divOp) {
            bool expUsesNeg = expOp.getOperand() == negOp.getResult();
            bool addUsesExp = (addOp.getLhs() == expOp.getResult() &&
                               isFloatConstant(addOp.getRhs(), 1.0)) ||
                              (addOp.getRhs() == expOp.getResult() &&
                               isFloatConstant(addOp.getLhs(), 1.0));
            bool divIsSigmoid = isFloatConstant(divOp.getLhs(), 1.0) &&
                                divOp.getRhs() == addOp.getResult();
            if (expUsesNeg && addUsesExp && divIsSigmoid) {
                match.type = "sigmoid";
                match.input = negOp.getOperand();
                match.output = divOp.getResult();
                match.consumed = 4;
                match.ops = {negOp.getOperation(), expOp.getOperation(),
                             addOp.getOperation(), divOp.getOperation()};
                return true;
            }
        }
    }

    if (index + 4 < elementwiseOps.size()) {
        auto divOp = dyn_cast<arith::DivFOp>(elementwiseOps[index]);
        auto erfOp = dyn_cast<math::ErfOp>(elementwiseOps[index + 1]);
        auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[index + 2]);
        auto mulOp0 = dyn_cast<arith::MulFOp>(elementwiseOps[index + 3]);
        auto mulOp1 = dyn_cast<arith::MulFOp>(elementwiseOps[index + 4]);
        if (divOp && erfOp && addOp && mulOp0 && mulOp1) {
            bool erfUsesDiv = erfOp.getOperand() == divOp.getResult();
            bool addUsesErf = (addOp.getLhs() == erfOp.getResult() &&
                               isFloatConstant(addOp.getRhs(), 1.0)) ||
                              (addOp.getRhs() == erfOp.getResult() &&
                               isFloatConstant(addOp.getLhs(), 1.0));
            bool mul0UsesAdd = (mulOp0.getLhs() == addOp.getResult() ||
                                mulOp0.getRhs() == addOp.getResult());
            bool mul1UsesMul0 = (mulOp1.getLhs() == mulOp0.getResult() ||
                                 mulOp1.getRhs() == mulOp0.getResult());
            bool mul1UsesInput = (mulOp1.getLhs() == divOp.getLhs() ||
                                  mulOp1.getRhs() == divOp.getLhs());
            if (erfUsesDiv && addUsesErf && mul0UsesAdd && mul1UsesMul0 &&
                mul1UsesInput) {
                match.type = "gelu";
                match.input = divOp.getLhs();
                match.output = mulOp1.getResult();
                match.consumed = 5;
                match.ops = {divOp.getOperation(), erfOp.getOperation(),
                             addOp.getOperation(), mulOp0.getOperation(),
                             mulOp1.getOperation()};
                return true;
            }
        }
    }

    if (auto tanhOp = dyn_cast<math::TanhOp>(elementwiseOps[index])) {
        match.type = "tanh";
        match.input = tanhOp.getOperand();
        match.output = tanhOp.getResult();
        match.consumed = 1;
        match.ops = {tanhOp.getOperation()};
        return true;
    }

    return false;
}

// 将generic块中的block arguement参数映射到对应的generic输入/初始化值
// Map a generic block argument to its input or init value.
// mlir::Value类型，提供类型，用户查询等
// Value carries type info and use-def relationships.
static Value mapBlockArgToGenericValue(linalg::GenericOp genericOp, Value v) {
    if (auto arg = v.dyn_cast<BlockArgument>()) {
        // 获取generic块的输入/初始化值
        // Read generic input/init values.
        unsigned numInputs = genericOp.getNumDpsInputs();
        // 获取参数索引
        // Get the argument index.
        unsigned index = arg.getArgNumber();
        if (index < numInputs)
            return genericOp.getDpsInputs()[index];
        unsigned initIndex = index - numInputs;
        if (initIndex < genericOp.getNumDpsInits())
            return genericOp.getDpsInits()[initIndex];
    }
    return Value();
}

// 确认Value值为常量：0
// Check whether the value is constant zero.
static bool isZeroConstant(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>())
            return floatAttr.getValue().isZero();
    }
    return false;
}

static bool isFloatConstant(Value v, double expectedValue) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>())
            return floatAttr.getValue().isExactlyValue(expectedValue);
    }
    return false;
}

static Value remapValue(Value v,
                        const llvm::DenseMap<Value, Value> &valueRemap) {
    auto it = valueRemap.find(v);
    while (it != valueRemap.end()) {
        v = it->second;
        it = valueRemap.find(v);
    }
    return v;
}

static Value chooseActivationInput(Value lhs, Value rhs,
                                   ArrayRef<Operation *> filteredOps,
                                   const llvm::DenseMap<Value, Value> &valueRemap) {
    auto lhsDef = remapValue(lhs, valueRemap).getDefiningOp();
    auto rhsDef = remapValue(rhs, valueRemap).getDefiningOp();
    bool lhsInChain = lhsDef && llvm::is_contained(filteredOps, lhsDef);
    bool rhsInChain = rhsDef && llvm::is_contained(filteredOps, rhsDef);
    if (lhsInChain && !rhsInChain)
        return lhs;
    if (rhsInChain && !lhsInChain)
        return rhs;
    return lhs;
}

static bool getGenericOperandIndex(linalg::GenericOp genericOp, Value v,
                                   int &operandIndex) {
    operandIndex = -1;
    if (!genericOp || !v)
        return false;
    auto arg = v.dyn_cast<BlockArgument>();
    if (!arg || arg.getOwner() != genericOp.getBody())
        return false;
    operandIndex = static_cast<int>(arg.getArgNumber());
    return true;
}

static Value getGenericInputValue(linalg::GenericOp genericOp,
                                  int operandIndex) {
    if (!genericOp || operandIndex < 0)
        return Value();

    unsigned idx = static_cast<unsigned>(operandIndex);
    unsigned numInputs = genericOp.getNumDpsInputs();
    if (idx < numInputs)
        return genericOp.getDpsInputs()[idx];

    idx -= numInputs;
    if (idx < genericOp.getNumDpsInits())
        return genericOp.getDpsInits()[idx];
    return Value();
}

static Value createSplatTensor(PatternRewriter &rewriter, Location loc,
                               ShapedType type, Value like, double value) {
    if (!type)
        type = like.getType().dyn_cast<ShapedType>();
    if (!type)
        return Value();

    auto rankedType = type.dyn_cast<RankedTensorType>();
    if (!rankedType || !rankedType.getElementType().isa<FloatType>())
        return Value();

    auto elemType = rankedType.getElementType().cast<FloatType>();
    Attribute scalar = rewriter.getFloatAttr(elemType, value);
    DenseElementsAttr splat = DenseElementsAttr::get(rankedType, scalar);
    return rewriter.create<arith::ConstantOp>(loc, rankedType, splat).getResult();
}

static Value createElementwiseBinaryTensorOp(PatternRewriter &rewriter,
                                             Location loc, Value lhs, Value rhs,
                                             BinaryKind kind) {
    if (!lhs || !rhs)
        return Value();
    if (lhs.getType() != rhs.getType())
        return Value();
    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType || lhsType != rhsType)
        return Value();

    Value init = rewriter.create<tensor::EmptyOp>(loc, lhsType.getShape(),
                                                  lhsType.getElementType());
    auto identity = rewriter.getMultiDimIdentityMap(lhsType.getRank());
    SmallVector<AffineMap, 3> maps = {identity, identity, identity};
    SmallVector<utils::IteratorType, 4> iterators(
        lhsType.getRank(), utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, lhsType, ValueRange{lhs, rhs}, ValueRange{init}, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value v;
            switch (kind) {
            case BinaryKind::Add:
                v = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
                break;
            case BinaryKind::Sub:
                v = b.create<arith::SubFOp>(nestedLoc, args[0], args[1]);
                break;
            case BinaryKind::Mul:
                v = b.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
                break;
            }
            b.create<linalg::YieldOp>(nestedLoc, v);
        });
    return generic.getResult(0);
}

static bool getGenericInputOperandIndex(linalg::GenericOp genericOp, Value v,
                                        int &operandIndex) {
    if (!getGenericOperandIndex(genericOp, v, operandIndex))
        return false;
    return operandIndex >= 0 &&
           operandIndex < static_cast<int>(genericOp.getNumDpsInputs());
}

static bool foldSingleSubToAdd(linalg::GenericOp genericOp,
                               arith::SubFOp subOp,
                               PatternRewriter &rewriter,
                               FusionPatternInfo &pattern,
                               Value &newResult) {
    (void)pattern;
    int rhsIndex = -1;
    if (!getGenericInputOperandIndex(genericOp, subOp.getRhs(), rhsIndex) ||
        rhsIndex < 0)
        return false;

    Value rhsTensor = getGenericInputValue(genericOp, rhsIndex);
    auto rhsType = rhsTensor.getType().dyn_cast<ShapedType>();
    if (!rhsTensor || !rhsType)
        return false;

    rewriter.setInsertionPoint(genericOp);
    Value zero = createSplatTensor(rewriter, genericOp.getLoc(), rhsType,
                                   rhsTensor, 0.0);
    if (!zero)
        return false;
    Value negRhs = createElementwiseBinaryTensorOp(rewriter, genericOp.getLoc(),
                                                   zero, rhsTensor,
                                                   BinaryKind::Sub);
    if (!negRhs)
        return false;

    genericOp.setOperand(rhsIndex, negRhs);

    rewriter.setInsertionPoint(subOp);
    auto addOp = rewriter.create<arith::AddFOp>(subOp.getLoc(), subOp.getLhs(),
                                                subOp.getRhs());
    newResult = addOp.getResult();
    return true;
}

} // namespace mlir::iree_compiler