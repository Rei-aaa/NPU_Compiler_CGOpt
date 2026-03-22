#include "FuseConvVecFunc.h"                
#include <string>
#include <utility>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "iree/compiler/Dialect/NPUOp/NPUOps.h"
#include "llvm/ADT/DenseMap.h"
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
static void replaceExtraOperand(FusionPatternInfo &pattern, StringRef role,
                                Value newValue);
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

/////////////////////////////////////////////////////////////////////
//                main functions used in rewrite                   //
/////////////////////////////////////////////////////////////////////

bool collectFusableOps(Operation *convOp,
                       Operation *&fusedGenericOp,
                       ElementwiseChain &elementwiseOps) {
    fusedGenericOp = nullptr;
    elementwiseOps.clear();

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
        fusedGenericOp = user;
        break;
    }
    return fusedGenericOp != nullptr;
}

void convertToNpuOp(linalg::GenericOp genericOp,
                        ElementwiseChain &elementwiseOps,
                        PatternRewriter &rewriter) {
    if (!genericOp || elementwiseOps.empty())
        return;
    using namespace mlir::iree::compiler::Dialect::NPUOp;

// 根据硬件特性从近匹配可融合的逐元素操作 -> npuop
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

    // Start matching from the nearest op in the chain.
    for (size_t i = 0; i < elementwiseOps.size();) {
        // 需要先匹配激活函数
        // Try to match activation patterns first.
        ActivationMatch activation;
        if (matchActivation(i, elementwiseOps, activation)) {
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::Act, slotIndex))
                break;
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

            if (actOp)
                activation.output.replaceAllUsesWith(actOp->getResult(0));
            i += activation.consumed;
            continue;
        }
        
        // mul : PE_a
        if (auto mulOp = dyn_cast<arith::MulFOp>(elementwiseOps[i])) {
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::PeA, slotIndex))
                break;
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
            ++i;
            continue;
        }

        // add : PE_b
        if (auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[i])) {
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::PeB, slotIndex))
                break;
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
            ++i;
            continue;
        }

        // sub : PE_b (x - y => x + (-y)
        if (auto subOp = dyn_cast<arith::SubFOp>(elementwiseOps[i])) {
            int slotIndex = 0;
            if (!consumeNextSlot(StageKind::PeB, slotIndex))
                break;
            rewriter.setInsertionPoint(subOp);
            auto loc = subOp.getLoc();
            auto resultType = subOp.getResult().getType();
            Operation *peOp = nullptr;
            if (slotIndex == 1)
                peOp = rewriter.create<PE1BOp>(loc, resultType, subOp.getLhs(),
                                               subOp.getRhs());
            else if (slotIndex == 2)
                peOp = rewriter.create<PE2BOp>(loc, resultType, subOp.getLhs(),
                                               subOp.getRhs());
            else
                peOp = rewriter.create<PE3BOp>(loc, resultType, subOp.getLhs(),
                                               subOp.getRhs());
            peOp->setAttr("npu.from_sub", rewriter.getBoolAttr(true));
            subOp.getResult().replaceAllUsesWith(peOp->getResult(0));
            ++i;
            continue;
        }

        break;
    }

    // 清理已经被 npuop 结果替换后的旧逐元素算子
    // Erase old elementwise ops whose uses were replaced by NPU ops.
    for (Operation *op : llvm::reverse(elementwiseOps)) {
        if (op && op->use_empty())
            rewriter.eraseOp(op);
    }
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

// rewrite convOp -> fusedConvOp
Operation *rewriteWithFusedOp(Operation *convOp,
                              Operation *fusedGenericOp,
                              const ElementwiseChain &elementwiseOps,
                              const FusionPatternInfo &pattern,
                              PatternRewriter &rewriter,
                              bool enableFastMath) {
    OpBuilder::InsertionGuard guard(rewriter);

    using namespace mlir::iree::compiler::Dialect::NPUOp;

    auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                    : linalg::GenericOp();
    if (fusedGenericOp)
        rewriter.setInsertionPoint(fusedGenericOp);
    else
        rewriter.setInsertionPoint(convOp);

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
                "inputs/inits; refusing fusion to avoid leaving unfused npuop "
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
        bool reachedResidualOps = false;

        // 激活位置，数字指前面出现的槽位顺序，所以开始是2
        // Activation position is keyed by prior slot order, so it starts with 2
        int activationAfter = 2;

        // Resolve additional side operands needed by fused conv.
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
            return static_cast<bool>(sideTensor);
        };

        // For sub-origin ops, build a pre-conv negate to fit add-only PE_b.
        auto maybeNegateSideBeforeConv = [&](Value sideTensor,
                                             Operation *peOp) -> Value {
            auto fromSub = peOp->getAttrOfType<BoolAttr>("npu.from_sub");
            if (!fromSub || !fromSub.getValue())
                return sideTensor;
            auto shapedType = sideTensor.getType().dyn_cast<ShapedType>();
            if (!shapedType)
                return sideTensor;
            Value zero = createSplatTensor(rewriter, convOp->getLoc(), shapedType,
                                           sideTensor, 0.0);
            return createElementwiseBinaryTensorOp(rewriter, convOp->getLoc(),
                                                   zero, sideTensor,
                                                   BinaryKind::Sub);
        };


        for (Operation &op : genericOp.getBody()->getOperations()) {
            if (op.hasTrait<OpTrait::IsTerminator>())
                continue;

            // Only activation or PE ops are valid in this fused prefix.
            if (!isa<ReluOp, LeakyReluOp, SiluOp, SigmoidOp, GeluOp, TanhOp,
                    PE1AOp, PE1BOp, PE2AOp, PE2BOp, PE3AOp, PE3BOp>(&op)) {
                reachedResidualOps = true;
                continue;
            }

            // 仅支持前缀连续的 npuop/activation融合，后续 elementwise 留在 generic 内。
            // Only a contiguous npuop/activation prefix can be fused.
            if (reachedResidualOps) {
                convOp->emitOpError(
                    "expected npuop/activation prefix before residual generic ops");
                return nullptr;
            }

            if (auto peOp = dyn_cast<PE1AOp>(&op)) {
                if (pe1_a)
                    return nullptr;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe1_a))
                    return nullptr;
                activationAfter = 2;
                // after 2：PE1    after 5：PE2    after 8：PE3
                currentStream = peOp.getResult();
                consumedOps.push_back(&op);
                continue;
            }
            if (auto peOp = dyn_cast<PE1BOp>(&op)) {
                if (pe1_b)
                    return nullptr;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe1_b))
                    return nullptr;
                pe1_b = maybeNegateSideBeforeConv(pe1_b, peOp.getOperation());
                activationAfter = 2;
                currentStream = peOp.getResult();
                consumedOps.push_back(&op);
                continue;
            }
            if (auto peOp = dyn_cast<PE2AOp>(&op)) {
                if (pe2_a)
                    return nullptr;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe2_a))
                    return nullptr;
                activationAfter = 5;
                currentStream = peOp.getResult();
                consumedOps.push_back(&op);
                continue;
            }
            if (auto peOp = dyn_cast<PE2BOp>(&op)) {
                if (pe2_b)
                    return nullptr;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe2_b))
                    return nullptr;
                pe2_b = maybeNegateSideBeforeConv(pe2_b, peOp.getOperation());
                activationAfter = 5;
                currentStream = peOp.getResult();
                consumedOps.push_back(&op);
                continue;
            }
            if (auto peOp = dyn_cast<PE3AOp>(&op)) {
                if (pe3_a)
                    return nullptr;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe3_a))
                    return nullptr;
                activationAfter = 8;
                currentStream = peOp.getResult();
                consumedOps.push_back(&op);
                continue;
            }
            if (auto peOp = dyn_cast<PE3BOp>(&op)) {
                if (pe3_b)
                    return nullptr;
                if (!resolveSideOperand(peOp.getLhs(), peOp.getRhs(), pe3_b))
                    return nullptr;
                pe3_b = maybeNegateSideBeforeConv(pe3_b, peOp.getOperation());
                activationAfter = 8;
                currentStream = peOp.getResult();
                consumedOps.push_back(&op);
                continue;
            }

            auto handleActivation = [&](StringRef type, Value input,
                                        Value output) -> bool {
                if (input != currentStream)
                    return false;
                ActivationInfo info;
                info.type = type.str();
                info.afterOpIndex = activationAfter;
                activationInfo.push_back(std::move(info));
                currentStream = output;
                return true;
            };

            if (auto act = dyn_cast<ReluOp>(&op)) {
                if (!handleActivation("relu", act.getInput(), act.getOutput()))
                    return nullptr;
                consumedOps.push_back(&op);
                continue;
            }
            if (auto act = dyn_cast<LeakyReluOp>(&op)) {
                if (!handleActivation("leakyrelu", act.getInput(), act.getOutput()))
                    return nullptr;
                consumedOps.push_back(&op);
                continue;
            }
            if (auto act = dyn_cast<SiluOp>(&op)) {
                if (!handleActivation("silu", act.getInput(), act.getOutput()))
                    return nullptr;
                consumedOps.push_back(&op);
                continue;
            }
            if (auto act = dyn_cast<SigmoidOp>(&op)) {
                if (!handleActivation("sigmoid", act.getInput(), act.getOutput()))
                    return nullptr;
                consumedOps.push_back(&op);
                continue;
            }
            if (auto act = dyn_cast<GeluOp>(&op)) {
                if (!handleActivation("gelu", act.getInput(), act.getOutput()))
                    return nullptr;
                consumedOps.push_back(&op);
                continue;
            }
            if (auto act = dyn_cast<TanhOp>(&op)) {
                if (!handleActivation("tanh", act.getInput(), act.getOutput()))
                    return nullptr;
                consumedOps.push_back(&op);
                continue;
            }

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

        // 保留剩余elementwise，仅移除已经融合进 conv 的 npuop
        // Keep residual elementwise ops and remove only consumed npuops.
        if (currentStream != streamBlockArg)
            currentStream.replaceAllUsesWith(streamBlockArg);
        for (Operation *consumedOp : llvm::reverse(consumedOps)) {
            if (consumedOp->use_empty())
                rewriter.eraseOp(consumedOp);
        }

        return fused.getOperation();
    }

    // 构造新的融合操作的操作数列表
    // Build operands for the new fused operation.
    SmallVector<Value, 10> newOperands(convOp->operand_begin(), convOp->operand_end());
    for (const ExtraOperandInfo &extra : pattern.extraOperands)
        newOperands.push_back(extra.value);

    StringRef fusedName = Conv2DOp::getOperationName();

    // 子算子信息绑定
    // Attach metadata describing fused sub-ops.
    SmallVector<Attribute, 10> fusedOpInfoAttrs;
    for (const FusedSubOpInfo &subOp : pattern.fusedSubOps) {
        for (const std::string &opName : subOp.opNames) {
            SmallVector<NamedAttribute, 2> dict;
            dict.emplace_back(rewriter.getStringAttr("op"),
            rewriter.getStringAttr(opName));
            if (!subOp.role.empty()) {
                dict.emplace_back(rewriter.getStringAttr("belongs_to_subop"),
                                  rewriter.getStringAttr(subOp.role));
            }
            fusedOpInfoAttrs.push_back(rewriter.getDictionaryAttr(dict));
        }
    }

    auto fusedOpInfoAttr = fusedOpInfoAttrs.empty()
                               ? Attribute()
                               : rewriter.getArrayAttr(fusedOpInfoAttrs);

    SmallVector<Attribute, 6> activationAttrs;
    for (const ActivationInfo &activation : pattern.activations) {
        SmallVector<NamedAttribute, 2> dict;
        dict.emplace_back(rewriter.getStringAttr("type"),
                          rewriter.getStringAttr(activation.type));
        dict.emplace_back(rewriter.getStringAttr("after"),
                          rewriter.getI64IntegerAttr(activation.afterOpIndex));
        activationAttrs.push_back(rewriter.getDictionaryAttr(dict));
    }
    auto activationAttr = activationAttrs.empty()
                              ? Attribute()
                              : rewriter.getArrayAttr(activationAttrs);

    Operation *newOp = createFusedOpFromPattern(fusedName, convOp, pattern,
                                                newOperands, rewriter);
    if (!newOp)
        return nullptr;

    // 继承原conv操作的属性
    // Inherit attributes from the original conv op.
    for (NamedAttribute attr : convOp->getAttrs())
        newOp->setAttr(attr.getName(), attr.getValue());

    // 融合进conv操作的算子信息
    // Attach info about ops merged into conv.
    if (fusedOpInfoAttr)
        newOp->setAttr("fused_op_info", fusedOpInfoAttr);

    // 记录激活类型与位置
    // Record activation types and insertion points.
    if (activationAttr)
        newOp->setAttr("npu.activation", activationAttr);

    if (enableFastMath)
        newOp->setAttr("npu.fastmath", rewriter.getBoolAttr(true));

    // 如果只匹配了前缀操作链，则保留generic块并删除已融合部分
    // For partial prefix matches, keep generic and drop only fused ops.
    if (fusedGenericOp && pattern.matchedOpCount > 0 &&
        pattern.matchedOpCount < static_cast<int>(elementwiseOps.size())) {
        auto genericOp = dyn_cast<linalg::GenericOp>(fusedGenericOp);
        if (!genericOp) {
            convOp->emitOpError("expected linalg.generic for partial fusion");
            return nullptr;
        }
        Block *body = genericOp.getBody();
        unsigned numInputs = genericOp.getNumDpsInputs();
        if (genericOp.getNumDpsInits() == 0) {
            convOp->emitOpError("expected output init for partial fusion");
            return nullptr;
        }
        // 使用 fused op 的输出作为 generic 的输出 init
        // Use fused output as the generic output init.
        genericOp->setOperand(numInputs, newOp->getResult(0));

        // 用输出 block 参数替换已融合链的最后结果
        // Replace the fused tail value with the output block arg.
        Value outputBlockArg = body->getArgument(numInputs);
        if (pattern.matchedValue) {
            Value matchedValue = pattern.matchedValue;
            matchedValue.replaceAllUsesWith(outputBlockArg);
        }

        // 删除已融合的逐元素操作（从后往前）
        // Erase fused elementwise ops from back to front.
        for (int i = pattern.matchedOpCount - 1; i >= 0; --i) {
            rewriter.eraseOp(elementwiseOps[i]);
        }
    }

    return newOp;
}

// 将链尾结果的用户重定向为新融合操作的结果
// Redirect users of the chain tail to the fused result.
void redirectFusedChain(Operation *convOp,
                        Operation *fusedGenericOp,
                        Operation *newOp,
                        const FusionPatternInfo &pattern,
                        const ElementwiseChain &elementwiseOps) {
    if (auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                        : linalg::GenericOp()) {
        if (hasNonTerminatorBodyOps(genericOp) &&
            !genericUsesOperandValue(genericOp, convOp->getResult(0)))
            return;
    }

    // 仅在完全匹配时重定向到融合结果
    // Redirect only when the chain is fully matched.
    if (fusedGenericOp && pattern.matchedOpCount > 0 &&
        pattern.matchedOpCount < static_cast<int>(elementwiseOps.size()))
        return;
    Operation *tailOp = fusedGenericOp ? fusedGenericOp : convOp;
    tailOp->getResult(0).replaceAllUsesWith(newOp->getResult(0));
}

void eraseFusedOps(Operation *convOp,
                   Operation *fusedGenericOp,
                   PatternRewriter &rewriter,
                   const FusionPatternInfo &pattern,
                   const ElementwiseChain &elementwiseOps) {
    if (auto genericOp = fusedGenericOp ? dyn_cast<linalg::GenericOp>(fusedGenericOp)
                                        : linalg::GenericOp()) {
        if (hasNonTerminatorBodyOps(genericOp) &&
            !genericUsesOperandValue(genericOp, convOp->getResult(0))) {
            rewriter.eraseOp(convOp);
            return;
        }
    }

    if (fusedGenericOp && pattern.matchedOpCount > 0 &&
        pattern.matchedOpCount < static_cast<int>(elementwiseOps.size())) {
        rewriter.eraseOp(convOp);
        return;
    }
    if (fusedGenericOp) {
        rewriter.eraseOp(fusedGenericOp);
        rewriter.eraseOp(convOp);
    }
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

// Operand position as a generic block argument.
static bool getGenericOperandIndex(linalg::GenericOp genericOp, Value v,
                                   int &operandIndex) {
    auto arg = v.dyn_cast<BlockArgument>();
    if (!arg)
        return false;
    if (arg.getOwner() != genericOp.getBody())
        return false;
    unsigned argNumber = arg.getArgNumber();
    if (argNumber >= genericOp.getNumDpsInputs())
        return false;
    operandIndex = static_cast<int>(argNumber);
    return true;
}

static Value getGenericInputValue(linalg::GenericOp genericOp,
                                  int operandIndex) {
    if (operandIndex < 0 ||
        operandIndex >= static_cast<int>(genericOp.getNumDpsInputs()))
        return Value();
    return genericOp.getDpsInputs()[operandIndex];
}

// Create a splat tensor with the same shape and type.
static Value createSplatTensor(PatternRewriter &rewriter, Location loc,
                               ShapedType type, Value like, double value) {
    SmallVector<OpFoldResult, 4> sizes;
    sizes.reserve(type.getRank());
    // Collect shape dimensions.
    for (int64_t i = 0; i < type.getRank(); ++i) {
        int64_t dim = type.getDimSize(i);
        if (dim == ShapedType::kDynamic) {
            sizes.push_back(rewriter.create<tensor::DimOp>(loc, like, i)
                                .getResult());
        } else {
            sizes.push_back(rewriter.getIndexAttr(dim));
        }
    }
    // Create an uninitialized tensor.
    auto empty = rewriter.create<tensor::EmptyOp>(loc, sizes, type.getElementType());
    // Build a floating-point scalar value.
    auto elementType = type.getElementType().cast<FloatType>();
    auto scalarAttr = rewriter.getFloatAttr(elementType, value);
    auto scalarConst = rewriter.create<arith::ConstantOp>(loc, elementType,
                                                          scalarAttr);
    auto filled = rewriter.create<linalg::FillOp>(
        loc, ValueRange{scalarConst.getResult()},
        ValueRange{empty.getResult()});
    return filled.getResult(0);
}

static Value createElementwiseBinaryTensorOp(PatternRewriter &rewriter,
                                             Location loc, Value lhs, Value rhs,
                                             BinaryKind kind) {
    auto lhsType = lhs.getType().cast<ShapedType>();
    SmallVector<OpFoldResult, 4> sizes;
    sizes.reserve(lhsType.getRank());
    for (int64_t i = 0; i < lhsType.getRank(); ++i) {
        int64_t dim = lhsType.getDimSize(i);
        if (dim == ShapedType::kDynamic) {
            sizes.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i)
                                .getResult());
        } else {
            sizes.push_back(rewriter.getIndexAttr(dim));
        }
    }

    auto empty = rewriter.create<tensor::EmptyOp>(loc, sizes,
                                                  lhsType.getElementType());
    auto map = AffineMap::getMultiDimIdentityMap(lhsType.getRank(),
                                                 rewriter.getContext());
    SmallVector<AffineMap, 3> maps = {map, map, map};
    SmallVector<utils::IteratorType, 4> iterTypes(
        lhsType.getRank(), utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{lhsType}, ValueRange{lhs, rhs},
        ValueRange{empty.getResult()}, maps, iterTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value result;
            switch (kind) {
            case BinaryKind::Add:
                result = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
                break;
            case BinaryKind::Sub:
                result = b.create<arith::SubFOp>(nestedLoc, args[0], args[1]);
                break;
            case BinaryKind::Mul:
                result = b.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
                break;
            }
            b.create<linalg::YieldOp>(nestedLoc, result);
        });

    return generic.getResult(0);
}

static bool combineAddSubPair(linalg::GenericOp genericOp,
                              Operation *firstOp,
                              Operation *secondOp,
                              PatternRewriter &rewriter,
                              FusionPatternInfo &pattern,
                              Value &newResult,
                              bool enableFastMath) {
    (void)enableFastMath;
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
    if (!getGenericOperandIndex(genericOp, addOp.getLhs(), biasIndex))
        getGenericOperandIndex(genericOp, addOp.getRhs(), biasIndex);
    if (!getGenericOperandIndex(genericOp, secondSideArg, secondSideIndex))
        return false;
    if (biasIndex < 0 || secondSideIndex < 0)
        return false;

    // Resolve the corresponding input tensors.
    Value biasTensor = getGenericInputValue(genericOp, biasIndex);
    Value secondSideTensor = getGenericInputValue(genericOp, secondSideIndex);
    if (!biasTensor || !secondSideTensor)
        return false;

    rewriter.setInsertionPoint(genericOp);
    Value newBias = createElementwiseBinaryTensorOp(
        rewriter, genericOp.getLoc(), biasTensor, secondSideTensor, combineKind);
    auto secondSideType = secondSideTensor.getType().cast<ShapedType>();
    Value zeroSecondSide = createSplatTensor(rewriter, genericOp.getLoc(),
                                             secondSideType, secondSideTensor,
                                             0.0);

    genericOp.setOperand(biasIndex, newBias);
    genericOp.setOperand(secondSideIndex, zeroSecondSide);
    replaceExtraOperand(pattern, "bias", newBias);
    replaceExtraOperand(pattern, "bn_mean", zeroSecondSide);

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
    if (!getGenericOperandIndex(genericOp, firstMul.getLhs(), varianceIndex))
        getGenericOperandIndex(genericOp, firstMul.getRhs(), varianceIndex);
    Value scaleArg = secondMul.getLhs() == firstMul.getResult()
                         ? secondMul.getRhs()
                         : secondMul.getLhs();
    if (!getGenericOperandIndex(genericOp, scaleArg, scaleIndex))
        return false;
    if (varianceIndex < 0 || scaleIndex < 0)
        return false;

    Value varianceTensor = getGenericInputValue(genericOp, varianceIndex);
    Value scaleTensor = getGenericInputValue(genericOp, scaleIndex);
    if (!varianceTensor || !scaleTensor)
        return false;

    rewriter.setInsertionPoint(genericOp);
    Value newScale = createElementwiseBinaryTensorOp(
        rewriter, genericOp.getLoc(), varianceTensor, scaleTensor,
        BinaryKind::Mul);
    auto scaleType = scaleTensor.getType().cast<ShapedType>();
    Value ones = createSplatTensor(rewriter, genericOp.getLoc(), scaleType,
                                   scaleTensor, 1.0);

    genericOp.setOperand(varianceIndex, newScale);
    genericOp.setOperand(scaleIndex, ones);
    replaceExtraOperand(pattern, "bn_variance", newScale);
    replaceExtraOperand(pattern, "bn_scale", ones);

    secondMul.getResult().replaceAllUsesWith(firstMul.getResult());
    newResult = firstMul.getResult();
    return true;
}

void foldConstantElementwiseOps(linalg::GenericOp genericOp,
                                ElementwiseChain &elementwiseOps,
                                PatternRewriter &rewriter,
                                FusionPatternInfo &pattern,
                                bool enableFastMath) {
    (void)enableFastMath;

    bool changed = false;
    // Rebuild the chain while folding adjacent op pairs when possible.
    ElementwiseChain combinedOps;
    combinedOps.reserve(elementwiseOps.size());

    // Keep pattern.matchedValue aligned with rewritten SSA values.
    auto updateMatchedValue = [&](Value oldValue, Value newValue) {
        if (pattern.matchedValue == oldValue)
            pattern.matchedValue = newValue;
    };

    for (size_t i = 0; i < elementwiseOps.size();) {
        Operation *op = elementwiseOps[i];

        // Try pairwise combine first: (add/sub, add/sub) or (mul, mul).
        if (i + 1 < elementwiseOps.size()) {
            Operation *nextOp = elementwiseOps[i + 1];
            Value newResult;
            bool combined =
                combineAddSubPair(genericOp, op, nextOp, rewriter, pattern,
                                  newResult, enableFastMath) ||
                combineMulPair(genericOp, op, nextOp, rewriter, pattern, newResult,
                               enableFastMath);
            if (combined) {
                if (newResult) {
                    updateMatchedValue(op->getResult(0), newResult);
                    updateMatchedValue(nextOp->getResult(0), newResult);
                }

                Operation *newOp = newResult ? newResult.getDefiningOp() : nullptr;
                bool keepFirst = (newOp == op) || !newOp;
                bool keepSecond = (newOp == nextOp);

                // Preserve exactly one representative op in the rebuilt chain.
                if (newOp && newOp != op && newOp != nextOp) {
                    combinedOps.push_back(newOp);
                } else if (newOp == op) {
                    combinedOps.push_back(op);
                } else if (newOp == nextOp) {
                    combinedOps.push_back(nextOp);
                } else {
                    combinedOps.push_back(op);
                }

                if (!keepFirst && op->use_empty())
                    rewriter.eraseOp(op);
                if (!keepSecond && nextOp->use_empty())
                    rewriter.eraseOp(nextOp);

                changed = true;
                // Consumed two ops as one combined step.
                i += 2;
                continue;
            }
        }

        // No combine happened: keep original op and advance by one.
        combinedOps.push_back(op);
        ++i;
    }

    if (!changed)
        return;

    // Commit the rebuilt chain and clamp matchedOpCount to new chain length.
    elementwiseOps = std::move(combinedOps);
    if (pattern.matchedOpCount > static_cast<int>(elementwiseOps.size()))
        pattern.matchedOpCount = static_cast<int>(elementwiseOps.size());
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

static void replaceExtraOperand(FusionPatternInfo &pattern, StringRef role,
                                Value newValue) {
    for (ExtraOperandInfo &extra : pattern.extraOperands) {
        if (extra.role == role) {
            extra.value = newValue;
            return;
        }
    }
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

} // namespace mlir::iree_compiler