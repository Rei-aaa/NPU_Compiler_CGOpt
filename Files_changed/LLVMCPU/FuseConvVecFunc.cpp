#include "FuseConvVecFunc.h"

#include <string>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "iree/compiler/Dialect/NPUFuseOp/NPUFuseOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler {

static bool matchAddReluPattern(Operation *convOp, linalg::GenericOp genericOp,
                                ArrayRef<Operation *> elementwiseOps,
                                FusionPatternInfo &fusionInfo);
static bool matchAddBnReluPattern(Operation *convOp, linalg::GenericOp genericOp,
                                  ArrayRef<Operation *> elementwiseOps,
                                  FusionPatternInfo &fusionInfo);
static Value mapBlockArgToGenericValue(linalg::GenericOp genericOp, Value v);
static bool isZeroConstant(Value v);
static Operation *createFusedOpFromPattern(StringRef fusedName,
                                           Operation *convOp,
                                           const FusionPatternInfo &pattern,
                                           ArrayRef<Value> operands,
                                           PatternRewriter &rewriter);

//*****************************************************************************
//                      FuseConvVecOp中主要逻辑函数
//****************************************************************************

// 收集generic块中可融合的算子，并且匹配硬件支持的融合模式
bool collectFusableOps(Operation *convOp,
                       Operation *&fusedGenericOp,
                       ElementwiseChain &elementwiseOps,
                       FusionPatternInfo &pattern) {
    fusedGenericOp = nullptr;
    elementwiseOps.clear();
    pattern = FusionPatternInfo();

    while (!convOp->getResult(0).use_empty()) {
        if (!convOp->getResult(0).hasOneUse())
        // 操作需要有单一用户
            break;
        Operation *user = *convOp->getResult(0).getUsers().begin();
        if (!user || user->getBlock() != convOp->getBlock())
        // 本操作和用户在同一个基本块内
            break;
        if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
        // 后面的逐元素操作在linalg.generic块中
        
            // 收集generic块中的逐元素操作
            ElementwiseChain currentElementwiseOps;
               Block *bodyBlock = genericOp.getBody();
               for (Operation &inner : *bodyBlock) {
                    if (inner.hasTrait<OpTrait::IsTerminator>())
                        continue;
                    currentElementwiseOps.push_back(&inner);
                }
            FusionPatternInfo fusionInfo;
            //匹配支持的融合pattern
            if (matchAddBnReluPattern(convOp, genericOp, currentElementwiseOps, fusionInfo) ||
                matchAddReluPattern(convOp, genericOp, currentElementwiseOps, fusionInfo)) {
                fusedGenericOp = user;
                elementwiseOps = std::move(currentElementwiseOps);
                pattern = std::move(fusionInfo);
                break; 
            }
        }
        break;
    }
    return fusedGenericOp != nullptr;
}

Operation *rewriteWithFusedOp(  Operation *convOp,
                                Operation *fusedGenericOp,
                                const ElementwiseChain &elementwiseOps,
                                const FusionPatternInfo &pattern,
                                PatternRewriter &rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(convOp);

    // 构造新的融合操作的操作数列表
    SmallVector<Value, 10> newOperands(convOp->operand_begin(), convOp->operand_end());
    for (const ExtraOperandInfo &extra : pattern.extraOperands)
        newOperands.push_back(extra.value);

    // 确定融合后操作的名称（目前pattern）
    if (pattern.fusedOpName.empty()) {
        convOp->emitOpError("missing fused op name in pattern");
        return nullptr;
    }
    StringRef fusedName = pattern.fusedOpName;

    // 新增属性：将融合算子及其所属子算子信息绑定在一起
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

    Operation *newOp = createFusedOpFromPattern(fusedName, convOp, pattern,
                                                newOperands, rewriter);
    if (!newOp)
        return nullptr;

    // 继承原conv操作的属性
    for (NamedAttribute attr : convOp->getAttrs())
        newOp->setAttr(attr.getName(), attr.getValue());

    // 融合进conv操作的算子信息
    if (fusedOpInfoAttr)
        newOp->setAttr("fused_op_info", fusedOpInfoAttr);

    return newOp;
}

// 将链尾结果的用户重定向为新融合操作的结果
void redirectFusedChain(Operation *convOp,
                        Operation *fusedGenericOp,
                        Operation *newOp) {
    Operation *tailOp = fusedGenericOp ? fusedGenericOp : convOp;
    tailOp->getResult(0).replaceAllUsesWith(newOp->getResult(0));
}

void eraseFusedOps(Operation *convOp,
                   Operation *fusedGenericOp,
                   PatternRewriter &rewriter) {
    if (fusedGenericOp){
        rewriter.eraseOp(fusedGenericOp);
        rewriter.eraseOp(convOp);
    }
}

//*************************************************************************
//                          融合pattern匹配函数
//*************************************************************************

// 匹配add+relu的模式
static bool matchAddReluPattern(Operation *convOp, linalg::GenericOp genericOp,
                                ArrayRef<Operation *> elementwiseOps,
                                FusionPatternInfo &fusionInfo) {
                                    
    //if (elementwiseOps.size() != 3)
    //    return false;

    // 确认操作模式
    auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[0]);
    auto cmpOp = dyn_cast<arith::CmpFOp>(elementwiseOps[1]);
    auto selectOp = dyn_cast<arith::SelectOp>(elementwiseOps[2]);
    if (!addOp || !cmpOp || !selectOp)
        return false;
    
    // 获取addf操作数和conv结果
    Value convResult = convOp->getResult(0);
    Value lhsValue = mapBlockArgToGenericValue(genericOp, addOp.getLhs());
    Value rhsValue = mapBlockArgToGenericValue(genericOp, addOp.getRhs());
    // 确认add操作的其中一个输入来自conv操作
    bool lhsIsConv = lhsValue && lhsValue == convResult;
    bool rhsIsConv = rhsValue && rhsValue == convResult;
    if (lhsIsConv == rhsIsConv)
        return false;
    
    // 获取加法bias值
    Value biasValue = lhsIsConv ? rhsValue : lhsValue;
    if (!biasValue)
        return false;

    // 确认cmp操作和select操作的模式（支持比较两侧交换）
    bool cmpAddOnLhs = cmpOp.getLhs() == addOp.getResult() &&
                       isZeroConstant(cmpOp.getRhs());
    bool cmpAddOnRhs = cmpOp.getRhs() == addOp.getResult() &&
                       isZeroConstant(cmpOp.getLhs());
    if (!cmpAddOnLhs && !cmpAddOnRhs)
        return false;
    using Pred = arith::CmpFPredicate;
    if (cmpAddOnLhs) {
        if (cmpOp.getPredicate() != Pred::OGT && cmpOp.getPredicate() != Pred::UGT)
            return false;
        // select(cmp(add, 0), add, 0)
        if (selectOp.getCondition() != cmpOp.getResult() ||
            selectOp.getTrueValue() != addOp.getResult() ||
            !isZeroConstant(selectOp.getFalseValue()))
            return false;
    } else {
        if (cmpOp.getPredicate() != Pred::OLT && cmpOp.getPredicate() != Pred::ULT)
            return false;
        // select(cmp(0, add), 0, add)
        if (selectOp.getCondition() != cmpOp.getResult() ||
            !isZeroConstant(selectOp.getTrueValue()) ||
            selectOp.getFalseValue() != addOp.getResult())
            return false;
    }
    
    // 融合信息：
    fusionInfo = FusionPatternInfo();
    fusionInfo.fusedOpName = "npufuseop.conv_add_relu";
    
    // 融合子算子
    FusedSubOpInfo addInfo;
    addInfo.role = "add";
    addInfo.opNames.push_back("arith.addf");
    fusionInfo.fusedSubOps.push_back(std::move(addInfo));

    FusedSubOpInfo reluInfo;
    reluInfo.role = "relu";
    reluInfo.opNames.push_back("arith.cmpf");
    reluInfo.opNames.push_back("arith.select");
    fusionInfo.fusedSubOps.push_back(std::move(reluInfo));

    // 额外操作数
    fusionInfo.extraOperands.push_back({biasValue, "bias"});
    return true;
}

// 匹配 add + bn + relu 的模式
static bool matchAddBnReluPattern(Operation *convOp, linalg::GenericOp genericOp,
                                  ArrayRef<Operation *> elementwiseOps,
                                  FusionPatternInfo &fusionInfo) {

    // 期望的逐元素链：addf -> subf -> mulf -> mulf -> addf -> cmpf -> select
    if (elementwiseOps.size() != 7)
        return false;

    auto addOp = dyn_cast<arith::AddFOp>(elementwiseOps[0]);
    auto subOp = dyn_cast<arith::SubFOp>(elementwiseOps[1]);
    auto mulOp0 = dyn_cast<arith::MulFOp>(elementwiseOps[2]);
    auto mulOp1 = dyn_cast<arith::MulFOp>(elementwiseOps[3]);
    auto addOp1 = dyn_cast<arith::AddFOp>(elementwiseOps[4]);
    auto cmpOp = dyn_cast<arith::CmpFOp>(elementwiseOps[5]);
    auto selectOp = dyn_cast<arith::SelectOp>(elementwiseOps[6]);

    if (!addOp || !subOp || !mulOp0 || !mulOp1 || !addOp1 || !cmpOp || !selectOp)
        return false;

    Value convResult = convOp->getResult(0);
    Value lhsValue = mapBlockArgToGenericValue(genericOp, addOp.getLhs());
    Value rhsValue = mapBlockArgToGenericValue(genericOp, addOp.getRhs());
    bool lhsIsConv = lhsValue && lhsValue == convResult;
    bool rhsIsConv = rhsValue && rhsValue == convResult;
    if (lhsIsConv == rhsIsConv)
        return false;

    Value biasValue = lhsIsConv ? rhsValue : lhsValue;
    if (!biasValue)
        return false;

    // subf(add, mean) or subf(mean, add)
    if (subOp.getLhs() != addOp.getResult() &&
        subOp.getRhs() != addOp.getResult())
        return false;
    Value meanValue = mapBlockArgToGenericValue(
        genericOp, subOp.getLhs() == addOp.getResult() ? subOp.getRhs()
                                                       : subOp.getLhs());
    if (!meanValue)
        return false;

    // mulf(sub, variance) or mulf(variance, sub)
    if (mulOp0.getLhs() != subOp.getResult() &&
        mulOp0.getRhs() != subOp.getResult())
        return false;
    Value varianceValue = mapBlockArgToGenericValue(
        genericOp, mulOp0.getLhs() == subOp.getResult() ? mulOp0.getRhs()
                                                       : mulOp0.getLhs());
    if (!varianceValue)
        return false;

    // mulf(mul0, scale) or mulf(scale, mul0)
    if (mulOp1.getLhs() != mulOp0.getResult() &&
        mulOp1.getRhs() != mulOp0.getResult())
        return false;
    Value scaleValue = mapBlockArgToGenericValue(
        genericOp, mulOp1.getLhs() == mulOp0.getResult() ? mulOp1.getRhs()
                                                         : mulOp1.getLhs());
    if (!scaleValue)
        return false;

    // addf(mul1, offset) or addf(offset, mul1)
    if (addOp1.getLhs() != mulOp1.getResult() &&
        addOp1.getRhs() != mulOp1.getResult())
        return false;
    Value offsetValue = mapBlockArgToGenericValue(
        genericOp, addOp1.getLhs() == mulOp1.getResult() ? addOp1.getRhs()
                                                         : addOp1.getLhs());
    if (!offsetValue)
        return false;

    // relu: cmp(add1, 0) + select（支持比较两侧交换）
    bool cmpAddOnLhs = cmpOp.getLhs() == addOp1.getResult() &&
                       isZeroConstant(cmpOp.getRhs());
    bool cmpAddOnRhs = cmpOp.getRhs() == addOp1.getResult() &&
                       isZeroConstant(cmpOp.getLhs());
    if (!cmpAddOnLhs && !cmpAddOnRhs)
        return false;
    using Pred = arith::CmpFPredicate;
    if (cmpAddOnLhs) {
        if (cmpOp.getPredicate() != Pred::OGT && cmpOp.getPredicate() != Pred::UGT)
            return false;
        if (selectOp.getCondition() != cmpOp.getResult() ||
            selectOp.getTrueValue() != addOp1.getResult() ||
            !isZeroConstant(selectOp.getFalseValue()))
            return false;
    } else {
        if (cmpOp.getPredicate() != Pred::OLT && cmpOp.getPredicate() != Pred::ULT)
            return false;
        if (selectOp.getCondition() != cmpOp.getResult() ||
            !isZeroConstant(selectOp.getTrueValue()) ||
            selectOp.getFalseValue() != addOp1.getResult())
            return false;
    }

    fusionInfo = FusionPatternInfo();
    fusionInfo.fusedOpName = "npufuseop.conv_add_bn_relu";

    FusedSubOpInfo addInfo;
    addInfo.role = "add";
    addInfo.opNames.push_back("arith.addf");
    fusionInfo.fusedSubOps.push_back(std::move(addInfo));

    FusedSubOpInfo bnInfo;
    bnInfo.role = "batch_norm";
    bnInfo.opNames.push_back("arith.subf");
    bnInfo.opNames.push_back("arith.mulf");
    bnInfo.opNames.push_back("arith.mulf");
    bnInfo.opNames.push_back("arith.addf");
    fusionInfo.fusedSubOps.push_back(std::move(bnInfo));

    FusedSubOpInfo reluInfo;
    reluInfo.role = "relu";
    reluInfo.opNames.push_back("arith.cmpf");
    reluInfo.opNames.push_back("arith.select");
    fusionInfo.fusedSubOps.push_back(std::move(reluInfo));

    fusionInfo.extraOperands.push_back({biasValue, "bias"});
    fusionInfo.extraOperands.push_back({meanValue, "bn_mean"});
    fusionInfo.extraOperands.push_back({varianceValue, "bn_variance"});
    fusionInfo.extraOperands.push_back({scaleValue, "bn_scale"});
    fusionInfo.extraOperands.push_back({offsetValue, "bn_offset"});
    return true;
}

//**************************************************************************
//                          融合操作创建函数
//**************************************************************************

static Operation *createFusedOpFromPattern(StringRef fusedName,
                                           Operation *convOp,
                                           const FusionPatternInfo &pattern,
                                           ArrayRef<Value> operands,
                                           PatternRewriter &rewriter) {
    Location loc = convOp->getLoc();

    if (fusedName == "npufuseop.conv_add_relu") {
        using namespace mlir::iree::compiler::Dialect::NPUFuseOp;
        if (operands.size() < 3 || pattern.extraOperands.empty()) {
            convOp->emitOpError("expected conv operands + bias for fused op");
            return nullptr;
        }
        Value input = operands[0];
        Value filter = operands[1];
        Value outputInit = operands[2];
        Value bias = pattern.extraOperands.front().value;
        auto fused = rewriter.create<ConvAddReluOp>(
            loc, convOp->getResult(0).getType(), input, filter, bias, outputInit);
        return fused.getOperation();
    }

    if (fusedName == "npufuseop.conv_add_bn_relu") {
        using namespace mlir::iree::compiler::Dialect::NPUFuseOp;
        if (operands.size() < 3 || pattern.extraOperands.size() < 5) {
            convOp->emitOpError("expected conv operands + bn params for fused op");
            return nullptr;
        }
        Value input = operands[0];
        Value filter = operands[1];
        Value outputInit = operands[2];
        Value addend = pattern.extraOperands[0].value;
        Value bnMean = pattern.extraOperands[1].value;
        Value bnVariance = pattern.extraOperands[2].value;
        Value bnScale = pattern.extraOperands[3].value;
        Value bnOffset = pattern.extraOperands[4].value;
        auto fused = rewriter.create<ConvAddBnReluOp>(
            loc, convOp->getResult(0).getType(), input, filter, addend, bnMean,
            bnVariance, bnScale, bnOffset, outputInit);
        return fused.getOperation();
    }

    // 通用创建方式
    OperationState state(loc, fusedName);
    state.addOperands(operands);
    state.addTypes(convOp->getResultTypes());
    return rewriter.create(state);
}

//******************************************************************
//                        其它辅助函数实现
//******************************************************************

// 将generic块中的block arguement参数映射到对应的generic输入/初始化值
// mlir::Value类型，提供类型，用户查询等
static Value mapBlockArgToGenericValue(linalg::GenericOp genericOp, Value v) {
    if (auto arg = v.dyn_cast<BlockArgument>()) {
        // 获取generic块的输入/初始化值
        unsigned numInputs = genericOp.getNumDpsInputs();
        // 获取参数索引
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
static bool isZeroConstant(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>())
            return floatAttr.getValue().isZero();
    }
    return false;
}

namespace {

} // namespace

} // namespace mlir::iree_compiler