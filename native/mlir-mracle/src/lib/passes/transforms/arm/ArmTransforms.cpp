#include "mlir-mracle/passes/transforms/arm/ArmTransforms.h"
#include "mlir-mracle/passes/transforms/Transforms.h"
#include "mlir-mracle/passes/transforms/MetamorphicTransform.h"
#include "mlir-mracle/context/context.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <random>
#include <utility>

namespace mlir {
namespace metamorphic {

// an unset memory order defaults to seq_cst, which must never be reordered
static bool isRelaxed(omp::AtomicReadOp op) {
    return op.getMemoryOrder().value_or(omp::ClauseMemoryOrderKind::Seq_cst) ==
           omp::ClauseMemoryOrderKind::Relaxed;
}

static bool isRelaxed(omp::AtomicWriteOp op) {
    return op.getMemoryOrder().value_or(omp::ClauseMemoryOrderKind::Seq_cst) ==
           omp::ClauseMemoryOrderKind::Relaxed;
}

static bool isPureArithmeticGap(Operation *op) {
    return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
               arith::ConstantOp, arith::CmpIOp>(op);
}

static bool blocksCommuteSearch(Operation *op) {
    return isa<memref::LoadOp, memref::StoreOp, omp::AtomicReadOp>(op);
}

// true when no operand of `moved` is produced by `anchor` or by any op between
// them, so moving `moved` before `anchor` cannot create a use-before-def
static bool operandsDominate(Operation *anchor, Operation *moved) {
    for (Operation *op = anchor; op && op != moved; op = op->getNextNode())
        for (Value res : op->getResults())
            for (Value operand : moved->getOperands())
                if (operand == res)
                    return false;
    return true;
}

// commutes an independent relaxed atomic read and write, so the write runs first
bool tryCommuteRelaxedReadWrite(func::FuncOp op, RewriterBase &rewriter,
                                std::mt19937 &rng) {
    SmallVector<std::pair<omp::AtomicReadOp, omp::AtomicWriteOp>> candidates;
    op.walk([&](omp::AtomicReadOp read) {
        if (!isRelaxed(read))
            return;
        Operation *nextOp = read->getNextNode();
        while (nextOp) {
            if (auto nextWrite = dyn_cast<omp::AtomicWriteOp>(nextOp)) {
                // outcome-preserving only for independent accesses: both ops
                // relaxed, the write must target a location the read does not
                // touch, and it must not depend on the read or the ops between
                if (isRelaxed(nextWrite) && read.getX() != nextWrite.getX() &&
                    read.getV() != nextWrite.getX() &&
                    operandsDominate(read, nextWrite))
                    candidates.push_back({read, nextWrite});
                break;
            }
            if (blocksCommuteSearch(nextOp))
                break;
            if (!isPureArithmeticGap(nextOp))
                break;
            nextOp = nextOp->getNextNode();
        }
    });

    if (candidates.empty())
        return false;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    auto [read, write] = candidates[dist(rng)];

    rewriter.moveOpBefore(write, read);

    return true;
}

// commutes two independent relaxed atomic writes, so the second runs first
bool tryCommuteRelaxedWriteWrite(func::FuncOp op, RewriterBase &rewriter,
                                 std::mt19937 &rng) {
    SmallVector<std::pair<omp::AtomicWriteOp, omp::AtomicWriteOp>> candidates;
    op.walk([&](omp::AtomicWriteOp first) {
        if (!isRelaxed(first))
            return;
        Operation *nextOp = first->getNextNode();
        while (nextOp) {
            if (auto nextWrite = dyn_cast<omp::AtomicWriteOp>(nextOp)) {
                // outcome-preserving only for independent accesses: both ops
                // relaxed, targeting different locations, and the second write
                // must not depend on the ops between
                if (isRelaxed(nextWrite) && first.getX() != nextWrite.getX() &&
                    operandsDominate(first, nextWrite))
                    candidates.push_back({first, nextWrite});
                break;
            }
            if (blocksCommuteSearch(nextOp))
                break;
            if (!isPureArithmeticGap(nextOp))
                break;
            nextOp = nextOp->getNextNode();
        }
    });

    if (candidates.empty())
        return false;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    auto [first, second] = candidates[dist(rng)];

    rewriter.moveOpBefore(second, first);

    return true;
}

namespace {
const FunctionTransform kArmTransforms[] = {
    FunctionTransform("commute-relaxed-read-write", "armv8",
                      &tryCommuteRelaxedReadWrite,
                      mlir_mracle::OutcomeRelation::Equality),
    FunctionTransform("commute-relaxed-write-write", "armv8",
                      &tryCommuteRelaxedWriteWrite,
                      mlir_mracle::OutcomeRelation::Equality),
};
} // namespace

llvm::ArrayRef<FunctionTransform> getArmTransforms() {
    return kArmTransforms;
}

} // namespace metamorphic
} // namespace mlir
