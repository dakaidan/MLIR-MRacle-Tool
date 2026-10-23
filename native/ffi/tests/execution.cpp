#include "mracle_execution.h"
#include "mlir-mracle/backend/jit/jit.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include <iostream>
#include <omp.h>
#include <pthread.h>

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    bool native = std::string(argv[2]) == "native";
    bool tsan = std::string(argv[3]) == "thread";
    mracleConfigureExecution(native, tsan, argv[1]);
    for (int opt = 0; opt < 4; ++opt) {
        llvm::LLVMContext context;
        auto module = std::make_unique<llvm::Module>("execution-contract", context);
        llvm::IRBuilder<> ir(context);
        auto *i64 = ir.getInt64Ty();
        auto *resultType = llvm::StructType::get(context, {i64, i64});
        auto *main = llvm::Function::Create(llvm::FunctionType::get(resultType, false),
            llvm::GlobalValue::ExternalLinkage, "main", module.get());
        ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", main));
        if (std::string(argv[4]) == "crash") {
            ir.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::trap));
        } else if (std::string(argv[4]) == "hang") {
            auto *loop = llvm::BasicBlock::Create(context, "loop", main);
            ir.CreateBr(loop);
            ir.SetInsertPoint(loop);
            ir.CreateBr(loop);
            ir.SetInsertPoint(llvm::BasicBlock::Create(context, "unused", main));
        }
        auto *counter = new llvm::GlobalVariable(*module, i64, false,
            llvm::GlobalValue::InternalLinkage, ir.getInt64(0), "counter");
        if (std::string(argv[4]) == "race") {
            auto *entry = ir.GetInsertBlock();
            auto *ptr = ir.getPtrTy();
            auto *null = llvm::ConstantPointerNull::get(ptr);
            auto *writer = llvm::Function::Create(llvm::FunctionType::get(ptr, {ptr}, false),
                llvm::GlobalValue::InternalLinkage, "racy_writer", module.get());
            ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", writer));
            ir.CreateStore(ir.getInt64(42), counter);
            ir.CreateRet(null);
            ir.SetInsertPoint(entry);
            auto *threadType = ir.getIntNTy(sizeof(pthread_t) * 8);
            auto create = module->getOrInsertFunction("pthread_create",
                llvm::FunctionType::get(ir.getInt32Ty(), {ptr, ptr, ptr, ptr}, false));
            auto join = module->getOrInsertFunction("pthread_join",
                llvm::FunctionType::get(ir.getInt32Ty(), {threadType, ptr}, false));
            auto *first = ir.CreateAlloca(threadType);
            auto *second = ir.CreateAlloca(threadType);
            ir.CreateCall(create, {first, null, writer, null});
            ir.CreateCall(create, {second, null, writer, null});
            ir.CreateCall(join, {ir.CreateLoad(threadType, first), null});
            ir.CreateCall(join, {ir.CreateLoad(threadType, second), null});
        }
        auto *value = ir.CreateAdd(ir.CreateLoad(i64, counter), ir.getInt64(1));
        ir.CreateStore(value, counter);
        auto threads = module->getOrInsertFunction("omp_get_max_threads",
            llvm::FunctionType::get(ir.getInt32Ty(), false));
        auto *count = ir.CreateSExt(ir.CreateCall(threads), i64);
        auto *result = ir.CreateInsertValue(llvm::UndefValue::get(resultType), value, 0);
        result = ir.CreateInsertValue(result, count, 1);
        ir.CreateRet(result);
        std::string error;
        auto fn = compileLLVMModuleToFunction(std::move(module), &error, tsan, opt);
        if (!fn) { std::cerr << error << '\n'; return 1; }
        for (int threadCount : {1, 2, 4, 1}) {
            omp_set_num_threads(threadCount);
            auto values = fn();
            if (std::string(argv[4]) == "race") return 0;
            if (values != std::vector<int64_t>{1, threadCount}) {
                std::cerr << "global reset, result capture, or thread agitation differs\n";
                return 1;
            }
        }
    }
    return 0;
}
