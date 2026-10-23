#pragma once
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetOptions.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Configured once by the adapter in each fresh worker process.
void mracleConfigureExecution(bool native, bool triage, const std::string &artifacts);
bool mracleUseNative();
bool mracleTsanTriageEnabled();
std::function<std::vector<int64_t>()> mracleCompileNative(
    std::unique_ptr<llvm::Module> module, unsigned resultCount,
    int optLevel, llvm::BasicBlockSection sections, std::string *error);
