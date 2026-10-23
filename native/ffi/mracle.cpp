#include "mracle.h"
#include "mracle_execution.h"
#include "mlir-mracle/pipeline/pipeline.h"
#include "mlir-mracle/execution/execution.h"
#include "mlir-mracle/io/io.h"
#include "mlir-mracle/context/context.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Config/llvm-config.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <dlfcn.h>

extern "C" int mracle_coverage_begin(const char *path);

namespace {
char *copy(const std::string &s) {
    auto *p = static_cast<char *>(std::malloc(s.size() + 1));
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

const char *comparisonName(mlir_mracle::ComparisonMethod method) {
    switch (method) {
    case mlir_mracle::ComparisonMethod::Tvd:
        return "tvd";
    case mlir_mracle::ComparisonMethod::Calibrated:
        return "calibrated";
    case mlir_mracle::ComparisonMethod::Agitation:
        break;
    }
    return "agitation";
}

// Validates and applies the campaign protocol; a null pointer keeps the
// agitation method with its defaults.
bool applyCampaignOptions(const mracle_campaign_options *campaign,
                          mlir_mracle::PipelineOptions &opts) {
    if (!campaign) return true;
    switch (campaign->comparison) {
    case 0:
        return true;
    case 1:
        opts.comparison = mlir_mracle::ComparisonMethod::Tvd;
        break;
    case 2:
        opts.comparison = mlir_mracle::ComparisonMethod::Calibrated;
        break;
    default:
        return false;
    }
    if (campaign->campaigns_initial < 2 ||
        campaign->campaigns_max < campaign->campaigns_initial ||
        campaign->executions_per_campaign <= 0 || campaign->team_size <= 0 ||
        campaign->bootstrap_replicates <= 0 ||
        !(campaign->tvd_threshold > 0.0 && campaign->tvd_threshold <= 1.0) ||
        !(campaign->min_change > 0.0 && campaign->min_change <= 1.0) ||
        !(campaign->confidence_level > 0.0 && campaign->confidence_level < 1.0))
        return false;
    opts.campaignsInitial = campaign->campaigns_initial;
    opts.campaignsMax = campaign->campaigns_max;
    opts.executionsPerCampaign = campaign->executions_per_campaign;
    opts.teamSize = campaign->team_size;
    opts.bootstrapReplicates = campaign->bootstrap_replicates;
    opts.tvdThreshold = campaign->tvd_threshold;
    opts.minChange = campaign->min_change;
    opts.confidenceLevel = campaign->confidence_level;
    return true;
}

// IR structure is additional corpus feedback. Bounded count buckets avoid
// treating every added jitter instruction as a new feature forever.
mlir_mracle::JsonValue structure(const std::string &text) {
    auto json = mlir_mracle::jsonArray();
    if (text.empty()) return json;
    mlir::MLIRContext context;
    mlir_mracle::initializeMLIRContext(context);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module) return json;
    std::map<std::string, unsigned> counts;
    std::set<std::string> features;
    module->walk([&](mlir::Operation *op) {
        const auto name = op->getName().getStringRef().str();
        ++counts[name];
        if (auto *parent = op->getParentOp())
            features.insert("parent:" + parent->getName().getStringRef().str() + ":" + name);
        if (auto order = op->getAttr("memory_order")) {
            std::string printed;
            llvm::raw_string_ostream stream(printed);
            order.print(stream);
            features.insert("order:" + name + ":" + printed);
        }
    });
    for (const auto &[name, count] : counts) {
        unsigned bucket = 1;
        while (bucket < 128 && bucket <= count / 2) bucket *= 2;
        features.insert("count:" + name + ":" + std::to_string(bucket));
    }
    for (const auto &feature : features)
        mlir_mracle::jsonPush(json, mlir_mracle::jsonString(feature));
    return json;
}
}
extern "C" uint32_t mracle_abi_version() { return 3; }
extern "C" void mracle_free(char *p) { std::free(p); }
extern "C" int mracle_run(const char *input, const char *output,
    const char *transform, const char *model, uint32_t seed, int32_t reps,
    int32_t replay_reps, int32_t max_runs, int32_t max_apply,
    uint32_t backend, uint32_t tsan_triage,
    const mracle_campaign_options *campaign, char **result) {
    if (!result) return 1;
    *result = nullptr;
    if (!input || !output || !transform || !model || reps <= 0 ||
        replay_reps <= 0 || max_runs <= 0 || max_apply <= 0 ||
        seed > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        backend > 1 || tsan_triage > 1 ||
        (campaign && campaign->comparison > 2) ||
        (campaign && campaign->comparison != 0 && tsan_triage)) return 1;
    try {
        mracleConfigureExecution(backend == 1, tsan_triage == 1, output);
        if (mracle_coverage_begin(std::getenv("MRACLE_COVERAGE_FILE"))) {
            *result = copy("cannot attach compiler coverage map"); return 2;
        }
        // LLVM may lower OpenMP atomics to these runtime calls. They must be
        // globally visible to LLJIT's current-process symbol resolver.
        if (!dlopen("libatomic.so.1", RTLD_NOW | RTLD_GLOBAL)) {
            *result = copy(dlerror()); return 2;
        }
        mlir_mracle::applyProcessSettings();
        mlir_mracle::PipelineOptions opts;
        opts.inputFile = input; opts.campaignDir = output;
        opts.transform = transform; opts.model = model;
        opts.seed = static_cast<int>(seed); opts.reps = reps;
        opts.retestReps = replay_reps; opts.maxSourceReps = max_runs;
        opts.maxApply = max_apply;
        if (!applyCampaignOptions(campaign, opts)) return 1;
        auto run = mlir_mracle::runPipeline(opts);
        if (run.runs.size() != 1) return 2;
        const auto &info = run.runs.front();
        auto json = mlir_mracle::runInfoToUnionJson(info);
        mlir_mracle::jsonPut(json, "execution_backend", mlir_mracle::jsonString(backend ? "native" : "jit"));
        mlir_mracle::jsonPut(json, "tsan_triage", mlir_mracle::jsonBool(tsan_triage));
        mlir_mracle::jsonPut(json, "comparison", mlir_mracle::jsonString(comparisonName(opts.comparison)));
        mlir_mracle::jsonPut(json, "compiler_version", mlir_mracle::jsonString(LLVM_VERSION_STRING));
        mlir_mracle::jsonPut(json, "transform_applied", mlir_mracle::jsonBool(info.transformApplied));
        mlir_mracle::jsonPut(json, "transformed_mlir", mlir_mracle::jsonString(info.transformedMLIR));
        mlir_mracle::jsonPut(json, "ir_features", structure(info.transformedMLIR));
        std::string text;
        llvm::raw_string_ostream stream(text);
        mlir_mracle::printJson(json, stream);
        *result = copy(text);
        mlir_mracle::shutdownPipeline();
        return *result ? 0 : 3;
    } catch (const std::exception &e) {
        *result = copy(e.what()); return 2;
    } catch (...) {
        *result = copy("unknown native exception"); return 2;
    }
}
