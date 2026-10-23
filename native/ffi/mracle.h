#ifndef MRACLE_FFI_H
#define MRACLE_FFI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Campaign protocol of the distributional comparison methods. Only the fields
 * of the selected method are read: the agitation method ignores all of them. */
typedef struct {
    uint32_t comparison;             /* 0 agitation, 1 tvd, 2 calibrated */
    int32_t campaigns_initial;       /* campaigns before the first verdict */
    int32_t campaigns_max;           /* cap while the verdict is inconclusive */
    int32_t executions_per_campaign; /* executions per campaign and variant */
    int32_t team_size;               /* OpenMP team size of every execution */
    int32_t bootstrap_replicates;    /* resamples of the uncertainty estimate */
    double tvd_threshold;            /* practical tolerance of the tvd method */
    double min_change;               /* calibrated method's practical floor */
    double confidence_level;         /* confidence of the interval estimate */
} mracle_campaign_options;

/* ABI v3. Strings are UTF-8, NUL-terminated, borrowed for the call.
 * Calls are single-threaded, one per fresh worker process. Result is owned
 * by this library and must be released with mracle_free, even on failure.
 * Native crashes/abort/exit are isolated by the Rust campaign supervisor. */
uint32_t mracle_abi_version(void);
int mracle_run(const char *input, const char *output, const char *transform,
               const char *model, uint32_t seed, int32_t reps,
               int32_t replay_reps, int32_t max_runs, int32_t max_apply,
               uint32_t backend, uint32_t tsan_triage,
               const mracle_campaign_options *campaign, char **result);
void mracle_free(char *result);
#ifdef __cplusplus
}
#endif
#endif
