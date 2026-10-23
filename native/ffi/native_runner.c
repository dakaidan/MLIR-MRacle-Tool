// Linked with each emitted object. A dedicated descriptor keeps program output
// out of the result protocol. Each request resets the MLIR-MRacle runtime state.
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <omp.h>
#ifdef MRACLE_WITH_TSAN
#include <stdatomic.h>
static _Atomic int raceReported;
// TSan may continue after a report. Preserve it as an exit status before the
// worker retires this persistent child, even with halt_on_error=0.
void __tsan_on_report(void *report) {
    (void)report;
    atomic_store_explicit(&raceReported, 1, memory_order_relaxed);
}
#endif
extern void __mlir_mracle_reset_state(void);
extern void __mlir_mracle_run(int64_t *results);

static int transfer(void *buffer, size_t size, int writing) {
    char *p = buffer;
    while (size) {
        ssize_t n = writing ? send(3, p, size, MSG_NOSIGNAL) : recv(3, p, size, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        size -= (size_t)n;
    }
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    size_t count = (size_t)strtoul(argv[1], NULL, 10);
    if (!count || count > 4096) return 2;
    int64_t *results = calloc(count, sizeof(*results));
    if (!results) return 2;
    uint32_t threads;
    while (!transfer(&threads, sizeof(threads), 0)) {
        if (!threads || threads > 65536) return 2;
        omp_set_dynamic(0);
        omp_set_num_threads((int)threads);
        for (size_t i = 0; i < count; ++i) results[i] = 0;
        __mlir_mracle_reset_state();
        __mlir_mracle_run(results);
#ifdef MRACLE_WITH_TSAN
        if (atomic_load_explicit(&raceReported, memory_order_relaxed)) return 66;
#endif
        if (transfer(results, count * sizeof(*results), 1)) return 2;
    }
    free(results);
    return 0;
}
