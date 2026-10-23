// The compiler libraries alone are built with trace-pc-guard.
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
uint32_t guardCount = 0;
uint8_t *mapping = nullptr;
size_t capacity = 0;
constexpr size_t headerSize = 16;
}

extern "C" void __sanitizer_cov_trace_pc_guard_init(uint32_t *begin, uint32_t *end) {
    if (begin == end || *begin) return;
    for (auto *guard = begin; guard != end; ++guard)
        *guard = __atomic_add_fetch(&guardCount, 1u, __ATOMIC_RELAXED);
}

extern "C" void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {
    const uint32_t id = *guard;
    if (!mapping || !id) return;
    if (id > capacity) {
        __atomic_store_n(mapping + 12, uint8_t{1}, __ATOMIC_RELAXED);
        return;
    }
    // Presence rather than noisy hit counts, with no hash collisions.
    __atomic_store_n(mapping + headerSize + id - 1, uint8_t{1}, __ATOMIC_RELAXED);
}

extern "C" int mracle_coverage_begin(const char *path) {
    if (!path || mapping) return 1;
    const int fd = open(path, O_RDWR);
    if (fd < 0) return 1;
    struct stat status {};
    if (fstat(fd, &status) || status.st_size <= static_cast<off_t>(headerSize)) {
        close(fd); return 1;
    }
    const size_t size = static_cast<size_t>(status.st_size);
    auto *memory = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (memory == MAP_FAILED) return 1;
    mapping = static_cast<uint8_t *>(memory);
    capacity = size - headerSize;
    std::memcpy(mapping, "MRACOV01", 8);
    for (unsigned i = 0; i < 4; ++i) mapping[8 + i] = (guardCount >> (8 * i)) & 255;
    mapping[12] = guardCount > capacity;
    return 0;
}
