#include <cstdlib>
#include <csignal>
extern "C" int mracle_coverage_begin(const char *path);
static volatile int observed;
__attribute__((noinline)) static void left() { observed = 42; }
__attribute__((noinline)) static void right() { observed = 17; }
int main(int argc, char **argv) {
    if (mracle_coverage_begin(std::getenv("MRACLE_COVERAGE_FILE"))) return 2;
    const int input = argc > 1 ? std::atoi(argv[1]) : 0;
    if (input == 1) left();
    else right();
    if (input == 2) std::raise(SIGKILL);
    return 0;
}
