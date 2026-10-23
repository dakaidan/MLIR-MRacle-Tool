#include "mracle_execution.h"
#include "runner_bytes.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <omp.h>

extern char **environ;
namespace {
bool useNative = false;
bool useTriage = false;
std::filesystem::path artifactRoot;
unsigned binaryIndex = 0;

std::vector<char *> arguments(std::vector<std::string> &values) {
    std::vector<char *> result;
    for (auto &value : values) result.push_back(value.data());
    result.push_back(nullptr);
    return result;
}
int reap(pid_t pid) {
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("cannot wait for native process");
    }
    return status;
}
bool transfer(int fd, void *buffer, size_t size, bool writing) {
    auto *p = static_cast<char *>(buffer);
    while (size) {
        ssize_t n = writing ? send(fd, p, size, MSG_NOSIGNAL) : recv(fd, p, size, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}
struct NativeProcess {
    pid_t pid = -1;
    int socket = -1;
    unsigned count;
    explicit NativeProcess(const std::string &executable, unsigned results) : count(results) {
        int sockets[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets))
            throw std::runtime_error("cannot create native result channel");
        posix_spawn_file_actions_t actions;
        int rc = posix_spawn_file_actions_init(&actions);
        if (!rc) {
            rc = posix_spawn_file_actions_addclose(&actions, sockets[0]);
            if (!rc) rc = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
            if (!rc && sockets[1] != 3) rc = posix_spawn_file_actions_addclose(&actions, sockets[1]);
            std::vector<std::string> values{executable, std::to_string(count)};
            auto argv = arguments(values);
            if (!rc) rc = posix_spawn(&pid, executable.c_str(), &actions, nullptr, argv.data(), environ);
            posix_spawn_file_actions_destroy(&actions);
        }
        close(sockets[1]);
        if (rc) {
            close(sockets[0]);
            throw std::runtime_error("cannot start native executable: " + std::string(strerror(rc)));
        }
        socket = sockets[0];
    }
    ~NativeProcess() {
        if (socket >= 0) close(socket);
        if (pid > 0) {
            kill(pid, SIGKILL);
            int status;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        }
    }
    std::vector<int64_t> run() {
        uint32_t threads = static_cast<uint32_t>(omp_get_max_threads());
        std::vector<int64_t> results(count);
        if (!transfer(socket, &threads, sizeof(threads), true) ||
            !transfer(socket, results.data(), results.size() * sizeof(int64_t), false)) {
            int status = reap(pid);
            pid = -1;
            // Preserve the actual program failure for Rust's crash/sanitizer
            // classification. The supervisor also kills surviving siblings.
            if (WIFSIGNALED(status)) {
                int signal = WTERMSIG(status);
                std::signal(signal, SIG_DFL);
                raise(signal);
                _exit(128 + signal);
            }
            if (WIFEXITED(status) && WEXITSTATUS(status)) _exit(WEXITSTATUS(status));
            throw std::runtime_error("native executable closed its result channel prematurely");
        }
        return results;
    }
};
}

void mracleConfigureExecution(bool native, bool triage, const std::string &artifacts) {
#ifndef MRACLE_WITH_TSAN
    if (triage) throw std::runtime_error("TSan triage requires a worker built with MRACLE_SANITIZER=thread");
#endif
    useNative = native;
    useTriage = triage;
    artifactRoot = std::filesystem::path(artifacts) / "native-binaries";
    binaryIndex = 0;
    if (native) {
        std::filesystem::create_directories(artifactRoot);
        std::ofstream runner(artifactRoot / "runner.o", std::ios::binary);
        runner.write(reinterpret_cast<const char *>(mracleRunnerObject), sizeof(mracleRunnerObject));
        if (!runner) throw std::runtime_error("cannot extract embedded native runner object");
    }
}
bool mracleUseNative() { return useNative; }
bool mracleTsanTriageEnabled() { return useTriage; }

std::function<std::vector<int64_t>()> mracleCompileNative(
    std::unique_ptr<llvm::Module> module, unsigned resultCount,
    int optLevel, llvm::BasicBlockSection sections, std::string *error) {
    auto fail = [&](const std::string &message) -> std::function<std::vector<int64_t>()> {
        if (error) *error = message;
        return nullptr;
    };
    if (!resultCount || resultCount > 4096) return fail("unsupported native result count");
    auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!builder) return fail(llvm::toString(builder.takeError()));
    if (optLevel >= 0) builder->setCodeGenOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));
    builder->setRelocationModel(llvm::Reloc::PIC_);
    if (builder->getTargetTriple().getObjectFormat() == llvm::Triple::ELF) {
        auto options = builder->getOptions();
        options.BBSections = sections;
        builder->setOptions(options);
    }
    auto target = builder->createTargetMachine();
    if (!target) return fail(llvm::toString(target.takeError()));
    module->setDataLayout((*target)->createDataLayout());
    module->setTargetTriple(builder->getTargetTriple());
    // The C runner owns the process entry point. Existing wrapper calls keep
    // referring to the renamed test function without changing its semantics.
    module->getFunction("main")->setName("__mlir_mracle_program_main");
    auto base = artifactRoot / ("binary-" + std::to_string(binaryIndex++));
    std::string object = base.string() + ".o";
    std::string executable = base.string() + ".exe";
    std::error_code ec;
    llvm::raw_fd_ostream stream(object, ec, llvm::sys::fs::OF_None);
    if (ec) return fail("cannot write native object: " + ec.message());
    llvm::legacy::PassManager passes;
    if ((*target)->addPassesToEmitFile(passes, stream, nullptr, llvm::CodeGenFileType::ObjectFile))
        return fail("target cannot emit a native object");
    passes.run(*module);
    stream.close();
    if (stream.has_error()) return fail("native object write failed");
    std::vector<std::string> command{MRACLE_NATIVE_CC, (artifactRoot / "runner.o").string(), object,
        "-o", executable, "-fopenmp", "-latomic", "-lm"};
#ifdef MRACLE_WITH_TSAN
    command.push_back("-fsanitize=thread");
    command.push_back("-shared-libsan");
    command.push_back("-Wl,-rpath," + std::filesystem::path(MRACLE_TSAN_RUNTIME).parent_path().string());
#endif
    std::ofstream recipe(base.string() + ".link.txt");
    recipe << "codegen_opt_level=" << optLevel << '\n';
    for (const auto &argument : command) recipe << argument << '\n';
    recipe.close();
    auto argv = arguments(command);
    pid_t linker;
    int rc = posix_spawn(&linker, command[0].c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc) return fail("cannot run native linker: " + std::string(strerror(rc)));
    int status = reap(linker);
    if (!WIFEXITED(status) || WEXITSTATUS(status))
        return fail("native link failed; see worker stderr and " + base.string() + ".link.txt");
    auto process = std::make_shared<NativeProcess>(executable, resultCount);
    return [process] { return process->run(); };
}
