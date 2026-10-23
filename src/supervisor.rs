use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::{
    fs::File,
    io,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Timeout,
    Crash(i32),
    Error(i32),
}

pub fn execute(
    worker: &Path,
    request: &Path,
    artifacts: &Path,
    timeout: Duration,
) -> io::Result<Outcome> {
    crate::coverage::prepare(artifacts)?;
    let mut command = Command::new(worker);
    command
        .arg(request)
        .env("MLIR_MRACLE_CACHE_DIR", "")
        .env("MRACLE_COVERAGE_FILE", artifacts.join("coverage.map"))
        // LLVM Archer recommends this for an uninstrumented OpenMP runtime.
        .env(
            "TSAN_OPTIONS",
            std::env::var("TSAN_OPTIONS")
                .unwrap_or_else(|_| "ignore_noninstrumented_modules=1".into()),
        )
        .stdin(Stdio::null())
        .stdout(File::create(artifacts.join("stdout.log"))?)
        .stderr(File::create(artifacts.join("stderr.log"))?)
        .process_group(0);
    // Core dumping a large JIT process can outlast the timeout and turn a
    // compiler crash into an apparent hang. Artifacts and stderr are retained.
    unsafe {
        command.pre_exec(|| {
            let limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &limit) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn()?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            // Also reap any surviving target descendants from this isolated group.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            return Ok(if status.success() {
                Outcome::Completed
            } else if let Some(signal) = status.signal() {
                Outcome::Crash(signal)
            } else {
                Outcome::Error(status.code().unwrap_or(-1))
            });
        }
        if start.elapsed() >= timeout {
            // SAFETY: the child owns its new process group; do not kill the campaign.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            child.wait()?;
            return Ok(Outcome::Timeout);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_exit_and_timeout_are_distinct() {
        let dir = std::env::temp_dir().join(format!("mracle-supervisor-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("request");
        for (text, expected) in [
            ("exit 0", Outcome::Completed),
            ("exit 2", Outcome::Error(2)),
            ("kill -KILL $$", Outcome::Crash(9)),
            ("sleep 10", Outcome::Timeout),
        ] {
            std::fs::write(&script, text).unwrap();
            assert_eq!(
                execute(
                    Path::new("/bin/sh"),
                    &script,
                    &dir,
                    Duration::from_millis(100)
                )
                .unwrap(),
                expected
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
