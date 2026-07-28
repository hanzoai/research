// Copyright © 2026 Hanzo AI. MIT License.

//! Zero-config provenance: the commit a result came from, and the box it ran on.
//!
//! Every probe here is best-effort. A missing `git` binary, a non-repo directory or an
//! unreadable environment yields an empty value and never fails an experiment — a result
//! measured without provenance is still a result, and refusing to record it would lose it.

use std::process::Command;
use std::time::Duration;

/// The commit an experiment was measured at.
///
/// `dirty` is load-bearing, not decorative: a number measured on an uncommitted tree is
/// not reproducible, and the plane records that fact rather than implying otherwise.
#[derive(Debug, Clone, Default)]
pub struct Vcs {
    pub sha: String,
    pub branch: String,
    pub dirty: bool,
}

/// The machine a run executed on — enough to correlate a result to hardware.
///
/// A kernel number is meaningless without it: 1.8x on GB10 and 1.8x on a laptop are
/// different claims.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Host {
    pub hostname: String,
    pub platform: String,
}

const GIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Runs `git -C repo args...` and returns trimmed stdout, or an empty string on any error.
fn git(repo: &str, args: &[&str]) -> String {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(repo).args(args);
    // std::process has no timeout; the git probes below are all local metadata reads that
    // return immediately. The constant documents the intended bound for a future async port.
    let _ = GIT_TIMEOUT;
    match cmd.output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => String::new(),
    }
}

impl Vcs {
    /// Reads HEAD's sha, branch and dirtiness for `repo`.
    pub fn detect(repo: &str) -> Self {
        Vcs {
            sha: git(repo, &["rev-parse", "HEAD"]),
            branch: git(repo, &["rev-parse", "--abbrev-ref", "HEAD"]),
            dirty: !git(repo, &["status", "--porcelain"]).is_empty(),
        }
    }
}

impl Host {
    /// Reads the hostname and platform of the current box.
    pub fn detect() -> Self {
        let hostname = std::env::var("HOSTNAME")
            .ok()
            .filter(|h| !h.trim().is_empty())
            .unwrap_or_else(|| {
                match Command::new("hostname").output() {
                    Ok(o) if o.status.success() => {
                        String::from_utf8_lossy(&o.stdout).trim().to_string()
                    }
                    _ => String::new(),
                }
            });
        Host {
            hostname,
            platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
        }
    }
}

/// Walks up from `start` to the nearest directory containing `.git`.
///
/// Falls back to the current directory so provenance capture is never a hard failure.
pub fn find_repo(start: &str) -> String {
    let mut dir = if start.is_empty() {
        std::env::current_dir().unwrap_or_default()
    } else {
        std::path::PathBuf::from(start)
    };
    loop {
        if dir.join(".git").exists() {
            return dir.to_string_lossy().into_owned();
        }
        if !dir.pop() {
            return std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
        }
    }
}

/// Resolves `name` to a version string from the environment.
///
/// A harness records the versions that actually shaped the number — the quant library,
/// the driver, the model revision — under keys it chooses.
pub fn lib_version(name: &str) -> Option<String> {
    let key = format!("{}_VERSION", name.to_uppercase().replace('-', "_"));
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_never_panic_off_a_repo() {
        // The contract that matters: provenance is best-effort. Pointing every probe at a
        // path that is definitely not a git repo must yield empties, not a panic or an error.
        let v = Vcs::detect("/definitely/not/a/repo/9137");
        assert!(v.sha.is_empty());
        assert!(v.branch.is_empty());
        assert!(!v.dirty);
    }

    #[test]
    fn host_always_reports_a_platform() {
        let h = Host::detect();
        assert!(h.platform.contains('/'), "platform is os/arch");
    }

    #[test]
    fn find_repo_falls_back_rather_than_failing() {
        let r = find_repo("/definitely/not/a/repo/9137");
        assert!(!r.is_empty());
    }
}
