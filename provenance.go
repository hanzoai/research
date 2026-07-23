// Copyright © 2026 Hanzo AI. MIT License.

package research

// Zero-config provenance auto-capture — like OpenTelemetry auto-instrument, for
// experiments. The caller supplies nothing; the SDK reads the run's environment: git
// sha/branch/dirty, the recent commit messages (the narrative of what changed), the
// Go toolchain + module versions, the host, and the calling code site.

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

// gitTimeout bounds every git subprocess so a pathological repo (huge tree, slow
// filesystem) degrades to empty provenance instead of stalling the experiment.
const gitTimeout = 5 * time.Second

// vcs is the producing repo's git identity at run time: commit sha, branch, and whether
// the working tree was dirty.
type vcs struct {
	SHA    string
	Branch string
	Dirty  bool
}

// box is the machine a run executed on — enough to correlate a result to hardware.
type box struct {
	Hostname string `json:"hostname"`
	Platform string `json:"platform"`
}

// git runs `git -C repo args...` and returns trimmed stdout, or "" on any error.
// Provenance is best-effort: a missing binary or non-repo never fails the experiment.
func git(repo string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), gitTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "git", append([]string{"-C", repo}, args...)...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// gitState reads the producing repo's commit sha, branch, and dirty flag.
func gitState(repo string) vcs {
	return vcs{
		SHA:    git(repo, "rev-parse", "HEAD"),
		Branch: git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
		Dirty:  git(repo, "status", "--porcelain") != "",
	}
}

// commitNarrative is the commit-message story: subjects SINCE the last recorded run's sha
// when known (<since>..HEAD), else the last window commits. Always non-nil so it serializes
// to an empty array [], not null — the shape every producer sends and the server expects.
func commitNarrative(repo, since string, window int) []string {
	var raw string
	// since can arrive from a server response; only a git object id (hex) reaches the git
	// arg, so it can never be read as a flag or a path — anything else falls back to window.
	if isObjectID(since) {
		raw = git(repo, "log", since+"..HEAD", "--format=%s")
	} else {
		raw = git(repo, "log", "-"+strconv.Itoa(window), "--format=%s")
	}
	out := []string{}
	for _, ln := range strings.Split(raw, "\n") {
		if strings.TrimSpace(ln) != "" {
			out = append(out, ln)
		}
	}
	return out
}

// isObjectID reports whether s is a git object id — hex only, 1..64 chars (sha1=40,
// sha256=64). The since value can arrive from a server response and reaches
// `git log <since>..HEAD`; gating it to hex means it can never be read as a flag (a leading
// '-') or a path ('/'), which closes the git arg-injection. Parity with the C++ port.
func isObjectID(s string) bool {
	if s == "" || len(s) > 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// libVersions is the Go-native {module: version} snapshot from the build info — the
// longitudinal "which lib version regressed X" record. It always names the Go toolchain
// and the main module; extra module-path prefixes pull in named dependencies. Always
// non-nil so it serializes to an empty object {}, not null — the shape every producer sends.
func libVersions(extra []string) map[string]string {
	out := map[string]string{"go": runtime.Version()}
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return out
	}
	if bi.Main.Path != "" && bi.Main.Version != "" {
		out[bi.Main.Path] = bi.Main.Version
	}
	for _, d := range bi.Deps {
		for _, p := range extra {
			if strings.HasPrefix(d.Path, p) {
				out[d.Path] = d.Version
				break
			}
		}
	}
	return out
}

// host reads the box the run executed on.
func host() box {
	name, _ := os.Hostname()
	return box{Hostname: name, Platform: runtime.GOOS}
}

// findRepo is the git toplevel the caller runs in, auto-detected from a start dir
// (default the process working directory). Falls back to the start dir when not a repo.
func findRepo(start string) string {
	if start == "" {
		if wd, err := os.Getwd(); err == nil {
			start = wd
		}
	}
	if top := git(start, "rev-parse", "--show-toplevel"); top != "" {
		return top
	}
	return start
}

// callerDoc is the code site that ran the experiment — the first stack frame OUTSIDE this
// package, as "<import-path>.<Func>". The Go-native analog of the caller's module
// docstring: zero-config attribution of a record to its producing code, which matters
// when every team across the company logs into the one store.
func callerDoc() string {
	pcs := make([]uintptr, 32)
	n := runtime.Callers(1, pcs) // frame 0 is callerDoc itself — this package
	if n == 0 {
		return ""
	}
	frames := runtime.CallersFrames(pcs[:n])
	self := ""
	for {
		fr, more := frames.Next()
		p := pkgOf(fr.Function)
		switch {
		case self == "":
			self = p // learn this SDK's import path from callerDoc's own frame
		case p != self && fr.Function != "":
			return fr.Function
		}
		if !more {
			break
		}
	}
	return ""
}

// pkgOf extracts the import path from a runtime function name:
// "github.com/hanzoai/x.Fn" -> "github.com/hanzoai/x"; "pkg.(*T).M" -> "pkg".
func pkgOf(fn string) string {
	slash := strings.LastIndex(fn, "/")
	dot := strings.Index(fn[slash+1:], ".")
	if dot < 0 {
		return fn
	}
	return fn[:slash+1+dot]
}
