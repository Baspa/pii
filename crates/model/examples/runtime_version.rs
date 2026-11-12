//! Prints the ONNX Runtime build the pinned `ort` crate actually links.
//!
//! The spec's latency table was measured against "ONNX
//! Runtime 1.29, CPU execution provider, INT8". A comparison against those
//! numbers is only meaningful on a comparable runtime, so this is run once
//! when the `ort` dependency is pinned or bumped, and the reported version is
//! recorded next to `onnx_runtime_version_at_export` wherever this task's
//! findings are written up.
//!
//! `ort::info()` is the accessor `ort` 2.0.0-rc.13 exposes for this: it
//! returns the linked runtime's own build info string (git branch, commit,
//! build type), not `ort::MINOR_VERSION`, which is only the ONNX Runtime API
//! version `ort` was compiled against and does not name a release.
fn main() {
    println!("{}", ort::info());
}
