#!/usr/bin/env bash
# BUG-25 interestingness test for `spirv-reduce`.
#
# `spirv-reduce` shrinks a module while a user-supplied test keeps saying "still interesting".
# Here interesting means **the driver dies**, so the mapping is:
#
#   exit 0  -> the module still kills the driver          (keep this reduction)
#   exit 1  -> the pipeline was created, or wgpu refused  (discard this reduction)
#
# A driver crash kills the probe with a signal, and bash reports that as 128+signo (139 for
# SIGSEGV, 134 for SIGABRT). Those become exit 0 here. Anything else — a clean pipeline (1), a
# wgpu error (2), a missing binary — is not interesting.
#
# The trap this guards against: a reduction that wgpu *rejects* also fails to create a pipeline,
# and if "no pipeline" counted as interesting the reducer would happily reduce to garbage that
# never reaches the driver. That is the SPIR-V-level form of the mistake the WGSL bisect made,
# where four naga-rejected variants were counted as passes. Only a signal counts here.
#
# Usage (from the repository root):
#   spirv-reduce in.spv -o out.spv -- scripts/bug25_interesting.sh
# The reducer appends the candidate module path as $1.

set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROBE="$REPO/target/release/examples/spirv_pipeline_probe"
CANDIDATE=${1:?usage: bug25_interesting.sh <module.spv>}
ENTRY=${BUG25_ENTRY:-main}

if [ ! -x "$PROBE" ]; then
    echo "build the probe first: cargo build --release --features bug25-spirv --example spirv_pipeline_probe" >&2
    exit 1
fi

# A module the reducer has broken structurally is not evidence about the driver. Rejecting it here
# keeps the reduction on modules that are still valid SPIR-V, which is the property that made this
# bug interesting in the first place.
if command -v spirv-val >/dev/null 2>&1; then
    spirv-val "$CANDIDATE" >/dev/null 2>&1 || exit 1
fi

WGPU_BACKEND=${WGPU_BACKEND:-vulkan} "$PROBE" "$CANDIDATE" "$ENTRY" >/dev/null 2>&1
rc=$?

# 128+signo means the driver took the process down with it: that is the bug.
if [ "$rc" -ge 128 ]; then
    exit 0
fi
exit 1
