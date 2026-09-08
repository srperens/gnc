#!/usr/bin/env python3
"""BUG-48 gate: only q=100 MED may move; everything else must be byte-identical.

    scripts/gate_bug48.py <before-binary> <after-binary>

Two binaries rather than one binary with `GNC_PAD_FILL`, because the env var forces the fill on
*every* path at once and this gate's whole question is which paths changed. Build the before arm
by stashing the change; both must come from the same tree otherwise (see COORDINATION, "ask which
tree").

40 encodes, ~4 minutes. Exits by printing PASS or the number of regressions; it does not set a
non-zero status, because a moved cell at q=100 is the intended change and only the caller knows
which direction was intended.
"""
import os, subprocess, sys, tempfile, pathlib
BEFORE, AFTER = sys.argv[1], sys.argv[2]
REPO = subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
STILLS = ["bbb_1080p","blue_sky_1080p","kristensara_720p","touchdown_1080p"]

def enc(binp,img,q,env_extra=None):
    with tempfile.NamedTemporaryFile(suffix=".gnc",delete=False) as t: out=t.name
    env = dict(os.environ, **(env_extra or {}))
    subprocess.run([binp,"encode","-i",f"{REPO}/test_material/frames/{img}.png","-o",out,"-q",str(q)],
                   env=env,capture_output=True,check=True)
    n=pathlib.Path(out).stat().st_size; os.unlink(out); return n

def seq(binp,name,q,ki,n=4,env_extra=None):
    with tempfile.NamedTemporaryFile(suffix=".gnv",delete=False) as t: out=t.name
    env = dict(os.environ, **(env_extra or {}))
    subprocess.run([binp,"encode-sequence","-i",f"{REPO}/test_material/frames/sequences/{name}/frame_%04d.png",
                    "-o",out,"-q",str(q),"--keyframe-interval",str(ki),"-n",str(n)],
                   env=env,capture_output=True,check=True)
    sz=pathlib.Path(out).stat().st_size; os.unlink(out); return sz

bad=0
print("=== stills: q=100 (MED) is the only cell allowed to move ===")
for q in [85,90,95,97,99,100]:
    for img in STILLS:
        b=enc(BEFORE,img,q); a=enc(AFTER,img,q)
        flag = "MOVED" if a!=b else "same"
        if a!=b and q!=100: bad+=1; flag="*** REGRESSION ***"
        print(f"  q={q:<4}{img:<18} {b:>9} -> {a:>9} {100*(a-b)/b:+7.3f}%  {flag}")
print("=== stills: q=100 with GNC_MED=0 (lossless wavelet) must NOT move ===")
for img in STILLS:
    b=enc(BEFORE,img,100,{"GNC_MED":"0"}); a=enc(AFTER,img,100,{"GNC_MED":"0"})
    if a!=b: bad+=1
    print(f"  {img:<18} {b:>9} -> {a:>9}  {'same' if a==b else '*** REGRESSION ***'}")
print("=== sequences must NOT move (the encoder clears the flag itself) ===")
for name in ["crowd_run","bbb"]:
    for q,ki in [(99,2),(99,9),(100,2),(100,9)]:
        b=seq(BEFORE,name,q,ki); a=seq(AFTER,name,q,ki)
        if a!=b: bad+=1
        print(f"  {name:<10} q={q} ki={ki}: {b:>9} -> {a:>9}  {'same' if a==b else '*** REGRESSION ***'}")
print(f"\n{'PASS' if bad==0 else str(bad)+' REGRESSIONS'}")
