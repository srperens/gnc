# GNC Multi-Agent Team

## Team Members

| Agent | Role | Worktree | Model | Purpose |
|-------|------|----------|-------|---------|
| debugger | Find bugs | debug/mc-divergence | minimax-m2.5-free | Isolate P-frame MC bug via diagnostic logging |
| optimizer (mv-entropy) | Optimize | optimize/mv-entropy | minimax-m2.5-free | Reduce MV data cost (target: <82KB per P-frame) |
| optimizer (residual-quant) | Optimize | optimize/residual-quant | minimax-m2.5-free | Tune residual quantization |
| analyst | Analyze | (main) | big-pickle | Compare runs, report metrics |

## Execution Order

1. **debugger** — First, isolate the P-frame motion compensation bug
   - Add diagnostic logging to trace encoder/decoder divergence
   - Target: encoder vs decoder reference diff = 0.00
   - Must NOT modify codec logic

2. **optimizer (mv-entropy)** — Second, optimize motion vector encoding
   - Reduce MV data cost from 82-84 KB per P-frame
   - Target: MV data as % of frame size minimized

3. **optimizer (residual-quant)** — Third, tune residual quantization
   - Improve Y/Co/Cg near_zero %
   - Target: P-frame ratio < 0.5, bits/coeff minimized

4. **analyst** — After each agent completes, verify metrics

## Test Sequence

Big Buck Bunny frames 816-825

## Target Metrics

| Metric | Target |
|--------|--------|
| P-frame ratio vs I-frame | < 0.5 |
| Y/Co/Cg near_zero % | > 50% |
| bits/coeff | minimize |
| MV data as % of frame size | minimize |
| Encoder vs decoder reference diff | 0.00 |

## Worktree Layout

```
gnc/                          # main (analyst)
├── .opencode/agents/         # agent definitions
├── AGENTS.md                 # this file
└── run-team.sh               # setup script

../gnc-debug/                 # debug/mc-divergence
../gnc-mv-entropy/            # optimize/mv-entropy
../gnc-residual-quant/        # optimize/residual-quant
```
