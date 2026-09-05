#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== GNC Multi-Agent Team Setup ==="

# Check worktrees exist
if [ ! -d "../gnc-debug" ]; then
    echo "Creating debug/mc-divergence worktree..."
    git worktree add -b debug/mc-divergence ../gnc-debug
fi

if [ ! -d "../gnc-mv-entropy" ]; then
    echo "Creating optimize/mv-entropy worktree..."
    git worktree add -b optimize/mv-entropy ../gnc-mv-entropy
fi

if [ ! -d "../gnc-residual-quant" ]; then
    echo "Creating optimize/residual-quant worktree..."
    git worktree add -b optimize/residual-quant ../gnc-residual-quant
fi

echo ""
echo "=== Launching agents in tmux sessions ==="

# Create tmux session for debugger
tmux new-session -d -s gnc-debugger -c "$SCRIPT_DIR/../gnc-debug"
tmux send-keys -t gnc-debugger "opencode --agent debugger --model opencode/minimax-m2.5-free --prompt 'Add diagnostic logging to trace encoder/decoder P-frame divergence. Target: Big Buck Bunny frames 816-825. Find the bug causing Y mean=1.78 diff between encoder and decoder references. Never modify codec logic, only add instrumentation.'" C-M

# Create tmux session for MV entropy optimizer
tmux new-session -d -s gnc-mv-entropy -c "$SCRIPT_DIR/../gnc-mv-entropy"
tmux send-keys -t gnc-mv-entropy "opencode --agent optimizer --model opencode/minimax-m2.5-free --prompt 'Optimize motion vector encoding to reduce MV data cost from 82-84KB per P-frame. Target: minimize MV data as % of frame size. Test sequence: Big Buck Bunny frames 816-825. Always run baseline before making changes.'" C-M

# Create tmux session for residual quant optimizer
tmux new-session -d -s gnc-residual-quant -c "$SCRIPT_DIR/../gnc-residual-quant"
tmux send-keys -t gnc-residual-quant "opencode --agent optimizer --model opencode/minimax-m2.5-free --prompt 'Tune residual quantization to improve Y/Co/Cg near_zero % (target: >50%) and achieve P-frame ratio < 0.5. Test sequence: Big Buck Bunny frames 816-825. Always run baseline before making changes.'" C-M

echo ""
echo "Sessions created:"
tmux list-sessions -F "#{session_name}"
echo ""
echo "Attach with: tmux attach -t <session-name>"
echo "  gnc-debugger"
echo "  gnc-mv-entropy"
echo "  gnc-residual-quant"
