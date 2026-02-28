#!/usr/bin/env bash
# Serve the GNC/GNV web player. Run from anywhere.
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${1:-8080}"

echo "Building WASM..."
(cd "$REPO" && wasm-pack build --target web --release)

echo ""
echo "Serving at:"
echo "  http://localhost:${PORT}/examples/web/player.html"
echo ""
echo "Press Ctrl-C to stop."

cd "$REPO" && python3 -m http.server "$PORT" --bind 0.0.0.0
