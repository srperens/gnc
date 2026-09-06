#!/usr/bin/env bash
#
# Serve the GNC web player.
#
# Default is plain HTTP on localhost, which is all WebGPU needs — http://localhost is a secure
# context. Use --https for access from another device on the LAN, where it is not.
#
#   ./serve.sh                 → http://localhost:8080/examples/web/player.html
#   ./serve.sh --https         → https://<lan-ip>:8080/... (needs mkcert)
#   ./serve.sh 9000            → a different port
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WEBDIR="$REPO/examples/web"
HTTPS=0
PORT=8080
for arg in "$@"; do
    case "$arg" in
        --https) HTTPS=1 ;;
        ''|*[!0-9]*) echo "Unknown argument: $arg" >&2; exit 1 ;;
        *) PORT="$arg" ;;
    esac
done

command -v node >/dev/null || { echo "node is required" >&2; exit 1; }

# LAN address for testing from another device, used only by --https. Detected, not hardcoded —
# a committed address identifies a network (CLAUDE.md, "Never commit secrets or local
# infrastructure detail"). Override with GNC_LAN_IP.
LAN_IP="${GNC_LAN_IP:-$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')}"
LAN_IP="${LAN_IP:-localhost}"

echo "Building WASM..."
(cd "$REPO" && wasm-pack build --target web --release)

if [ ! -f "$WEBDIR"/range_q75.gnv ]; then
    echo
    echo "No demo files found. Run ./generate_demos.sh to build them."
fi

if [ "$HTTPS" = 1 ]; then
    command -v mkcert >/dev/null || { echo "--https needs mkcert" >&2; exit 1; }
    CERT="$WEBDIR/localhost+1.pem"; KEY="$WEBDIR/localhost+1-key.pem"
    [ -f "$CERT" ] && [ -f "$KEY" ] || (cd "$WEBDIR" && mkcert localhost "$LAN_IP")
    echo; echo "  https://localhost:${PORT}/examples/web/player.html"
    echo "  https://${LAN_IP}:${PORT}/examples/web/player.html"
else
    CERT=""; KEY=""
    echo; echo "  http://localhost:${PORT}/examples/web/player.html"
    echo "  http://localhost:${PORT}/examples/web/index.html   (single frames)"
fi
echo; echo "Ctrl-C to stop."; echo

cd "$REPO" && HTTPS=$HTTPS PORT=$PORT CERT=$CERT KEY=$KEY node -e '
const fs = require("fs"), path = require("path");
const useTls = process.env.HTTPS === "1";
const mod = useTls ? require("https") : require("http");
const MIME = {
  ".html":"text/html", ".js":"application/javascript", ".wasm":"application/wasm",
  ".css":"text/css", ".json":"application/json", ".png":"image/png",
  ".gnv":"application/octet-stream", ".gnv2":"application/octet-stream",
  ".gnc":"application/octet-stream", ".log":"text/plain", ".txt":"text/plain",
};
const opts = useTls
  ? { cert: fs.readFileSync(process.env.CERT), key: fs.readFileSync(process.env.KEY) }
  : {};
const handler = (req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  const file = path.join(".", url === "/" ? "/examples/web/index.html" : url);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end("Not found: " + url); return; }
    res.writeHead(200, {
      "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    res.end(data);
  });
};
const server = useTls ? mod.createServer(opts, handler) : mod.createServer(handler);
server.listen(Number(process.env.PORT), "0.0.0.0",
  () => console.log("Listening on 0.0.0.0:" + process.env.PORT));
'
