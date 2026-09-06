#!/usr/bin/env bash
# Serve the GNC/GNV web player over HTTPS (mkcert). Run from anywhere.
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${1:-8080}"
WEBDIR="$REPO/examples/web"
CERT="$WEBDIR/localhost+1.pem"
KEY="$WEBDIR/localhost+1-key.pem"

# LAN address for testing from another device. Detected, not hardcoded — a committed
# address identifies a network (CLAUDE.md, "Never commit secrets or local infrastructure
# detail"). Override with GNC_LAN_IP, or leave unset for localhost only.
LAN_IP="${GNC_LAN_IP:-$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')}"

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
    echo "No TLS certs found. Generating with mkcert..."
    (cd "$WEBDIR" && mkcert localhost ${LAN_IP:+"$LAN_IP"})
fi

echo "Building WASM..."
(cd "$REPO" && wasm-pack build --target web --release)

echo ""
echo "Serving at:"
echo "  https://localhost:${PORT}/examples/web/player.html"
[ -n "$LAN_IP" ] && echo "  https://${LAN_IP}:${PORT}/examples/web/player.html"
echo ""
echo "Press Ctrl-C to stop."

cd "$REPO" && node -e "
const https = require('https');
const fs = require('fs');
const path = require('path');

const MIME = {
  '.html':'text/html','.js':'application/javascript','.wasm':'application/wasm',
  '.css':'text/css','.json':'application/json','.png':'image/png',
  '.gnv':'application/octet-stream','.gnv2':'application/octet-stream',
  '.gnc':'application/octet-stream','.diag':'text/plain','.txt':'text/plain',
};

const server = https.createServer({
  cert: fs.readFileSync('${CERT}'),
  key: fs.readFileSync('${KEY}'),
}, (req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  const file = path.join('.', url === '/' ? '/index.html' : url);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(file);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(data);
  });
});

server.listen(${PORT}, '0.0.0.0', () => console.log('Listening on 0.0.0.0:${PORT}'));
"
