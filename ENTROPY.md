8 alternativ, rankade

  ┌──────────────┬──────────┬──────────┬─────────┬─────────┐
  │              │ bpp-over │ Parallel │ Komplex │ Patent- │
  │    Metod     │ head vs  │ lism per │  itet   │  risk   │
  │              │   rANS   │   tile   │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Significance │          │ ~10 000  │         │         │
  │  map +       │ 3-7%     │ (max)    │ Låg     │ Ingen   │
  │ Golomb-Rice  │          │          │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Canonical    │          │ ~12 000  │         │         │
  │ Huffman (per │ 1-5%     │ (max)    │ Medel   │ Ingen   │
  │ -subband)    │          │          │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Zero-run +   │          │          │         │         │
  │ Huffman      │ 2-5%     │ ~6 000   │ Medel   │ Ingen   │
  │ hybrid       │          │          │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Huffman +    │          │          │         │         │
  │ raw          │ 5-10%    │ ~12 000  │ Låg     │ Ingen   │
  │ refinement   │          │          │         │         │
  │ bits         │          │          │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Vektor-kvant │ 10-20%   │ ~3 000   │ Medel   │ Låg     │
  │ isering      │          │          │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ ANS 256      │ 8-15%    │ 256      │ Låg     │ Medel-H │
  │ strömmar     │          │          │         │ ög      │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Neural       │ -10%     │ ~12 000  │ Extrem  │ Medel   │
  │ entropy      │ till +5% │          │         │         │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ ANS prefix   │ 0% (iden │ ~12 000  │ Extrem  │ Medel-H │
  │ scan         │ tisk)    │ (teori)  │         │ ög      │
  ├──────────────┼──────────┼──────────┼─────────┼─────────┤
  │ Bitplane     │ 0-5%     │ ~30-300  │ Hög     │ Medel   │
  │ (EBCOT)      │          │ (dålig)  │         │         │
  └──────────────┴──────────┴──────────┴─────────┴─────────┘

  Överraskning: Microsoft har patent (US11234023B2) på
  rANS-modifikationer. Vår nuvarande kodare har potentiell
  patentrisk.

  Topp 3

  1. Significance map + Golomb-Rice — Enklast. Separerar
  zero/non-zero (binär map) från magnituder (Golomb-Rice). Varje
   koefficient kodas oberoende. ~20 rader shader-kod. Perfekt
  match för wavelet-data (geometrisk fördelning).

  2. Canonical Huffman — Mest forskat på GPU. Fast codeword per
  symbol, ingen state-dependency. Prefix-sum för bit-packing.
  Bäst känd prestanda på GPU (multi-GB/s i papers).

  3. Zero-run + Huffman — Närmast rANS i kvalitet (2-5%). Kodar
  (run-length, värde)-par med Huffman. Rader oberoende →
  parallella.

  Vad det betyder för prestanda

  Om entropy coding tar 85% av 50ms = 42.5ms, och vi ersätter
  med en fullt parallell kodare:

  Parallell entropy borde ta ungefär samma tid som
  wavelet+kvantisering (~7ms), inte 6x mer. Konservativt:

  ┌──────────────┬────────────────┬───────────────────────┐
  │              │      Idag      │ Med parallell entropy │
  ├──────────────┼────────────────┼───────────────────────┤
  │ 1080p encode │ 50ms / 20 fps  │ ~15ms / 67 fps        │
  ├──────────────┼────────────────┼───────────────────────┤
  │ 4K encode    │ 144ms / 7 fps  │ ~40ms / 25 fps        │
  ├──────────────┼────────────────┼───────────────────────┤
  │ iPhone 720p  │ ~35ms / 28 fps │ ~10ms / 100 fps       │
  └──────────────┴────────────────┴───────────────────────┘

  Och — vi slipper Microsofts rANS-patent.

---

## Resultat: Rice GPU (2026-02-27)

  Implementerade #1 (Significance map + Golomb-Rice) med GPU-shader.
  256 interleaved streams per tile, fullt parallella.

  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │  rANS    │ Rice GPU │ Diff     │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ Encode 1080p │ 34ms     │ 21ms     │ 1.6x     │
  │              │ 29 fps   │ 46 fps   │ snabbare │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ Decode 1080p │ 29ms     │ 14ms     │ 2.0x     │
  │              │ 34 fps   │ 69 fps   │ snabbare │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BPP q=75     │ 4.22     │ 6.04     │ +43%     │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BPP q=90     │          │ ~match   │ ~0%      │
  └──────────────┴──────────┴──────────┴──────────┘

  Slutsats: Hastighetsförbättringen validerar tesen — att ta
  bort rANS sekventiella state chain (2048 steg → 256 steg
  per tråd) ger massiva GPU-parallellism-vinster.

  Kompressionsgapet vid låga bitrates beror på per-koefficient
  significance bits. ZRL (zero-run-length) skulle minska detta,
  men bättre approach: Canonical Huffman som anpassar sig till
  verklig fördelning istället för att anta geometrisk.

## Analys: Nästa steg — Canonical Huffman

  Av de 8 alternativen sticker Canonical Huffman ut:

  ┌─────────────────┬────────┬─────────┬──────────┐
  │                 │ Rice   │ Huffman │ rANS     │
  ├─────────────────┼────────┼─────────┼──────────┤
  │ BPP overhead    │ 3-7%   │ 1-5%   │ baseline │
  ├─────────────────┼────────┼─────────┼──────────┤
  │ Parallellism    │ 10 000 │ 12 000 │ 32       │
  ├─────────────────┼────────┼─────────┼──────────┤
  │ State chain     │ Ingen  │ Ingen  │ 2048/trd │
  ├─────────────────┼────────┼─────────┼──────────┤
  │ Patentrisk      │ Ingen  │ Ingen  │ Medel-Hög│
  ├─────────────────┼────────┼─────────┼──────────┤
  │ Shared mem      │ <1KB   │ 8KB    │ 16KB+    │
  ├─────────────────┼────────┼─────────┼──────────┤
  │ GPU dispatches  │ 1      │ 2      │ 3        │
  └─────────────────┴────────┴─────────┴──────────┘

  Varför Huffman vinner:
  - Bättre kompression: variabel kodlängd anpassar sig till
    verklig symbolfördelning, inte bara geometrisk (Rice)
  - Maximal parallellism: varje symbol → fast codeword lookup
  - Snabb decode: 8-bit prefix table lookup (O(1) per symbol)
  - Ingen patentrisk (public domain sedan 1952)
  - 8KB shared mem → full M1 occupancy (2 WG/core)

  Nackdel vs Rice:
  - 2 GPU dispatches istället för 1 (histogram + encode)
  - CPU codebook-bygge mellan dispatches (~<1ms)
  - Mer komplex decoder (prefix table vs enkel Rice formula)

  Förväntat resultat:
  - Hastighet: ~samma som Rice (kanske 10-20% långsammare pga
    extra dispatch + codebook roundtrip)
  - Kompression: ~1-5% overhead vs rANS (vs Rice 43% vid q=75)
  - Netto: dramatiskt bättre speed/compression tradeoff

  Implementation plan: se huffman_plan.md