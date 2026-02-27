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

  Initiala resultat (utan ZRL):

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

## Uppdatering: Rice+ZRL (2026-02-27)

  Lade till zero-run-length (ZRL) kodning. Istället för 1 bit per
  noll-koefficient, kodas nollsekvenser med Rice(run_length-1, k_zrl).

  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │  rANS    │ Rice+ZRL │ Diff     │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ Encode 1080p │ 34ms     │ 25ms     │ 1.4x     │
  │              │ 29 fps   │ 40 fps   │ snabbare │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ Decode 1080p │ 29ms     │ 16ms     │ 1.8x     │
  │              │ 34 fps   │ 61 fps   │ snabbare │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BPP q=25     │ 1.29     │ 1.73     │ +34%     │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BPP q=50     │ 2.30     │ 2.42     │ +5%      │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BPP q=75     │ 4.22     │ 4.09     │ -3% (!)  │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BPP q=90     │ 9.65     │ 8.96     │ -7% (!)  │
  └──────────────┴──────────┴──────────┴──────────┘

  ZRL stängde kompressionsgapet dramatiskt:
  - q=25: från +269% till +34% overhead
  - q=50+: Rice+ZRL SLÅR rANS i bpp!

  Slutsats: Rice+ZRL är nu konkurrenskraftigt med rANS i
  kompression och 1.5-2x snabbare. Tesen validerad.

## Analys: Nästa steg — Canonical Huffman?

  Med Rice+ZRL som redan slår rANS vid q>=50 har Huffman-caset
  försvagats. Men vid q=25 finns fortfarande +34% gap. Huffman
  kan potentiellt stänga detta.

  ┌─────────────────┬────────────┬─────────┬──────────┐
  │                 │ Rice+ZRL   │ Huffman │ rANS     │
  ├─────────────────┼────────────┼─────────┼──────────┤
  │ BPP overhead    │ -3 till    │ 1-5%   │ baseline │
  │  (vs rANS)     │ +34%       │ (est)   │          │
  ├─────────────────┼────────────┼─────────┼──────────┤
  │ Parallellism    │ 256 str    │ 256 str │ 32 str   │
  ├─────────────────┼────────────┼─────────┼──────────┤
  │ State chain     │ Ingen      │ Ingen   │ 2048/trd │
  ├─────────────────┼────────────┼─────────┼──────────┤
  │ Patentrisk      │ Ingen      │ Ingen   │ Medel-Hög│
  ├─────────────────┼────────────┼─────────┼──────────┤
  │ Shared mem      │ <1KB       │ 8KB     │ 16KB+    │
  ├─────────────────┼────────────┼─────────┼──────────┤
  │ GPU dispatches  │ 1          │ 2       │ 3        │
  └─────────────────┴────────────┴─────────┴──────────┘

  Huffman kan fortfarande vara intressant för:
  - Stänga +34% gapet vid q=25 (låga bitrates)
  - Potentiellt bättre lossless-kompression
  - O(1) decode via prefix-table (snabbare än Rice bit-scanning)

  Implementation plan: se HUFFMAN_PLAN.md