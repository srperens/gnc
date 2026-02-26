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