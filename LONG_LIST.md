# GNC Backlog

Baseline: v0.1-spatial (commit 617d8e6) — spatial-only, I+P+B default, temporal Haar experimentellt.

---

## Kompression (bpp)

- [ ] **4:2:0 chroma subsampling** — ~15-25% bpp-vinst, breaking bitstream-ändring, gör i ett svep med 10-bit
- [ ] **CfL (Chroma-from-Luma) i temporal mode** — avstängd/ej portad till temporal path, koppla in
- [ ] **Adaptiv highpass-kvantiering per tile** — weight map finns men AQ weight map kostar 2.5 KB overhead på nästan tomma highpass-frames; skippa weight map om >80% all-skip tiles
- [ ] **LL subband-prediktion** — 0.39% av koefficienter men energität; testa delta-prediktion mellan angränsande LL-tiles; skippa om vinst <2% bpp
- [ ] **Rice-entropi: ~30% redundans kvar** — identifierat men ej åtgärdat
- [ ] **Tile header-komprimering för sparse frames** — present-bitmask istället för full header när all_skip_tiles > 0

## Prestanda (fps)

- [ ] **Profilera encode-flaskhalsen** — 20 fps på crowd_run/park_joy är för långsamt för broadcast; okänt om flaskhalsen är spatial wavelet, Rice-entropi eller GPU-dispatch overhead
- [ ] **Tile-storleksexperiment** — 256×256 är nuvarande default; mindre tiles (128×128, 64×64) kan ge bättre AQ-granularitet men ökar header-overhead; kräver benchmark-sviten för meningsfull mätning
- [ ] **Single command encoder för GOP** — fixat för temporal path, verifiera att spatial path också är optimerad

## Kvalitet (PSNR)

- [ ] **Adaptiv mul per GOP** — mät highpass-energi efter Haar, välj mul dynamiskt; stillastående GOPs → mul=3.0, hög rörelse → mul=0.8; enklaste vinsten för temporal path
- [ ] **Adaptiv GOP-selektion baserad på rörelseestimering** — estimera rörelse billigt (SAD/temporal varians på luma) per GOP, välj temporal mode dynamiskt; Haar för låg rörelse, None/I+P+B för hög
- [ ] **Per-tile temporal mode** — tile-oberoende arkitektur möjliggör Haar per tile istället för globalt; tiles med låg rörelse kör Haar, hög rörelse kör All-I

## Broadcast-kompatibilitet

- [ ] **4:2:0 + 10-bit i ett svep** — 4:2:0 är broadcast-standard (ST 2110, JPEG XS); 10-bit är trivial förändring (f32 internt redan); gör tillsammans som "Phase 6 broadcast-kompatibilitet"
- [ ] **BT.709 / BT.2020 färgrymd** — korrekt färgrymdshantering för broadcast-pipeline
- [ ] **SRT-transport-integration**

## Validering

- [ ] **200-frame benchmarks** — nuvarande svit kör 120 frames; utöka för mer statistisk stabilitet
- [ ] **Rate control** — konstant bitrate-läge

---

## Gjort (referens)

- [x] Spatial wavelet — korrekt, 148 tester
- [x] Haar temporal multilevel (experimentellt) — GPU race-bug fixad
- [x] Benchmark-svit — CSV-output, Xiph-sekvenser
- [x] WASM-player — iPhone/Safari-stöd
- [x] GNV2 bitstream — roundtrip bit-exact
- [x] v0.1-spatial taggad — stabil baseline
