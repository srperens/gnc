//! The bitstream generation number is a shared id namespace, and it had no allocator.
//!
//! On 2026-09-08 **two different formats claimed `GP19`**: ENT-9's context-coded abac prefix, on
//! `main`, and TILE-1's plane padding, uncommitted in another worktree (BUG-51). A duplicated
//! decision number is a documentation nuisance and a duplicated item id is a queue nuisance, but
//! a duplicated *generation* means a file can be either format while
//! `deserialize_compressed_validated`'s `gen >= N` gates are right for only one of them —
//! `docs/decisions/0074` spells out that this yields a plausible wrong image rather than an error.
//!
//! `scripts/claim` allocates decision numbers, BUG ids and item ids out of a compare-and-swap
//! (`0050`, `0065`). The generation is still picked by reading `format.rs` and adding one, which
//! is the read-then-write race that produced the `0018`, `0024` and `0027` pairs. This test does
//! not close the race — two sessions can still both pick the next number — but it makes the
//! collision **fail at `cargo test` rather than at a merge**, where a clean textual resolution
//! hides it. Same shape as `tests/requested_limits.rs`: the table is the single source, so assert
//! against the table.

use std::collections::BTreeMap;

/// Every `b"GPnn" => nn,` arm of the generation table in `format.rs`, as (magic, generation).
fn generation_table(src: &str) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        // Only the table's arms: `b"GP18" => 18,`. Comments mentioning GP18 are not arms.
        let Some(rest) = line.strip_prefix("b\"") else { continue };
        let Some((magic, tail)) = rest.split_once('"') else { continue };
        let Some(num) = tail.trim().strip_prefix("=>") else { continue };
        let num = num.trim().trim_end_matches(',').trim();
        if let Ok(gen) = num.parse::<u32>() {
            out.push((magic.to_string(), gen));
        }
    }
    out
}

fn format_rs() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/format.rs"))
        .expect("src/format.rs is readable")
}

#[test]
fn every_generation_number_is_claimed_by_exactly_one_magic() {
    let table = generation_table(&format_rs());
    assert!(
        table.len() >= 12,
        "parsed only {} generation arms, so the parser has drifted from the table it checks",
        table.len()
    );

    let mut by_gen: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for (magic, gen) in &table {
        by_gen.entry(*gen).or_default().push(magic.clone());
    }
    let clashes: Vec<_> = by_gen.iter().filter(|(_, m)| m.len() > 1).collect();
    assert!(
        clashes.is_empty(),
        "two magics claim one generation: {clashes:?} — a file at that generation could be \
         either format and the `gen >= N` gates are right for only one of them (BUG-51). \
         Renumber the later one and give it its own gate."
    );

    let mut by_magic: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for (magic, gen) in &table {
        by_magic.entry(magic.as_str()).or_default().push(*gen);
    }
    let dup_magic: Vec<_> = by_magic.iter().filter(|(_, g)| g.len() > 1).collect();
    assert!(
        dup_magic.is_empty(),
        "one magic maps to several generations: {dup_magic:?}"
    );
}

#[test]
fn the_magic_written_is_the_newest_generation_in_the_table() {
    let src = format_rs();
    let table = generation_table(&src);
    let newest = table
        .iter()
        .max_by_key(|(_, gen)| *gen)
        .expect("the generation table is not empty");

    // What `serialize_compressed` actually stamps on a frame.
    let written = src
        .lines()
        .filter_map(|l| l.trim().strip_prefix("out.extend_from_slice(b\""))
        .filter_map(|r| r.split_once('"').map(|(m, _)| m))
        .find(|m| m.starts_with("GP"))
        .expect("serialize_compressed writes a GP magic");

    assert_eq!(
        written, newest.0,
        "the encoder stamps {written} but the newest generation in the table is {} — either a \
         generation was added without switching the writer to it, or the writer was bumped \
         without a table entry, and one of those decodes as the wrong format",
        newest.0
    );
}
