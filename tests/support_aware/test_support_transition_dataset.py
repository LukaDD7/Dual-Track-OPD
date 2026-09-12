"""CPU tests for STP-OPD data-side scaffold construction."""

from dual_track_opd.support_aware.support_transition_dataset import (
    build_samples,
    load_prefixes,
)


def test_load_prefixes_roundtrip(tmp_path):
    path = tmp_path / "prefixes.json"
    path.write_text('{"geo3k:1298": [1, 2, 3], "geo3k:596": [4, 5]}')
    prefixes = load_prefixes(path)
    assert prefixes["geo3k:1298"] == (1, 2, 3)
    assert prefixes["geo3k:596"] == (4, 5)


def test_build_samples_pairs_both_branches():
    rows = [
        {"sample_uid": "geo3k:1298", "question": "Find x.", "images": "img"},
    ]
    prefixes = {"geo3k:1298": (10, 11, 12)}
    samples = build_samples(rows, prefixes, scaffold_flags={"geo3k:1298": True})
    assert len(samples) == 2
    scaffolded = [s for s in samples if s.scaffolded]
    unscaffolded = [s for s in samples if not s.scaffolded]
    assert len(scaffolded) == 1 and len(unscaffolded) == 1
    assert scaffolded[0].prefix_token_ids == (10, 11, 12)
    # scaffolded branch renders an assistant turn; unscaffolded does not
    assert len(scaffolded[0].messages) == 2
    assert len(unscaffolded[0].messages) == 1


def test_build_samples_unscaffolded_prompt_gets_no_prefix():
    rows = [{"sample_uid": "geo3k:609", "question": "q", "images": None}]
    samples = build_samples(rows, {}, scaffold_flags={"geo3k:609": False})
    assert all(not sample.scaffolded for sample in samples)
    assert all(sample.prefix_token_ids == () for sample in samples)
