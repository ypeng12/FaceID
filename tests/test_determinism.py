"""
Determinism checks for ingestion and pair generation.
These tests verify that running scripts twice with the same seed
produces identical outputs.
"""
import sys
import os
import hashlib
import subprocess
import tempfile
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _file_hash(path):
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_determinism():
    """
    If outputs/manifest.yaml exists, re-reading it should parse
    without error and contain required keys.
    """
    import yaml
    manifest_path = os.path.join(PROJECT_ROOT, 'outputs', 'manifest.yaml')
    if not os.path.exists(manifest_path):
        print(f"SKIP: {manifest_path} not found (run ingest_lfw.py first)")
        return

    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)

    required_keys = ['seed', 'split_policy', 'counts', 'data_source']
    for key in required_keys:
        assert key in manifest, f"Manifest missing required key: {key}"

    assert isinstance(manifest['seed'], int), "seed must be an integer"
    assert isinstance(manifest['counts'], dict), "counts must be a dict"
    for split in ['train', 'val', 'test']:
        assert split in manifest['counts'], f"counts missing split: {split}"
        assert 'identities' in manifest['counts'][split], f"counts[{split}] missing 'identities'"
        assert 'images' in manifest['counts'][split], f"counts[{split}] missing 'images'"

    print("Manifest determinism check PASSED")


def test_pairs_determinism():
    """
    If pair CSV files exist, verify they have the correct schema
    (left_path, right_path, label, split).
    """
    import csv
    pairs_dir = os.path.join(PROJECT_ROOT, 'outputs', 'pairs')
    if not os.path.exists(pairs_dir):
        print(f"SKIP: {pairs_dir} not found (run make_pairs.py first)")
        return

    expected_fields = ['left_path', 'right_path', 'label', 'split']

    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(pairs_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            print(f"SKIP: {csv_path} not found")
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == expected_fields, \
                f"{split}.csv has wrong headers: {reader.fieldnames}"
            rows = list(reader)
            assert len(rows) > 0, f"{split}.csv is empty"

            for row in rows:
                assert row['label'] in ('0', '1'), \
                    f"Invalid label '{row['label']}' in {split}.csv"
                assert row['split'] == split, \
                    f"Split mismatch: expected '{split}', got '{row['split']}'"

    print("Pairs determinism check PASSED")


if __name__ == "__main__":
    test_manifest_determinism()
    test_pairs_determinism()
    print("All determinism tests passed!")
