"""Explicitly download public, pinned Kronos config/safetensors into an immutable local cache."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from tradecopilot.forecast.kronos import _VARIANTS, CHECKPOINT_FILES, CHECKPOINT_REVISIONS, _verify_cache, _verify_file


def download_checkpoints(output: Path, *, variants: Sequence[str] = ("mini", "small")) -> Path:
    """Fetch only registered public files; never replace an existing manifest or mismatched file."""
    if (
        not output.is_absolute() or output.is_symlink() or not variants
        or len(set(variants)) != len(variants) or any(variant not in _VARIANTS for variant in variants)
    ):
        raise ValueError("Kronos setup requires an absolute destination and unique mini/small variants")
    names = list(dict.fromkeys(name for variant in variants for name in _VARIANTS[variant][:2]))
    required = (_VARIANTS[variants[0]][0], _VARIANTS[variants[0]][1])
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing_records = _verify_cache(output, required)
        if any(name not in existing_records for name in names):
            raise ValueError("Kronos manifest is immutable; choose a new destination for additional variants")
        return manifest_path

    from huggingface_hub import hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="kronos-setup-", dir=output) as temporary:
        for name in names:
            directory = output / name
            if directory.is_symlink():
                raise ValueError("Kronos checkpoint directories must not be symlinks")
            directory.mkdir(exist_ok=True)
            if any(path.name not in {"config.json", "model.safetensors"} for path in directory.iterdir()):
                raise ValueError("Kronos setup accepts config and safetensors only")
            files = []
            for filename, (size, digest) in CHECKPOINT_FILES[name].items():
                target = directory / filename
                if not target.exists() and not target.is_symlink():
                    downloaded = Path(hf_hub_download(
                        repo_id=f"NeoQuasar/{name}", filename=filename, revision=CHECKPOINT_REVISIONS[name],
                        token=False, local_files_only=False, local_dir=Path(temporary) / name,
                        endpoint="https://huggingface.co",
                    ))
                    _verify_file(downloaded, size, digest)
                    with downloaded.open("rb") as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                files.append(_verify_file(target, size, digest))
            records.append({"name": name, "repository": f"NeoQuasar/{name}",
                            "revision": CHECKPOINT_REVISIONS[name], "files": files})
    manifest = {"schema_version": "kronos-local-checkpoints-v1", "downloaded_at": datetime.now(UTC).isoformat(),
                "checkpoints": records}
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    _verify_cache(output, required)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("mini", "small"), action="append", dest="variants")
    args = parser.parse_args()
    print(download_checkpoints(args.output_dir, variants=args.variants or ("mini", "small")))


if __name__ == "__main__":
    main()
