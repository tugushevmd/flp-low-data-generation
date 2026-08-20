from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "results" / "external_anchors" / "raw"
OUTPUT = ROOT / "artifacts" / "external_generations"
OUTPUT.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "gpmolformer_42_raw.zip": "external_gpmolformer_25_results.zip",
    "molgpt_raw.zip": "external_molgpt_results.zip",
    "reinvent_raw.zip": "external_reinvent_results.zip",
}


def include(name):
    lower = name.lower()
    if name.endswith("/") or "tensorboard" in lower or "events.out.tfevents" in lower:
        return False
    return lower.endswith((".csv", ".json", ".toml", ".smi", ".txt", ".yaml", ".yml"))


manifest = {}
for output_name, source_name in ARCHIVES.items():
    source_path = SOURCE / source_name
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    included = []
    with ZipFile(source_path) as source_zip, ZipFile(
        OUTPUT / output_name, "w", compression=ZIP_DEFLATED
    ) as output_zip:
        for info in source_zip.infolist():
            if include(info.filename):
                data = source_zip.read(info.filename)
                output_zip.writestr(info.filename, data)
                included.append(info.filename)
    manifest[output_name] = {
        "source_archive": source_name,
        "source_sha256": source_hash,
        "included_files": included,
        "note": "TensorBoard event logs and model weights were omitted; raw generation tables and run settings were retained.",
    }

(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Created", len(manifest), "compact archives in", OUTPUT)
