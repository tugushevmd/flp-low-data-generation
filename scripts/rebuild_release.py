from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

print("\nEvaluating fixed-checkpoint generations")
subprocess.run([
    sys.executable,
    ROOT / "scripts" / "evaluate_fixed_checkpoint_sensitivity.py",
], cwd=ROOT, check=True)

STEPS = [
    "analyze_corpus_composition.py",
    "analyze_statistical_evidence.py",
    "analyze_model_context.py",
    "analyze_fragment_library_proximity.py",
    "audit_protocol.py",
    "score_blinded_decoy_review.py",
    "build_publication_figures_v211.py",
]

for script in STEPS:
    print(f"\nRunning {script}")
    subprocess.run([sys.executable, ROOT / "scripts" / script], check=True)

print("\nRunning tests")
subprocess.run([
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
], cwd=ROOT, check=True)

print("\nRelease outputs rebuilt successfully.")
