import re
import hashlib
from pathlib import Path
from collections import Counter

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def parse_run_summary(text: str):
    """
    Extract statuses from lines like:
      - Job <name>: success|failure|unknown
    Returns a list of statuses.
    """
    statuses = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*Job\s+.+:\s*(success|failure|unknown)\s*$", line.strip())
        if m:
            statuses.append(m.group(1))
    return statuses

def summarize_unknowns(output_root="summaries"):
    output_path = Path(output_root)
    run_files = sorted(output_path.rglob("run_summary.txt"))

    seen_hashes = set()
    deduped_files = []
    all_statuses = []

    for f in run_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        h = hash_text(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped_files.append(f)

        all_statuses.extend(parse_run_summary(text))

    counts = Counter(all_statuses)
    total_jobs = sum(counts.values())
    unknown_jobs = counts.get("unknown", 0)
    unknown_pct = (unknown_jobs / total_jobs * 100.0) if total_jobs else 0.0

    print(f"Found {len(run_files)} run_summary.txt files")
    print(f"Deduped to {len(deduped_files)} unique run_summary.txt files (by content)")
    print(f"Total jobs counted: {total_jobs}")
    print(f"Unknown jobs: {unknown_jobs}")
    print(f"Unknown percentage: {unknown_pct:.2f}%")
    print("Breakdown:", dict(counts))

if __name__ == "__main__":
    summarize_unknowns("summaries")
