import re
from pathlib import Path
from collections import defaultdict


def strip_timestamp(line: str) -> str:
    """
    Remove leading ISO timestamps like:
    2023-09-15T23:08:24.3318468Z Message...
    """
    m = re.match(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+(.*)$", line)
    return m.group(1) if m else line


def parse_groups(lines):
    """
    Parse ##[group] ... ##[endgroup] blocks from a log.
    Returns a list of dicts with keys:
      - name: the text after ##[group]
      - body: list of lines inside the group
    """
    groups = []
    current = None

    for line in lines:
        if line.startswith("##[group]"):
            if current:
                groups.append(current)
            name = line[len("##[group]"):].strip()
            current = {"name": name, "body": []}
        elif line.startswith("##[endgroup]"):
            if current:
                groups.append(current)
                current = None
        else:
            if current is not None:
                current["body"].append(line)

    if current:
        groups.append(current)

    return groups


def extract_basic_metadata(lines, job_dir: Path):
    """
    Pull out simple metadata: job name, repo/ref, OS, image.
    Uses all lines from all steps of a job.
    """
    job_name = job_dir.name
    repo = None
    ref = None
    os_name = None
    runner_image = None

    for i, line in enumerate(lines):
        if "Job defined at:" in line and not repo:
            raw = line.split("Job defined at:", 1)[1].strip()
            if "@" in raw:
                path_part, ref = raw.split("@", 1)
            else:
                path_part, ref = raw, None
            parts = path_part.split("/")
            if len(parts) >= 2:
                repo = "/".join(parts[:2])

        if "repository:" in line and not repo:
            repo = line.split("repository:", 1)[1].strip()

        if "ref:" in line and not ref:
            possible = line.split("ref:", 1)[1].strip()
            if possible.startswith("refs/"):
                ref = possible

        if line == "Operating System" and i + 1 < len(lines):
            os_name = lines[i + 1].strip()

        if line.startswith("Image:") and not runner_image:
            runner_image = line.split("Image:", 1)[1].strip()
        if "image:" in line.lower() and not runner_image:
            m = re.search(r"image[: ]+([A-Za-z0-9_.:-]+)", line, re.IGNORECASE)
            if m:
                runner_image = m.group(1)

    return {
        "job_name": job_name,
        "repo": repo,
        "ref": ref,
        "os_name": os_name,
        "runner_image": runner_image,
    }


def extract_test_commands(groups):
    """
    Look for commands that are likely test runners across many frameworks.
    Uses group headers like "Run ./gradlew testDebugUnitTest".
    """
    test_cmds = []
    TEST_KEYWORDS = [
        "pytest", "nose", "unittest", "py.test",
        "mvn test", "mvn verify",
        "gradlew", "gradle test",
        "npm test", "yarn test", "pnpm test",
        "go test", "cargo test",
        "phpunit", "rspec",
        "ctest", "jest ", " karma", "dokka",
    ]

    for g in groups:
        header = g["name"].lower()
        body_first = g["body"][0].strip().lower() if g["body"] else ""
        combined = header + " " + body_first

        if any(kw in combined for kw in TEST_KEYWORDS) and "run " in header:
            raw_header = g["name"]
            m = re.search(r"\bRun\b(.*)$", raw_header, re.IGNORECASE)
            cmd = m.group(1).strip() if m else raw_header.strip()
            if cmd and cmd not in test_cmds:
                test_cmds.append(cmd)

    return test_cmds


def extract_root_cause(lines):
    """
    Try to pull out a short root-cause snippet from the end of a (step) log.
    """
    KEY_PHRASES = [
        "##[error]",
        "::error::",
        "What went wrong:",
        "FAILURE: Build completed",
        "FAILURE: Build failed",
        "BUILD FAILED",
        "There were failing tests",
        "Test Failed",
        "Tests failed",
        "error:",
        "Error:",
        "ERROR ",
        "Traceback (most recent call last):",
        "Exception",
        "AssertionError",
    ]

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if any(phrase in line for phrase in KEY_PHRASES):
            block = [line.strip()]
            for j in range(i + 1, min(len(lines), i + 6)):
                if not lines[j].strip():
                    break
                block.append(lines[j].strip())
            text = " ".join(block)
            text = re.sub(r"\s+", " ", text)
            return text

    return None


def infer_step_status(lines):
    """
    Infer step status (success/failure/unknown) from the step log.
    Uses several patterns:
      - explicit exit code
      - GitHub error markers
      - build/test success/failure phrases
    """
    exit_code = None
    has_error_marker = False
    has_failure_keyword = False
    has_success_keyword = False

    for line in lines:
        l = line.strip()

        m = re.search(r"Process completed with exit code\s+(-?\d+)", l)
        if m:
            exit_code = int(m.group(1))

        if "##[error]" in l or "::error::" in l:
            has_error_marker = True

        if any(x in l for x in [
            "BUILD FAILED",
            "FAILURE: Build completed",
            "FAILURE: Build failed",
            "Error:",
            "error:",
            "ERROR ",
            "There were failing tests",
            "Tests failed",
            "Test Failed",
            "test failed",
            "Exiting with error",
            "Completed with result 'Failed'",
            "Completed with result 'failed'",
        ]):
            has_failure_keyword = True

        if any(x in l for x in [
            "BUILD SUCCESSFUL",
            "BUILD SUCCEEDED",
            "Finished: SUCCESS",
            "All tests passed",
            "Tests passed",
            "Completed with result 'Succeeded'",
            "Completed with result 'succeeded'",
        ]):
            has_success_keyword = True

    if exit_code is not None:
        return ("success" if exit_code == 0 else "failure", exit_code)

    if has_error_marker or has_failure_keyword:
        return "failure", None

    if has_success_keyword:
        return "success", None

    return "unknown", None


def parse_step_name(step_file: Path, lines):
    """
    Derive a human-readable step name.

    Priority:
      1. From filename prefix: '2_Check out repo.txt' -> 'Check out repo'
      2. From first ##[group] line
      3. Fallback: file stem
    """
    name = None
    stem = step_file.stem

    m = re.match(r"^\d+_(.+)$", stem)
    if m:
        name = m.group(1).replace("_", " ").strip()

    if not name:
        for raw in lines:
            l = raw.strip()
            if l.startswith("##[group]"):
                group_name = l[len("##[group]"):].strip()
                m2 = re.match(r"Run\s+(.*)$", group_name, re.IGNORECASE)
                name = (m2.group(1) if m2 else group_name).strip()
                break

    if not name:
        name = stem

    return name


def summarize_job_directory(job_dir: Path) -> tuple[str, str]:
    """
    Summarize a single job by reading all step logs in job_dir.
    Returns (summary_text, job_status) where job_status is
    'success' / 'failure' / 'unknown'.
    """
    step_files = []
    for p in job_dir.glob("*.txt"):
        m = re.match(r"^(\d+)_", p.name)
        if m:
            step_num = int(m.group(1))
        else:
            step_num = None
        step_files.append((step_num, p))

    if not step_files:
        return f"Context: Job '{job_dir.name}'. No step logs found.", "unknown"

    step_files.sort(key=lambda x: (x[0] if x[0] is not None else 9999, x[1].name))

    all_lines = []
    step_summaries = []
    auto_num = 1

    for num, step_path in step_files:
        step_num = num if num is not None else auto_num
        auto_num = step_num + 1

        text = step_path.read_text(encoding="utf-8", errors="ignore")
        raw_lines = text.splitlines()
        stripped = [strip_timestamp(l) for l in raw_lines]

        all_lines.extend(stripped)

        step_name = parse_step_name(step_path, stripped)
        status, exit_code = infer_step_status(stripped)
        root_cause = extract_root_cause(stripped)

        step_summaries.append(
            {
                "num": step_num,
                "name": step_name,
                "status": status,
                "exit_code": exit_code,
                "root_cause": root_cause,
            }
        )

    meta = extract_basic_metadata(all_lines, job_dir)
    groups = parse_groups(all_lines)
    test_cmds = extract_test_commands(groups)

    failing_steps = [s for s in step_summaries if s["status"] == "failure"]
    if failing_steps:
        job_status = "failure"
        failing_steps.sort(key=lambda d: d["num"])
        last_fail = failing_steps[-1]
    else:
        if all(s["status"] == "success" for s in step_summaries):
            job_status = "success"
        else:
            job_status = "unknown"
        last_fail = None

    root_cause = last_fail["root_cause"] if last_fail and last_fail["root_cause"] else None

    if job_status == "failure" and last_fail:
        fail_num = last_fail["num"]
        for s in step_summaries:
            if s["num"] < fail_num and s["status"] == "unknown":
                s["status"] = "success"
    elif job_status == "success":
        for s in step_summaries:
            if s["status"] == "unknown":
                s["status"] = "success"

    out_lines = []

    ctx = f"Context: Job '{meta['job_name']}'"
    extra_bits = []
    if meta["runner_image"]:
        extra_bits.append(f"image {meta['runner_image']}")
    elif meta["os_name"]:
        extra_bits.append(meta["os_name"])
    if meta["repo"]:
        extra_bits.append(f"repo {meta['repo']}")
    if meta["ref"]:
        extra_bits.append(f"ref {meta['ref']}")
    if extra_bits:
        ctx += " (" + ", ".join(extra_bits) + ")"
    out_lines.append(ctx)

    if test_cmds:
        if len(test_cmds) == 1:
            out_lines.append(f"Test command: {test_cmds[0]}")
        else:
            out_lines.append("Test commands:")
            for cmd in test_cmds:
                out_lines.append(f"  - {cmd}")

    if job_status == "failure" and last_fail:
        out_lines.append(
            f"Result: job failed in step {last_fail['num']} ('{last_fail['name']}')."
        )
    elif job_status == "success":
        out_lines.append("Result: job succeeded.")
    else:
        out_lines.append(
            "Result: job status unknown (no clear success/failure markers)."
        )

    if root_cause:
        out_lines.append("Likely root cause:")
        out_lines.append(f"  {root_cause}")

    out_lines.append("Step overview:")
    for s in sorted(step_summaries, key=lambda d: d["num"]):
        out_lines.append(f"  - Step {s['num']}: {s['name']} [{s['status']}]")

    return "\n".join(out_lines), job_status


def process_dataset(dataset_root: str, output_root: str):
    """
    Walk through dataset_root, find job dirs at ANY depth (dirs that contain
    step logs like '1_*.txt'), and for each run dir (parent of each job dir)
    write:
      - one summary per job: <output_root>/<rel_run_path>/<job>/job_summary.txt
      - one merged summary per run: <output_root>/<rel_run_path>/run_summary.txt
    Run summary only lists which jobs succeeded / failed / unknown.
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)

    runs = defaultdict(list)  # type: ignore[assignment]

    for d in dataset_path.rglob("*"):
        if not d.is_dir():
            continue

        step_files = [p for p in d.glob("*.txt") if re.match(r"^(\d+)_", p.name)]
        if not step_files:
            continue

        job_dir = d
        run_dir = job_dir.parent
        runs[run_dir].append(job_dir)

    for run_dir in sorted(runs.keys(), key=lambda p: str(p)):
        run_rel = run_dir.relative_to(dataset_path)
        run_out_dir = output_path / run_rel
        run_out_dir.mkdir(parents=True, exist_ok=True)

        stray_job_summary = run_out_dir / "job_summary.txt"
        if stray_job_summary.exists():
            stray_job_summary.unlink()

        unique_job_dirs = sorted({p for p in runs[run_dir]}, key=lambda p: p.name)

        job_summaries = []

        for job_dir in unique_job_dirs:
            summary_text, job_status = summarize_job_directory(job_dir)
            job_summaries.append((job_dir.name, job_status))

            job_out_dir = run_out_dir / job_dir.name
            job_out_dir.mkdir(parents=True, exist_ok=True)
            job_summary_file = job_out_dir / "job_summary.txt"
            job_summary_file.write_text(summary_text, encoding="utf-8")

        if job_summaries:
            run_lines = [
                f"Run '{run_rel}' summary ({len(job_summaries)} jobs):"
            ]
            for job_name, status in job_summaries:
                run_lines.append(f"  - Job {job_name}: {status}")
            run_summary_text = "\n".join(run_lines)
            (run_out_dir / "run_summary.txt").write_text(
                run_summary_text, encoding="utf-8"
            )


if __name__ == "__main__":
    DATASET_ROOT = "dataset"
    OUTPUT_ROOT = "summaries"

    process_dataset(DATASET_ROOT, OUTPUT_ROOT)
