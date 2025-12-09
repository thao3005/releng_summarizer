import os
import argparse
from typing import List, Tuple, Dict

from dotenv import load_dotenv
from openai import OpenAI
from evaluator import LogSummaryEvaluator


def load_api_key():
    """
    Load the OpenAI API key from the .env file (OPENAI_API_KEY).
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Did you create a .env file with OPENAI_API_KEY=... ?"
        )
    return api_key

def summarize_run_with_llm(
    client: OpenAI,
    run_display_name: str,
    steps: List[Tuple[str, str]],
) -> str:
    """
    Use the LLM to summarize an entire run, given all its job logs.

    steps: list of (job_name, log_text)

    Expected output format:

    Run '<run_display_name>' summary (<N> jobs):
    - Job <job name>: <succeeded|failed|unknown>
    - Job <job name>: <succeeded|failed|unknown>
    ...

    Description:
    <2–4 sentence description>

    Recommendation:
    <next actions>
    """
    max_chars_per_step = 4000  # safety limit per job log
    jobs_block_parts: List[str] = []

    for job_name, log_text in steps:
        if len(log_text) > max_chars_per_step:
            log_text = log_text[-max_chars_per_step:]
        jobs_block_parts.append(
            f"Job: {job_name}\nLog:\n{log_text}\n\n---\n"
        )

    jobs_block = "\n".join(jobs_block_parts)

    prompt = f"""
You are summarizing a CI/CD pipeline run composed of multiple jobs.

Run identifier: {run_display_name}
Number of jobs: {len(steps)}

For each job, decide whether it **succeeded**, **failed**, or is **unknown**
(if the log is ambiguous or incomplete).

Produce output in EXACTLY this format:

Run '{run_display_name}' summary ({len(steps)} jobs):
- Job <job name>: <succeeded|failed|unknown>
- Job <job name>: <succeeded|failed|unknown>
...

Description:
<2–4 sentences in plain English describing how the run went overall,
mentioning any failures and the main issues/steps.>

Recommendation:
- If all jobs succeeded and there are no clear errors, say that no
  immediate follow-up is needed.
- If any job failed or shows issues, briefly state what to investigate
  next and at which job(s).

Here are the job logs:

{jobs_block}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return response.output[0].content[0].text.strip()

def parse_run_status_from_summary(run_summary: str) -> Tuple[str, List[str]]:
    """
    Parse job lines from the run summary to infer:
      - overall run status: 'succeeded', 'failed', or 'unknown'
      - list of failing job names
    We look for lines like: "- Job <name>: <status>"
    """
    failed_steps: List[str] = []
    has_success = False

    for line in run_summary.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- Job "):
            continue

        # Expect "- Job <job name>: <status text>"
        try:
            prefix, status_part = stripped.split(":", 1)
        except ValueError:
            continue

        job_name = prefix[len("- Job "):].strip()
        status_text = status_part.strip().lower()

        if "failed" in status_text:
            failed_steps.append(job_name)
        elif "succeeded" in status_text or "success" in status_text:
            has_success = True

    if failed_steps:
        overall_status = "failed"
    elif has_success:
        overall_status = "succeeded"
    else:
        overall_status = "unknown"

    return overall_status, failed_steps

def write_pipeline_summary(
    pipeline_out_dir: str,
    parent_folder_name: str,
    runs_info: List[Dict[str, object]],
) -> str:
    """
    Write a pipeline-level summary file:

    summaries/<repo>_<pipeline>/pipeline_summary.txt

    Contents: list of runs and whether they succeeded/failed/unknown.
    If a run failed, mention which run and to check its run_summary.txt.
    """
    os.makedirs(pipeline_out_dir, exist_ok=True)
    out_path = os.path.join(pipeline_out_dir, "pipeline_summary.txt")

    lines: List[str] = []
    lines.append(f"Pipeline summary for {parent_folder_name}")
    lines.append("")

    if not runs_info:
        lines.append("No runs were found for this pipeline.")
    else:
        lines.append("Runs:")
        for info in sorted(runs_info, key=lambda x: x["run_name"]):
            run_name = info["run_name"]
            status = info["status"]
            failed_steps = info["failed_steps"]

            if status == "failed":
                if failed_steps:
                    failed_str = ", ".join(failed_steps)
                    lines.append(
                        f"- Run {run_name}: FAILED (failed steps: {failed_str}). "
                        f"See {run_name}_summary/run_summary.txt for details."
                    )
                else:
                    lines.append(
                        f"- Run {run_name}: FAILED. "
                        f"See {run_name}_summary/run_summary.txt for details."
                    )
            elif status == "succeeded":
                lines.append(f"- Run {run_name}: succeeded.")
            else:
                lines.append(
                    f"- Run {run_name}: status unknown. "
                    f"See {run_name}_summary/run_summary.txt for details."
                )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return out_path

def read_log_file(path: str) -> str:
    """
    Read the entire log file as text.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def summarize_log_text(client: OpenAI, log_text: str) -> str:
    """
    Call the LLM to summarize the log into a short, human-friendly paragraph.
    """
    max_chars = 12000
    if len(log_text) > max_chars:
        log_text = log_text[-max_chars:]

    prompt = f"""
        You are a log summarizer for CI/CD logs.

        Read the log below and produce a very short summary in plain English.

        Rules:
        - First say clearly if the run/runs succeeded or failed (if you can tell).
        - If there are errors or failures, briefly mention the main issue
        (e.g., which step/test failed and why) in one sentence.
        - If there are no obvious errors, explicitly say that no issues were
        detected and the run appears successful.
        - Be concise, direct, and avoid extra explanations or guesses.
        - Do NOT invent details that you cannot see in the log.

        Log:
        {log_text}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",  # "gpt-4o-mini" for evaluator
        input=prompt,
    )

    return response.output[0].content[0].text.strip()

def write_summary_to_file(log_path: str, summary: str, output_dir: str | None = None) -> str:
    base_name = os.path.basename(log_path)
    name_without_ext, _ = os.path.splitext(base_name)

    if output_dir is None:
        output_dir = os.path.dirname(log_path) or "."

    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, f"{name_without_ext}_summary.txt")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"{base_name} summary\n\n")
        f.write(summary)
        f.write("\n")

    return output_filename

def process_single_log(client: OpenAI, log_path: str) -> None:
    """
    Summarize a single log file and write a <name>_summary.txt next to it.
    """
    print(f"[single] Reading log: {log_path}")
    log_text = read_log_file(log_path)
    print(f"[single] Loaded {len(log_text)} characters")

    summary = summarize_log_text(client, log_text)

    print("\n=== Log Summary ===\n")
    print(summary)
    print("\n===================\n")

    output_file = write_summary_to_file(log_path, summary)
    print(f"[single] Summary written to: {output_file}")

def process_dataset(
    client: OpenAI,
    evaluator: LogSummaryEvaluator,
    root_logs: str,
    out_root: str = "summaries",
) -> None:
    """
    Walk the dataset tree:

    root_logs/
      creator/
        repo/
          pipeline/
            run/
              *.txt   <-- these are the log files (jobs/steps) we summarize

    For each run folder, we create:
      - One run-level summary:
          out_root/<repo>_<pipeline>/<run>_summary/run_summary.txt
      - One evaluation of that run summary:
          out_root/<repo>_<pipeline>/<run>_summary/evaluation_run_summary.txt

    For each pipeline folder (<repo>_<pipeline>), we also create:
      - pipeline_summary.txt (listing each run and overall status).
    """
    root_logs = os.path.abspath(root_logs)
    out_root = os.path.abspath(out_root)

    print(f"[dataset] Scanning logs under: {root_logs}")
    print(f"[dataset] Summaries will be written under: {out_root}\n")

    for creator_name in sorted(os.listdir(root_logs)):
        creator_path = os.path.join(root_logs, creator_name)
        if not os.path.isdir(creator_path):
            continue

        for repo_name in sorted(os.listdir(creator_path)):
            repo_path = os.path.join(creator_path, repo_name)
            if not os.path.isdir(repo_path):
                continue

            for pipeline_name in sorted(os.listdir(repo_path)):
                pipeline_path = os.path.join(repo_path, pipeline_name)
                if not os.path.isdir(pipeline_path):
                    continue

                parent_folder_name = f"{repo_name}_{pipeline_name}"
                pipeline_out_dir = os.path.join(out_root, parent_folder_name)

                print(f"[dataset] Pipeline: {creator_name}/{repo_name}/{pipeline_name}")
                runs_info: List[Dict[str, object]] = []

                for run_name in sorted(os.listdir(pipeline_path)):
                    run_path = os.path.join(pipeline_path, run_name)
                    if not os.path.isdir(run_path):
                        continue

                    # Collect all .txt job logs in this run
                    txt_files = [
                        f for f in os.listdir(run_path)
                        if os.path.isfile(os.path.join(run_path, f))
                        and f.lower().endswith(".txt")
                    ]
                    if not txt_files:
                        continue

                    print(f"  [run] {run_name}")
                    run_summary_folder = f"{run_name}_summary"
                    out_run_dir = os.path.join(pipeline_out_dir, run_summary_folder)
                    os.makedirs(out_run_dir, exist_ok=True)

                    # Gather logs for this run (all jobs/steps)
                    step_logs: List[Tuple[str, str]] = []

                    for log_filename in sorted(txt_files):
                        log_path = os.path.join(run_path, log_filename)
                        log_text = read_log_file(log_path)
                        print(f"    - {log_filename}: {len(log_text)} chars")

                        job_name = os.path.splitext(log_filename)[0]
                        step_logs.append((job_name, log_text))

                    # 1) Run-level summary using all step logs
                    run_display_name = f"{repo_name}\\{pipeline_name}\\{run_name}"
                    run_summary_text = summarize_run_with_llm(
                        client,
                        run_display_name,
                        step_logs,
                    )

                    run_summary_path = os.path.join(out_run_dir, "run_summary.txt")
                    with open(run_summary_path, "w", encoding="utf-8") as f:
                        f.write(run_summary_text)
                        f.write("\n")
                    print(f"      -> run summary written to {run_summary_path}")

                    # 2) Evaluate the run summary (LLM-based evaluator)
                    #    Use the combined logs of all steps in this run
                    combined_run_log = "\n\n".join(
                        f"Job: {name}\nLog:\n{text}" for name, text in step_logs
                    )
                    run_eval_report = evaluator.evaluate(combined_run_log, run_summary_text)
                    run_eval_path = write_evaluation_to_file(run_summary_path, run_eval_report)
                    print(f"      -> run summary evaluation written to {run_eval_path}")

                    # 3) Extract run-level status + failing steps for pipeline summary
                    status, failing_steps = parse_run_status_from_summary(run_summary_text)
                    runs_info.append(
                        {
                            "run_name": run_name,
                            "status": status,
                            "failed_steps": failing_steps,
                        }
                    )

                # 4) After all runs for this pipeline, write pipeline-level summary
                pipeline_summary_path = write_pipeline_summary(
                    pipeline_out_dir,
                    parent_folder_name,
                    runs_info,
                )
                print(f"  -> pipeline summary written to: {pipeline_summary_path}\n")

def write_evaluation_to_file(summary_path: str, evaluation_text: str) -> str:
    """
    Write the evaluation to a text file named:

        evaluation_<summary_basename>.txt

    in the same folder as the summary file.
    """
    base_name = os.path.basename(summary_path)
    name_without_ext, _ = os.path.splitext(base_name)

    eval_filename = f"evaluation_{name_without_ext}.txt"
    out_dir = os.path.dirname(summary_path) or "."

    eval_path = os.path.join(out_dir, eval_filename)

    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(evaluation_text)
        f.write("\n")

    return eval_path

def main():
    parser = argparse.ArgumentParser(
        description="Summarize CI/CD log files in a dataset using an LLM."
    )
    parser.add_argument(
        "--dataset-root",
        default="logs",
        help="Root 'logs' folder to process recursively (default: logs).",
    )
    parser.add_argument(
        "--output-root",
        default="summaries",
        help="Where to store summaries (default: summaries).",
    )

    args = parser.parse_args()

    # 1. Load API key & create client
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # 2. Create evaluator
    evaluator = LogSummaryEvaluator(client)

    # 3. Process the dataset
    process_dataset(client, evaluator, args.dataset_root, args.output_root)

if __name__ == "__main__":
    main()
