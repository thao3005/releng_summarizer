import re
from openai import OpenAI


class LogSummaryEvaluator:
    """
    LLM-based evaluator for CI/CD summaries.

    It can evaluate:
      - a run-level summary (multiple jobs/steps in one run), or
      - a pipeline-level summary (multiple runs in a pipeline).

    It rates:
      - Fact coverage (1–5)
      - Groundedness (1–5)
      - Length adherence (1–5)
    and returns a short comment.

    You call evaluate(log_text, summary) and it returns a text report.
    """

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    @staticmethod
    def count_sentences(text: str) -> int:
        """
        Rough sentence count using . ! ? as delimiters.
        This is mainly used to give the model a sense of length.
        """
        parts = re.split(r"[.!?]+", text)
        sentences = [p.strip() for p in parts if p.strip()]
        return len(sentences)

    def evaluate(self, log_text: str, summary: str) -> str:
        """
        Run the LLM-based evaluation and return a small text report.

        log_text:
          - For a run summary: combined logs for all jobs in that run.
          - For a pipeline summary: combined logs or metadata for all runs.

        summary:
          - A human-readable summary that may include:
            * A header with run/pipeline name
            * A bullet list of jobs/runs with statuses
            * Description / Recommendation sections.
        """
        sentence_count = self.count_sentences(summary)

        eval_prompt = f"""
You are evaluating a human-readable summary of CI/CD results.

The summary may describe either:
- a single CI/CD run that contains multiple jobs/steps, or
- an entire CI/CD pipeline that contains multiple runs.

You are given:
- The original logs (for that run or pipeline).
- The summary text.

Original logs:
{log_text}

Summary:
{summary}

The summary currently has approximately {sentence_count} sentences
(you may treat short bullet lines without . ! ? as sentence fragments).

Rate the summary on:

Fact coverage (1-5): Does it mention the most important outcomes
(e.g., which jobs or runs succeeded or failed, and the main error if any)?

Groundedness (1-5): Does it avoid inventing details that are not
supported by the logs?

Length adherence (1-5): Is the summary concise and easy to read overall?
It is OK to have a bullet list plus a short Description/Recommendation
section, but it should not be overly long or repetitive.

Respond EXACTLY in this format:

Log Name/File Name
Fact coverage (1-5): <number>
Groundedness (1-5): <number>
Length adherence (1-5): <number>
Comments: <one or two short sentences>
"""
        response = self.client.responses.create(
            model=self.model,
            input=eval_prompt,
        )
        return response.output[0].content[0].text.strip()
