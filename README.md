# CPSC601_Final_Project_Log_Summarizer
- Michael Shi (10174675)
- Phuong Thao Nguyen (30118157)

This project is a CI/CD log summarization tool. It includes an LLM-based summarizer (using the OpenAI API) that generates short, human-readable run summaries from raw log files, plus an LLM-based evaluator that scores each summary on fact coverage, groundedness, and length adherence. The repo also contains a rule-based summarizer that parses GitHub Actions–style logs (groups, steps, exit codes, error markers) to infer job status, extract likely root causes, and produce run-level summaries without using an LLM, providing a simple baseline to compare against the LLM approach.