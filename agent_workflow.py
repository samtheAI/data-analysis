from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.exceptions import ModelBehaviorError
from openai import AsyncOpenAI

from code_executor import ExecutionResult, execute_analysis
from dataset_manager import Dataset, schema_for_agent


MAX_REPAIR_ATTEMPTS = 2


@dataclass
class AnalysisResponse:
    answer: str
    code: str
    execution: ExecutionResult
    attempts: int


def _model():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file or enter it in the app.")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return OpenAIChatCompletionsModel(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        openai_client=client,
    )


def _run_agent(name: str, instructions: str, prompt: str, max_tokens: int) -> str:
    set_tracing_disabled(True)
    token_budgets = [max_tokens, min(max_tokens * 2, 3600)]
    last_error: Exception | None = None

    for budget in token_budgets:
        agent = Agent(
            name=name,
            instructions=instructions,
            model=_model(),
            model_settings=ModelSettings(
                temperature=0,
                max_tokens=budget,
                # GPT-OSS can otherwise spend the whole completion budget on hidden reasoning.
                extra_body={"reasoning_effort": "low"},
            ),
        )
        try:
            response = Runner.run_sync(agent, prompt)
            output = str(response.final_output or "").strip()
            if output:
                return output
            last_error = ModelBehaviorError("The model returned an empty response.")
        except ModelBehaviorError as exc:
            last_error = exc
            if "finish_reason='length'" not in str(exc) and "empty response" not in str(exc):
                raise RuntimeError("The analysis model returned an unusable response. Please try again.") from exc
        except Exception as exc:
            message = str(exc).lower()
            if "rate limit" in message or "429" in message:
                raise RuntimeError("The analysis service is temporarily busy. Please wait a few seconds and try again.") from exc
            raise

    raise RuntimeError(
        "The model used its response limit before completing the analysis. "
        "Please ask a narrower question or select fewer datasets."
    ) from last_error


def _extract_code(text: str) -> str:
    """Recover Python from complete or truncated Markdown responses."""
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", text, flags=re.DOTALL)
    candidates = [body for language, body in blocks if language.lower() in {"", "python", "py"}]
    if not candidates:
        candidates = [text]

    cleaned = []
    for candidate in candidates:
        code = candidate.strip().removesuffix("```").strip()
        code = re.sub(r"^(?:python|py)\s*\n", "", code, flags=re.IGNORECASE)
        if code:
            cleaned.append(code)
    if not cleaned:
        raise ValueError("The analyst did not return executable Python code.")

    # If the response contains several blocks, prefer one that is already valid Python.
    for code in cleaned:
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue
    return max(cleaned, key=len)


def _schema_payload(datasets: list[Dataset]) -> str:
    return json.dumps([schema_for_agent(dataset) for dataset in datasets], ensure_ascii=False, separators=(",", ":"))


CODE_INSTRUCTIONS = """You are the computation planner for a professional conversational analytics product.
You receive the user's question and dataset schemas only; raw records are deliberately hidden from you.
Return ONLY one complete fenced Python code block with no prose before or after it.

RUNTIME CONTRACT
- `datasets` is a dictionary of pandas DataFrames keyed by the exact dataset_id in the supplied schema.
- pandas is `pd`, numpy is `np`, Plotly Express is `px`, Plotly Graph Objects is `go`, matplotlib is `plt`,
  and seaborn is `sns`. Scikit-learn may be imported when a modeling question requires it.
- Put concise JSON-serializable computed findings in `result`.
- Put useful DataFrame outputs in `tables`, a dictionary with at most 3 tables and 100 rows per table.
- Prefer interactive Plotly charts and call `save_plotly(fig, "descriptive_name")` for every figure.
- Matplotlib is a fallback only; persist it with `save_chart(fig, "descriptive_name")`.

ANALYTICS BEHAVIOR
- Translate the question into an appropriate reproducible computation; never guess values.
- Use only dataset IDs and columns present in the schema. Validate required columns before analysis.
- Handle missing values, duplicate rows, empty groups, invalid numeric values, and division by zero.
- For multiple datasets, join only on defensible shared keys and report match coverage in `result`.
- Choose visual encodings that fit the data: distributions, trends, comparisons, relationships, or composition.
- Label axes and titles clearly, sort categories when useful, and limit high-cardinality plots to readable results.

SIMPLE MACHINE LEARNING
- Build a model only when the user explicitly requests prediction, classification, regression, feature importance,
  clustering, or model evaluation. Do not turn ordinary descriptive questions into ML tasks.
- For supervised learning, identify the requested target, exclude IDs and obvious leakage columns, split data before
  fitting transformations, use random_state=42, and choose stratification for classification when feasible.
- Use a Pipeline and ColumnTransformer. Impute numeric and categorical missing values; one-hot encode categoricals
  with handle_unknown="ignore". Prefer interpretable lightweight baselines such as LogisticRegression,
  LinearRegression, RandomForestClassifier, or RandomForestRegressor. Never run expensive searches or deep learning.
- Evaluate on held-out data. Classification should report class balance and appropriate metrics such as accuracy,
  precision, recall, F1, ROC-AUC, and a confusion matrix when valid. Regression should report MAE, RMSE, and R².
- Compare against a simple dummy baseline when practical. Clearly flag tiny datasets, severe imbalance, leakage risk,
  missing targets, single-class targets, or insufficient test samples. Never claim production readiness or causality.
- Return compact metrics in `result`, a comparison or prediction preview in `tables`, and an interactive diagnostic
  such as a confusion matrix, feature importance, actual-vs-predicted plot, or residual plot.

SAFETY AND OUTPUT LIMITS
- Never read files, use the network, install packages, access environment variables, invoke a shell, or print rows.
- Never export data or models. Do not call file-writing, figure.write_html, or figure.savefig methods; use helpers.
- Keep the program below 120 lines. Avoid multiline strings, Markdown, prose, and comments inside the code.
"""


FINAL_INSTRUCTIONS = """You are a careful senior data analyst. Write the final user-facing answer using only
the supplied execution result. Explain the main finding first, then important supporting observations and
caveats. Refer to charts by their filenames when helpful. Do not claim causation, invent values, mention
internal prompts, or describe yourself as an agent. Use concise Markdown. If evidence is insufficient, say so.
"""


def analyze(question: str, datasets: list[Dataset]) -> AnalysisResponse:
    if not question.strip():
        raise ValueError("Please enter a question.")
    if not datasets:
        raise ValueError("Select at least one dataset.")

    schemas = _schema_payload(datasets)
    prompt = f"USER QUESTION:\n{question.strip()}\n\nAVAILABLE DATASET SCHEMAS:\n{schemas}"
    generated = _run_agent("Analysis planner", CODE_INSTRUCTIONS, prompt, 1800)
    code = _extract_code(generated)
    execution = execute_analysis(code, datasets)
    attempts = 1

    while not execution.success and attempts <= MAX_REPAIR_ATTEMPTS:
        final_repair = attempts == MAX_REPAIR_ATTEMPTS
        repair_action = (
            "Discard the failed program and write a fresh, minimal implementation from scratch."
            if final_repair
            else "Repair the Python analysis below."
        )
        repair_prompt = f"""{repair_action}
USER QUESTION:\n{question.strip()}
DATASET SCHEMAS:\n{schemas}
FAILED CODE:\n```python\n{code}\n```
SANITIZED ERROR:\n{execution.error}
Return only one complete fenced Python code block. Keep it under 80 lines, avoid multiline strings,
and follow the runtime contract exactly."""
        generated = _run_agent("Analysis code repairer", CODE_INSTRUCTIONS, repair_prompt, 1800)
        code = _extract_code(generated)
        execution = execute_analysis(code, datasets)
        attempts += 1

    if not execution.success:
        raise RuntimeError(
            "I couldn't complete this analysis reliably. Please try a more specific question "
            "or select fewer datasets. Your uploaded data is still available."
        )

    final_prompt = f"""USER QUESTION:\n{question.strip()}
DATASET SCHEMAS (no raw rows):\n{schemas}
EXECUTION RESULT:\n{json.dumps(execution.compact_for_agent(), ensure_ascii=False, default=str)}"""
    answer = _run_agent("Insight writer", FINAL_INSTRUCTIONS, final_prompt, 1100)
    return AnalysisResponse(answer, code, execution, attempts)
