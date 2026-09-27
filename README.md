# Northstar Analytics

Northstar Analytics is a conversational, low-code data analysis workspace. Users upload several datasets, select one or more sources for a question, and receive a written finding supported by Python-generated tables and charts.

It also supports lightweight machine-learning questions. Users can request baseline classification or regression models, held-out evaluation, comparison with a dummy baseline, feature importance, confusion matrices, and actual-versus-predicted diagnostics. The modeling workflow is intentionally limited to explainable classroom-scale models rather than expensive training or deep learning.

## How it works

1. The app loads CSV, Excel, JSON, or Parquet files into local memory.
2. It creates schema metadata: names, types, dimensions, missing counts, cardinality, and numeric summaries. Raw rows are not included in model prompts.
3. An OpenAI Agents SDK agent, connected to Groq's OpenAI-compatible endpoint, writes Python analysis code from the question and schema.
4. The app validates the code, executes it in an isolated temporary process with a timeout, and captures tables, charts, and errors.
5. Failed code is sent through a maximum of two repair attempts using only the schema, failed code, and sanitized error.
6. A second agent converts the compact execution result into the final written analysis. Streamlit combines that narrative with locally generated evidence.

Charts are rendered as interactive Plotly visualizations whenever possible, with Matplotlib retained as a fallback.

## Run on macOS or Linux

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/samtheAI/data-analysis.git
cd data-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your Groq key to `.env`, then run:

```bash
streamlit run app.py
```

## Run on Windows PowerShell

```powershell
git clone https://github.com/samtheAI/data-analysis.git
cd data-analysis
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Groq key to `.env`, then run:

```powershell
streamlit run app.py
```

## Configuration

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

## Reliability and safety

- Each generated program is parsed before execution. Filesystem, network, shell, dynamic-code, and data-export operations are rejected.
- Execution happens in a fresh temporary process with a 25-second timeout.
- Errors are sanitized and repaired at most twice, preventing endless retry loops.
- Result tables are limited to 100 rows and model-facing result previews to 30 rows.
- Uploaded data remains local, but schemas and aggregate statistics are sent to the configured model provider.

The code restrictions are a strong classroom safeguard, not a hardened production sandbox. A public multi-tenant deployment should run generated code inside disposable containers or microVMs with operating-system-level network, filesystem, CPU, and memory controls.
