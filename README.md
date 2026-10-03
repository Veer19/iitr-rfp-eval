# RFP Review Studio

A standalone Streamlit application for evidence-first RFP/proposal evaluation.

## What it does

- Upload multiple supplier proposal PDFs.
- Extract proposal text locally with PyMuPDF.
- Evaluate every proposal against configurable criteria with an OpenRouter LLM.
- Require evidence and justification for each score.
- Validate and normalize model output.
- Clip invalid scores and flag missing/duplicate criteria.
- Calculate weighted absolute scores deterministically outside the LLM.
- Calculate peer benchmarks, score gaps and relative performance.
- Calculate a weighted peer-performance index (PPI).
- Apply deterministic ranking/tie-break rules.
- Generate a risk radar using transparent keyword classification.
- Persist runs and supplier results in SQLite.
- Export the complete result as JSON and the leaderboard as CSV.
- Browse previous runs from the UI.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then paste an OpenRouter API key into the sidebar.

## Architecture

```text
PDF uploads
    |
    v
Text extraction
    |
    v
Evidence-based LLM review
    |
    v
JSON parsing + schema validation
    |
    v
Normalization / safety checks
    |
    v
Deterministic weighted scoring
    |
    v
Peer benchmark -> relative performance -> PPI
    |
    v
Deterministic ranking
    |
    +--> Risk radar
    |
    +--> SQLite persistence
    |
    v
Streamlit scorecards / CSV / JSON
```

## Important implementation choice

The model evaluates the proposal and supplies evidence. Arithmetic, normalization, benchmarking, PPI and ranking happen in Python. This keeps the business logic reproducible instead of asking the LLM to perform the final calculations.

The experience field is deliberately set to `0.0` for uploaded proposals because the source workflow treated experience as external supplier metadata. This app does not invent that value from proposal text.
