from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import fitz
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


DEFAULT_CRITERIA = [
    {
        "criterion_id": 1,
        "name": "Technical Capability",
        "description": "Architecture, integrations, scalability, and technical fit",
        "weight": 30.0,
        "max_score": 10.0,
    },
    {
        "criterion_id": 2,
        "name": "Implementation Plan",
        "description": "Timeline, milestones, staffing, and delivery risk plan",
        "weight": 20.0,
        "max_score": 10.0,
    },
    {
        "criterion_id": 3,
        "name": "Commercial Value",
        "description": "Pricing clarity, total cost, and commercial assumptions",
        "weight": 20.0,
        "max_score": 10.0,
    },
    {
        "criterion_id": 4,
        "name": "Security & Compliance",
        "description": "Security controls, certifications, privacy, and auditability",
        "weight": 20.0,
        "max_score": 10.0,
    },
    {
        "criterion_id": 5,
        "name": "Support & Experience",
        "description": "Support model, relevant projects, and customer references",
        "weight": 10.0,
        "max_score": 10.0,
    },
]


class CriterionReview(BaseModel):
    criterion_id: int
    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    justification: str
    evidence: str
    evidence_quality: str


class ProposalReview(BaseModel):
    supplier_name: str
    criteria: list[CriterionReview]
    risks: list[str] = []
    overall_summary: str = ""


def init_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluation_criteria (
            criterion_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            weight REAL NOT NULL,
            max_score REAL NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS rfp_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS supplier_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            supplier_name TEXT NOT NULL,
            absolute_score REAL,
            ppi REAL,
            final_rank INTEGER,
            result_json TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES rfp_runs(run_id)
        );
        """)
        count = conn.execute(
            "SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
        if count == 0:
            conn.executemany(
                """INSERT INTO evaluation_criteria
                   (criterion_id, name, description, weight, max_score)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        c["criterion_id"], c["name"], c["description"],
                        c["weight"], c["max_score"]
                    )
                    for c in DEFAULT_CRITERIA
                ],
            )


def save_criteria(db_path: Path, criteria: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM evaluation_criteria")
        conn.executemany(
            """INSERT INTO evaluation_criteria
               (criterion_id, name, description, weight, max_score)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    c["criterion_id"], c["name"], c["description"],
                    c["weight"], c["max_score"]
                )
                for c in criteria
            ],
        )


def extract_pdf_text(pdf_bytes: bytes) -> str:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page_no, page in enumerate(document, start=1):
        chunks.append(f"\n--- PAGE {page_no} ---\n{page.get_text()}")
    document.close()
    return "\n".join(chunks)


def build_prompt(supplier_name: str, document_text: str, criteria: list[dict[str, Any]]) -> str:
    criteria_block = "\n".join(
        f"""Criterion ID: {c['criterion_id']}
Name: {c['name']}
Description: {c['description']}
Maximum score: {c['max_score']}
Weight: {c['weight']}%
"""
        for c in criteria
    )

    return f"""
You are an evidence-first procurement proposal reviewer.

Supplier:
{supplier_name}

Evaluation framework:
{criteria_block}

Supplier proposal:
--------------------
{document_text}
--------------------

Review every active criterion.

Rules:
1. Use only information explicitly present in the proposal.
2. Do not invent capabilities, certifications, prices, references, dates, or experience.
3. Return exactly one criterion result for every supplied criterion ID.
4. Score from 0 through the criterion's maximum.
5. Explain the score briefly.
6. Quote or paraphrase concrete proposal evidence.
7. Set evidence_quality to HIGH, MEDIUM, or LOW.
8. Identify material risks, omissions, assumptions, or dependencies.
9. Do not calculate weighted totals.
10. Do not calculate peer metrics.
11. Return JSON only.

Required JSON:
{{
  "supplier_name": "{supplier_name}",
  "criteria": [
    {{
      "criterion_id": 1,
      "score": 0,
      "max_score": 10,
      "justification": "...",
      "evidence": "...",
      "evidence_quality": "HIGH"
    }}
  ],
  "risks": ["..."],
  "overall_summary": "..."
}}
"""


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_review(raw: dict[str, Any], criteria: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    by_id: dict[int, dict[str, Any]] = {}

    for item in raw.get("criteria", []):
        try:
            cid = int(item.get("criterion_id"))
        except (TypeError, ValueError):
            warnings.append(
                "A criterion result had an invalid criterion ID and was ignored.")
            continue
        if cid in by_id:
            warnings.append(
                f"Duplicate result for criterion {cid}; first result retained.")
            continue
        by_id[cid] = item

    normalized = []
    valid_ids = {c["criterion_id"] for c in criteria}

    for criterion in criteria:
        cid = criterion["criterion_id"]
        maximum = float(criterion["max_score"])

        if cid not in by_id:
            warnings.append(
                f"Criterion {cid} ({criterion['name']}) was missing; score set to 0.")
            normalized.append({
                "criterion_id": cid,
                "score": 0.0,
                "max_score": maximum,
                "justification": "No evaluation was returned for this criterion.",
                "evidence": "No evidence returned.",
                "evidence_quality": "LOW",
            })
            continue

        item = by_id[cid]
        raw_score = item.get("score", 0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            warnings.append(
                f"Criterion {cid} had invalid score {raw_score!r}; score set to 0.")
            score = 0.0

        if score < 0:
            warnings.append(
                f"Criterion {cid} had a negative score; clipped to 0.")
            score = 0.0
        if score > maximum:
            warnings.append(
                f"Criterion {cid} exceeded its maximum; clipped to {maximum}.")
            score = maximum

        quality = str(item.get("evidence_quality", "LOW")).upper()
        if quality not in {"HIGH", "MEDIUM", "LOW"}:
            warnings.append(
                f"Criterion {cid} used unknown evidence quality; set to LOW.")
            quality = "LOW"

        normalized.append({
            "criterion_id": cid,
            "score": score,
            "max_score": maximum,
            "justification": str(item.get("justification", "No justification provided.")),
            "evidence": str(item.get("evidence", "No evidence provided.")),
            "evidence_quality": quality,
        })

    unknown = sorted(set(by_id) - valid_ids)
    if unknown:
        warnings.append(f"LLM returned unknown criterion IDs: {unknown}")

    return {
        "supplier_name": raw.get("supplier_name", "Unknown Supplier"),
        "criteria": normalized,
        "risks": [str(x) for x in raw.get("risks", [])],
        "overall_summary": str(raw.get("overall_summary", "")),
    }, warnings


def score_review(review: dict[str, Any], criteria: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    lookup = {x["criterion_id"]: x for x in review["criteria"]}
    total = 0.0
    details = []

    for criterion in criteria:
        item = lookup[criterion["criterion_id"]]
        contribution = (
            item["score"] / float(criterion["max_score"])) * float(criterion["weight"])
        total += contribution

        details.append({
            **item,
            "criterion_name": criterion["name"],
            "weight": float(criterion["weight"]),
            "weighted_contribution": round(contribution, 4),
        })

    return round(total, 4), details


def benchmark(details: list[dict[str, Any]], all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for detail in details:
        cid = detail["criterion_id"]
        scores = [
            d["score"]
            for supplier in all_results
            for d in supplier["criterion_details"]
            if d["criterion_id"] == cid
        ]
        best = max(scores) if scores else 0.0
        gap = detail["score"] - best
        relative = 100.0 if best == 0 and detail["score"] == 0 else (
            0.0 if best == 0 else (detail["score"] / best) * 100.0
        )
        out.append({
            **detail,
            "benchmark": round(best, 4),
            "gap": round(gap, 4),
            "relative_performance": round(relative, 4),
        })
    return out


def calculate_ppi(details: list[dict[str, Any]], criteria: list[dict[str, Any]]) -> float:
    weights = {c["criterion_id"]: float(c["weight"]) for c in criteria}
    return round(
        sum(d["relative_performance"] *
            weights[d["criterion_id"]] / 100 for d in details),
        4,
    )


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        results,
        key=lambda x: (
            -x["ppi"],
            x["submission_date"],
            -x["experience_rating"],
            x["supplier_name"].lower(),
        ),
    )
    for i, item in enumerate(ranked, start=1):
        item["final_rank"] = i
    return ranked


def classify_risk(text: str) -> str:
    value = text.lower()
    high = ("security", "compliance", "regulatory", "privacy",
            "certification", "critical", "missing", "no evidence")
    medium = ("integration", "dependency", "timeline", "assumption",
              "unclear", "limited", "change request", "scope")
    if any(k in value for k in high):
        return "HIGH"
    if any(k in value for k in medium):
        return "MEDIUM"
    return "LOW"


def build_risk_radar(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    radar = []
    for supplier in results:
        radar.append({
            "supplier_name": supplier["supplier_name"],
            "risks": [
                {"risk": risk, "severity": classify_risk(risk)}
                for risk in supplier["evaluation"].get("risks", [])
            ],
        })
    return radar


def persist_result(db_path: Path, run_id: str, results: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO rfp_runs(run_id, created_at, status) VALUES (?, ?, ?)",
            (run_id, datetime.now().isoformat(timespec="seconds"), "COMPLETED"),
        )
        conn.execute(
            "DELETE FROM supplier_results WHERE run_id = ?", (run_id,))
        for result in results:
            conn.execute(
                """INSERT INTO supplier_results
                   (run_id, supplier_name, absolute_score, ppi, final_rank, result_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    result["supplier_name"],
                    result["absolute_score"],
                    result["ppi"],
                    result["final_rank"],
                    json.dumps(result, ensure_ascii=False, default=str),
                ),
            )


def load_run_history(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT
                rfp_runs.run_id,
                rfp_runs.created_at,
                rfp_runs.status,
                COUNT(supplier_results.result_id) AS suppliers
            FROM rfp_runs
            LEFT JOIN supplier_results
                ON supplier_results.run_id = rfp_runs.run_id
            GROUP BY
                rfp_runs.run_id,
                rfp_runs.created_at,
                rfp_runs.status
            ORDER BY rfp_runs.created_at DESC
        """).fetchall()

    return [
        {
            "Run ID": row[0],
            "Created": row[1],
            "Status": row[2],
            "Suppliers": row[3],
        }
        for row in rows
    ]


def evaluate_batch(
    files,
    criteria: list[dict[str, Any]],
    api_key: str,
    model: str,
    db_path: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not files:
        raise ValueError("No proposal files were supplied.")
    if abs(sum(float(c["weight"]) for c in criteria) - 100.0) > 0.001:
        raise ValueError("Criterion weights must total 100%.")

    print(f"API KEY: '{api_key}' (length: {len(api_key) if api_key else 0})")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    run_id = "REV-" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
        "-" + uuid.uuid4().hex[:6]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO rfp_runs(run_id, created_at, status) VALUES (?, ?, ?)",
            (run_id, datetime.now().isoformat(timespec="seconds"), "RUNNING"),
        )

    results = []
    warnings = []

    for uploaded in files:
        supplier_name = Path(uploaded.name).stem.replace("_", " ")
        if progress_callback:
            progress_callback(f"Reviewing **{supplier_name}**")

        pdf_bytes = uploaded.getvalue()
        text = extract_pdf_text(pdf_bytes)
        prompt = build_prompt(supplier_name, text, criteria)

        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict procurement review engine. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,
        )

        raw_text = response.choices[0].message.content or ""
        try:
            raw = parse_json(raw_text)
            # Schema validation catches malformed structures before normalization.
            ProposalReview.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            warnings.append(
                f"{supplier_name}: model output required normalization ({exc}).")
            try:
                raw = parse_json(raw_text)
            except json.JSONDecodeError:
                raw = {
                    "supplier_name": supplier_name,
                    "criteria": [],
                    "risks": ["Model returned invalid JSON."],
                    "overall_summary": "Evaluation could not be parsed.",
                }

        normalized, validation_warnings = normalize_review(raw, criteria)
        score, details = score_review(normalized, criteria)

        results.append({
            "supplier_name": supplier_name,
            "submission_date": datetime.now().date().isoformat(),
            "experience_rating": 0.0,
            "status": "SUCCESS",
            "evaluation": normalized,
            "absolute_score": score,
            "criterion_details": details,
            "validation_warnings": validation_warnings,
        })
        warnings.extend(f"{supplier_name}: {w}" for w in validation_warnings)

    # Experience is kept as a metadata field for compatibility with the source workflow.
    # The app does not invent an experience rating from proposal text.
    benchmarks = {}
    for criterion in criteria:
        cid = criterion["criterion_id"]
        values = [
            d["score"] for r in results for d in r["criterion_details"]
            if d["criterion_id"] == cid
        ]
        benchmarks[cid] = max(values) if values else 0.0

    for result in results:
        result["criterion_details"] = benchmark(
            result["criterion_details"], results)
        result["ppi"] = calculate_ppi(result["criterion_details"], criteria)

    ranked = rank_results(results)
    risk_radar = build_risk_radar(ranked)
    persist_result(db_path, run_id, ranked)

    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "generated_at": datetime.now().isoformat(),
        "criteria": criteria,
        "leaderboard": [
            {
                "rank": r["final_rank"],
                "supplier_name": r["supplier_name"],
                "absolute_score": r["absolute_score"],
                "ppi": r["ppi"],
                "experience_rating": r["experience_rating"],
                "submission_date": r["submission_date"],
            }
            for r in ranked
        ],
        "scorecards": [
            {
                "rank": r["final_rank"],
                "supplier_name": r["supplier_name"],
                "absolute_score": r["absolute_score"],
                "ppi": r["ppi"],
                "experience_rating": r["experience_rating"],
                "criteria": r["criterion_details"],
                "risks": r["evaluation"].get("risks", []),
                "overall_summary": r["evaluation"].get("overall_summary", ""),
                "validation_warnings": r.get("validation_warnings", []),
            }
            for r in ranked
        ],
        "peer_benchmarks": benchmarks,
        "risk_radar": risk_radar,
        "warnings": warnings,
    }


def export_csv(result: dict[str, Any]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank", "Supplier", "Absolute Score", "PPI (%)",
        "Experience", "Submission Date"
    ])
    for row in result["leaderboard"]:
        writer.writerow([
            row["rank"], row["supplier_name"], row["absolute_score"],
            row["ppi"], row["experience_rating"], row["submission_date"]
        ])
    return output.getvalue().encode("utf-8")
