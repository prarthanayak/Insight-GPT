# app.py — InsightGPT (Groq-ready, final)
# Run with: python -m streamlit run app.py
# FEATURES:
# - Streamlit UI for CSV upload
# - Robust encoding fallback (utf-8, latin1, cp1252)
# - Schema detection, cleaning, EDA, anomaly detection
# - Groq HTTP LLM integration (free tier)
# - Download cleaned CSV

SAMPLE_DEMO_IMAGE_PATH = "/mnt/data/cc62d9d2-5c35-4513-813e-d2c17771af37.png"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import requests
import json
import re
from sklearn.ensemble import IsolationForest

# ---------- CONFIG ----------
st.set_page_config(layout="wide", page_title="InsightGPT — Groq")

# ---------- IMPORTANT FIX ----------
# Correct Groq OpenAI-compatible HTTP endpoint (note the '/openai/' path)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Valid example models (choose one that your Groq account supports)
DEFAULT_MODEL = "llama-3.1-8b-instant"

# ---------- HELPERS ----------
def detect_column_types(df, n_unique_thresh=30):
    types = {}
    for c in df.columns:
        ser = df[c]
        try:
            if pd.api.types.is_datetime64_any_dtype(ser):
                types[c] = "datetime"
                continue
        except Exception:
            pass
        try:
            if pd.api.types.is_numeric_dtype(ser):
                types[c] = "numeric"
                continue
        except Exception:
            pass
        nunique = ser.dropna().nunique()
        avg_len = ser.dropna().astype(str).map(len).mean() if nunique > 0 else 0
        if nunique > n_unique_thresh or (avg_len and avg_len > 50):
            types[c] = "text"
        else:
            types[c] = "categorical"
    return types

def basic_clean(df, types):
    df = df.copy()
    df = df.drop_duplicates()
    for c, t in types.items():
        if t == "datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c, t in types.items():
        if t == "numeric":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median()
            df[c].fillna(med, inplace=True)
        elif t == "categorical":
            df[c] = df[c].astype(object)
            mode = df[c].mode()
            if len(mode) > 0:
                df[c].fillna(mode[0], inplace=True)
            else:
                df[c].fillna("Unknown", inplace=True)
        elif t == "text":
            df[c] = df[c].astype(str).fillna("")
    return df

def summarize_numeric(df, col):
    s = df[col].describe()
    iqr = s.get("75%", 0) - s.get("25%", 0)
    lower = s.get("25%", None)
    upper = s.get("75%", None)
    outliers = pd.Series([], dtype=df[col].dtype)
    try:
        if lower is not None and upper is not None:
            lower_bound = s["25%"] - 1.5 * iqr
            upper_bound = s["75%"] + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    except Exception:
        outliers = pd.Series([], dtype=df[col].dtype)
    return {
        "count": int(s.get("count", 0)),
        "mean": float(s.get("mean") if not pd.isna(s.get("mean")) else 0.0),
        "std": float(s.get("std") if not pd.isna(s.get("std")) else 0.0),
        "min": float(s.get("min") if not pd.isna(s.get("min")) else 0.0),
        "25%": float(s.get("25%") if not pd.isna(s.get("25%")) else 0.0),
        "50%": float(s.get("50%") if not pd.isna(s.get("50%")) else 0.0),
        "75%": float(s.get("75%") if not pd.isna(s.get("75%")) else 0.0),
        "max": float(s.get("max") if not pd.isna(s.get("max")) else 0.0),
        "outlier_count": int(outliers.count()),
    }

def top_categories(df, col, n=5):
    try:
        return df[col].value_counts().head(n).to_dict()
    except Exception:
        return {}

def correlation_matrix(df, numeric_cols):
    try:
        return df[numeric_cols].corr()
    except Exception:
        return pd.DataFrame()

def detect_anomalies(df, numeric_cols):
    if len(numeric_cols) < 1:
        return pd.Series([False] * len(df), index=df.index)
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    X = df[numeric_cols].fillna(0).values
    preds = iso.fit_predict(X)
    return pd.Series(preds == -1, index=df.index)

# ---------- Groq LLM wrapper ----------
def build_llm_prompt(brief_summary, top_insights, suggested_actions):
    prompt = f"""
You are an expert business analyst. Given the following brief dataset summary and top insights, produce 3 concise, prioritized business recommendations (each 1 sentence) and a 2-sentence executive summary.

Dataset summary:
{brief_summary}

Top insights:
{top_insights}

Suggested actions derived from analytics:
{suggested_actions}

Return JSON with keys: executive_summary, recommendations (list of 3)."""
    return prompt


def call_llm(prompt, model=DEFAULT_MODEL, max_tokens=300, api_key=None):
    """
    Call Groq HTTP chat completions (OpenAI-compatible path).
    - api_key: your Groq API key (paste into Streamlit UI)
    - model: pick from Groq console (default llama-3.1-8b-instant)
    """
    if not api_key:
        raise RuntimeError("No Groq API key provided. Paste your Groq key into the Streamlit sidebar field 'Groq API Key'.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.2}

    try:
        resp = requests.post(GROQ_API_URL, json=body, headers=headers, timeout=30)
    except requests.Timeout:
        raise RuntimeError("LLM request timed out. Try again or reduce max_tokens.")
    except requests.RequestException as e:
        raise RuntimeError(f"Network error while calling LLM: {e}")

    # helpful debugging messages for common failures
    if resp.status_code in (401, 403):
        raise RuntimeError("Authentication failed (401/403). Check your Groq API key and that the key has chat/completions permission.")
    if resp.status_code == 404:
        raise RuntimeError("404 from Groq — check GROQ_API_URL. It should be: https://api.groq.com/openai/v1/chat/completions")
    if resp.status_code == 429:
        raise RuntimeError("Rate limit or quota exceeded (429). Try again later or check your Groq quota.")

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(f"LLM HTTP error {resp.status_code}: {resp.text}")

    try:
        j = resp.json()
    except Exception:
        return resp.text

    # OpenAI-compatible shape: choices -> message -> content
    if isinstance(j, dict) and j.get("choices"):
        text = ""
        for choice in j.get("choices", []):
            msg = choice.get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text") or ""
            else:
                content = ""
            text += content
        if text.strip():
            return text.strip()

    # Groq-specific/alternative: output or text fields
    if isinstance(j, dict) and j.get("output"):
        out = j.get("output")
        if isinstance(out, list):
            return "\n".join(map(str, out))
        return str(out)

    if isinstance(j, dict) and j.get("text"):
        return str(j.get("text"))

    return str(j)

# ---------- tolerant JSON parsing + UI helper ----------
def parse_llm_json_like(text: str):
    """
    Try to extract and parse JSON from the LLM output.
    Returns (parsed_dict or None, error_message or None).
    Handles cases where the LLM prepends explanation text or returns single quotes.
    """
    if not text or not isinstance(text, str):
        return None, "No text to parse."

    # 1) Try direct json.loads
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    # 2) Try to find a JSON substring with regex (from first { to matching last })
    m = re.search(r'(\{.*\})', text, flags=re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate), None
        except Exception:
            try:
                candidate_fixed = candidate.replace("'", '"')
                return json.loads(candidate_fixed), None
            except Exception:
                return None, "Found JSON-like content but could not parse it."

    # 3) Try a safe normalization (replace Python bool/None tokens)
    safe = text.strip()
    safe = safe.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(safe), None
    except Exception:
        pass

    return None, "Could not extract JSON from the LLM output."

# ---------- MAIN STREAMLIT UI ----------
st.title("InsightGPT-Automated Business Insights (Groq)")

with st.expander("LLM / API Settings (required for recommendations)"):
    st.write("Get a free Groq API key at: https://console.groq.com/keys")
    groq_key = st.text_input("Groq API Key", type="password")
    model_choice = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    raw = uploaded.read()
    encodings_to_try = ["utf-8", "latin1", "cp1252"]
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            st.info(f"Loaded file using encoding: {enc}")
            break
        except Exception:
            df = None
    if df is None:
        st.error("Could not read the CSV with common encodings (utf-8, latin1, cp1252). Save the file as UTF-8 and try again.")
    else:
        st.sidebar.header("Dataset info")
        st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        types = detect_column_types(df)
        st.sidebar.write(types)

        st.markdown("## Raw data (first 50 rows)")
        st.dataframe(df.head(50))

        df_clean = basic_clean(df, types)
        st.markdown("## Automatic Cleaning: preview")
        st.dataframe(df_clean.head(20))

        numeric_cols = [c for c, t in types.items() if t == "numeric"]
        categorical_cols = [c for c, t in types.items() if t == "categorical"]
        text_cols = [c for c, t in types.items() if t == "text"]
        st.markdown("### Column counts")
        st.write(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Text: {len(text_cols)}")

        st.markdown("## Top-level Insights")
        insights = []
        for c in numeric_cols[:8]:
            try:
                s = summarize_numeric(df_clean, c)
                insights.append((c, s))
                st.write(f"**{c}** — mean {s['mean']:.3f}, std {s['std']:.3f}, outliers {s['outlier_count']}")
            except Exception:
                pass

        if len(numeric_cols) >= 2:
            corr = correlation_matrix(df_clean, numeric_cols)
            if not corr.empty:
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
                st.plotly_chart(fig, use_container_width=True)

        for c in categorical_cols[:3]:
            st.write(f"Top categories for **{c}**")
            top = top_categories(df_clean, c, n=5)
            if top:
                st.bar_chart(pd.Series(top))

        anomalies_mask = detect_anomalies(df_clean, numeric_cols)
        n_anom = int(anomalies_mask.sum())
        st.write(f"Detected {n_anom} potential anomalies (IsolationForest).")
        if n_anom > 0:
            st.dataframe(df_clean[anomalies_mask].head(20))

        brief = f"Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}. Numeric cols: {numeric_cols[:6]}. Cat cols: {categorical_cols[:6]}."
        top_insight_lines = []
        for k, s in insights[:5]:
            top_insight_lines.append(f"{k} mean={s['mean']:.2f}, std={s['std']:.2f}, outliers={s['outlier_count']}")
        top_insights = "\n".join(top_insight_lines)
        suggested_actions = "Consider investigating columns with many outliers, segment top categories, and run root-cause for top correlated pairs."

        st.markdown("## LLM Recommendations (Groq)")

        # ----------------------
        # Replaces previous "Generate recommendations from LLM" block
        # with tolerant JSON parsing & friendly UI rendering
        # ----------------------
        if st.button("Generate recommendations from LLM"):
            with st.spinner("Calling Groq..."):
                try:
                    prompt = build_llm_prompt(brief, top_insights, suggested_actions)
                    raw_out = call_llm(prompt, model=model_choice, api_key=groq_key)

                    # show raw output collapsed so user can inspect if needed
                    with st.expander("Raw LLM output (expand to inspect)"):
                        st.code(raw_out)

                    parsed, err = parse_llm_json_like(raw_out)
                    if parsed and isinstance(parsed, dict):
                        exec_sum = parsed.get("executive_summary") or parsed.get("summary") or parsed.get("Executive summary") or ""
                        recs = parsed.get("recommendations") or parsed.get("recommendation") or parsed.get("recommendations_list") or []

                        # executive summary card-like display
                        st.markdown("### Executive summary")
                        if exec_sum:
                            st.info(exec_sum)
                        else:
                            st.write("_No executive_summary field found in JSON._")

                        st.markdown("### Recommendations")
                        if isinstance(recs, list) and len(recs) > 0:
                            for i, r in enumerate(recs, start=1):
                                st.markdown(f"**{i}.** {r}")
                        elif isinstance(recs, str) and recs.strip():
                            st.markdown(f"1. {recs}")
                        else:
                            st.write("_No recommendations list found in JSON._")

                        # allow download of the parsed JSON
                        st.download_button("Download recommendations (JSON)", data=json.dumps(parsed, indent=2), file_name="insights.json", mime="application/json")

                    else:
                        st.warning("Could not parse structured JSON from LLM output.")
                        if err:
                            st.info(err)
                        st.button("Retry LLM call")
                        st.download_button("Download raw LLM output", data=str(raw_out), file_name="llm_raw.txt", mime="text/plain")

                except Exception as e:
                    st.error(f"LLM call failed: {e}")

        towrite = io.BytesIO()
        df_clean.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button("Download cleaned CSV", data=towrite, file_name="cleaned.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: Do not commit API keys to GitHub. Use the Groq console to rotate or revoke keys.")
