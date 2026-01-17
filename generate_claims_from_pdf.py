#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#& "C:\Users\krati\AppData\Local\Programs\Python\Python313\python.exe" .\generate_claims_from_pdf.py --pdf ".\basics_of_evs.pdf" --out "claims.json" --labels-out "claims_with_labels.csv" --generator-model "deepseek/deepseek-chat" --total-claims 525


import argparse
import csv
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import requests

# ---------------------------
# Safe write helper
# ---------------------------

def safe_write_json(path, data):
    """
    Atomic write: write to a temp file then replace (Windows-safe).
    """
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix="claims_", suffix=".tmp", dir=d)
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------
# PDF extraction (robust)
# ---------------------------

def extract_pdf_text(pdf_path: str, start_page: int = 1, max_pages: Optional[int] = None) -> str:
    """
    Extract text from PDF using pypdf first, fallback to pdfplumber.
    """
    text = ""
    start_idx = max(0, start_page - 1)
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if start_idx >= total_pages:
            return ""

        end_idx = total_pages if max_pages is None else min(total_pages, start_idx + max_pages)
        for i in range(start_idx, end_idx):
            t = reader.pages[i].extract_text() or ""
            text += "\n" + t
    except Exception:
        pass

    if not text.strip():
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if start_idx >= total_pages:
                    return ""

                end_idx = total_pages if max_pages is None else min(total_pages, start_idx + max_pages)
                for p in pdf.pages[start_idx:end_idx]:
                    t = p.extract_text() or ""
                    text += "\n" + t
        except Exception:
            pass

    return text or ""


# ---------------------------
# Text cleanup
# ---------------------------

REPLACEMENTS = {
    "ï¬": "fi", "ﬁ": "fi",
    "ï¬‚": "fl", "ﬂ": "fl",
    "â€™": "'", "’": "'",
    "â€œ": '"', "â€�": '"', "“": '"', "”": '"',
    "â€“": "-", "–": "-", "—": "-", "â€”": "-",
    "â€¦": "...",
    "Â": "",
    "œ": "oe",
    "´": "'",
}

def clean_text(s: str) -> str:
    for bad, good in REPLACEMENTS.items():
        s = s.replace(bad, good)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Remove very long dotted headings or leaders
    s = re.sub(r"\.{5,}", " ", s)
    return s.strip()


# ---------------------------
# Chunking
# ---------------------------

def chunk_text(text: str, chunk_chars: int = 2500, overlap: int = 200) -> List[str]:
    """
    Simple robust chunker by characters (keeps paragraph boundaries when possible).
    """
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    cur = ""

    for p in paras:
        if len(cur) + len(p) + 1 <= chunk_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) > chunk_chars:
                for i in range(0, len(p), chunk_chars - overlap):
                    chunks.append(p[i:i + (chunk_chars - overlap)])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)

    return chunks


# ---------------------------
# Claim parsing & filtering
# ---------------------------


@dataclass
class ClaimCandidate:
    text: str
    label_hint: Optional[str] = None


def normalize_label_hint(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    label = str(raw).strip().lower()
    if label in {"true", "supported", "yes"}:
        return "true"
    if label in {"false", "unsupported", "no"}:
        return "false"
    return None


def parse_claims_from_model_output(txt: str) -> List[ClaimCandidate]:
    """
    Try JSON first: {"claims": [{"text": "...", "label": "true"}]}, else fall back to numbered/bulleted lines.
    """
    candidates: List[ClaimCandidate] = []

    # Try JSON
    try:
        candidate = txt.strip()
        m = re.search(r"\{.*\}", candidate, flags=re.S)
        if m:
            candidate = m.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "claims" in obj and isinstance(obj["claims"], list):
            for item in obj["claims"]:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    label = normalize_label_hint(item.get("label"))
                else:
                    text = str(item).strip()
                    label = None
                if text:
                    candidates.append(ClaimCandidate(text=text, label_hint=label))
            return candidates
    except Exception:
        pass

    # Fallback: parse lines starting with digits or dashes
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    for ln in lines:
        ln = re.sub(r"^\s*(?:-|\*|\d+[\).\]]?)\s*", "", ln)
        if ln:
            candidates.append(ClaimCandidate(text=ln, label_hint=None))
    return candidates


def postprocess_model_claim_text(s: str) -> str:
    s = s.strip()
    # strip wrapping quotes at start/end
    s = re.sub(r'^[\'"]+', '', s)
    s = re.sub(r'[\'"]+[,\.]*$', '', s)
    # strip trailing ",." / ".,." / ",.," etc
    s = re.sub(r'[,\.]+\s*$', '', s)
    # normalize funky spacing/encodings
    s = clean_text(s)
    return s


def is_atomic_claim(s: str) -> bool:
    s = s.strip()
    if len(s) < 30 or len(s) > 220:
        return False
    if s.endswith("?"):
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    if not s.endswith("."):
        s += "."
    if re.fullmatch(r"[A-Z\s\-:]+\.?", s):
        return False
    if "http://" in s or "https://" in s or "www." in s:
        return False
    if re.search(r"%\d{2,}", s):  # junk like %679
        return False
    if re.match(r"^(see|figure|table|chapter|section)\b", s.strip().lower()):
        return False
    if s.count(";") > 0:
        return False
    if s.count(",") > 3:
        return False
    return True


def normalize_claim(s: str) -> str:
    s = clean_text(s)
    if not s.endswith("."):
        s += "."
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


def dedupe_claims(items: List[ClaimCandidate]) -> List[ClaimCandidate]:
    seen: dict[str, ClaimCandidate] = {}
    out: List[ClaimCandidate] = []
    for x in items:
        key = x.text.lower()
        existing = seen.get(key)
        if existing is None:
            seen[key] = x
            out.append(x)
        elif existing.label_hint is None and x.label_hint is not None:
            existing.label_hint = x.label_hint
    return out


def claims_to_serializable(items: Sequence[ClaimCandidate]) -> List[dict]:
    return [{"text": c.text, "label_hint": c.label_hint} for c in items]


def claims_from_serialized(data: Sequence) -> List[ClaimCandidate]:
    out: List[ClaimCandidate] = []
    for item in data:
        if isinstance(item, str):
            out.append(ClaimCandidate(text=item, label_hint=None))
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                out.append(ClaimCandidate(text=text, label_hint=normalize_label_hint(item.get("label_hint"))))
    return out


def claim_candidates_to_texts(items: Sequence[ClaimCandidate]) -> List[str]:
    return [c.text for c in items]


def write_plain_claims_csv(path: str, items: Sequence[ClaimCandidate]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim", "label_hint"])
        for c in items:
            writer.writerow([c.text, c.label_hint or ""])


# ---------------------------
# Verification helpers
# ---------------------------

VERIFIER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z]+", text.lower()) if t not in VERIFIER_STOPWORDS}


def score_chunk(claim: str, chunk: str) -> float:
    claim_tokens = token_set(claim)
    chunk_tokens = token_set(chunk)
    if not claim_tokens or not chunk_tokens:
        overlap = 0.0
    else:
        matches = len(claim_tokens & chunk_tokens)
        overlap = matches / max(1, len(claim_tokens))
    seq_ratio = SequenceMatcher(None, claim.lower(), chunk.lower()).ratio()
    return 0.6 * overlap + 0.4 * seq_ratio


def select_context(claim: str, chunks: Sequence[str], top_k: int = 2) -> str:
    if not chunks:
        return ""
    scored = sorted(
        ((score_chunk(claim, chunk), chunk) for chunk in chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )
    selected = [chunk for score, chunk in scored[:top_k] if score > 0]
    if not selected:
        selected = [scored[0][1]]
    return "\n\n".join(selected)


def call_openrouter(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    referer: Optional[str] = None,
    title: Optional[str] = None,
    timeout: float = 60.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def parse_verifier_label(text: str) -> str:
    cleaned = text.strip().lower()
    if cleaned in {"true", "false"}:
        return cleaned
    match = re.search(r"\b(true|false)\b", cleaned)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse label from verifier response: {text!r}")


def verify_claims(
    claims: Sequence[ClaimCandidate],
    chunks: Sequence[str],
    *,
    api_key: str,
    model: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
    sleep_seconds: float,
    referer: Optional[str],
    title: Optional[str],
    checkpoint_path: str,
    out_path: str,
    labels_path: str,
    desired_total: int,
) -> Tuple[List[dict], List[dict]]:
    system_prompt = (
        "You are a meticulous fact checker. "
        "When the supplied context clearly supports the claim, respond with 'true'. "
        "If the context contradicts the claim or lacks enough information, respond with 'false'. "
        "Always answer with a single word."
    )

    records_map = load_verified_records(checkpoint_path)
    selected_cache: List[dict] = []

    def persist() -> None:
        nonlocal selected_cache
        records_list = list(records_map.values())
        safe_write_json(checkpoint_path, records_list)
        selected_cache = select_balanced_subset(records_list, desired_total)
        safe_write_json(out_path, [item["text"] for item in selected_cache])
        write_labeled_csv(labels_path, selected_cache)

    # ensure any existing progress is flushed to the desired output files
    persist()

    total = len(claims)
    for idx, candidate in enumerate(claims, start=1):
        key = candidate.text.lower()
        existing = records_map.get(key)
        if existing and existing.get("label") in {"true", "false"}:
            if existing.get("hint") is None and candidate.label_hint:
                existing["hint"] = candidate.label_hint
                records_map[key] = existing
            print(f"[Verifier {idx}/{total}] cached label -> {existing.get('label')}")
            persist()
            continue

        context = select_context(candidate.text, chunks, top_k=top_k)
        user_prompt = (
            f"Claim:\n{candidate.text}\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Answer 'true' only if the context proves the claim; otherwise answer 'false'."
        )
        try:
            response_text = call_openrouter(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                referer=referer,
                title=title,
            )
            label = parse_verifier_label(response_text)
        except Exception as exc:
            print(f"[Verifier {idx}/{total}] error: {exc}", file=sys.stderr)
            label = "error"
            response_text = str(exc)

        hint = candidate.label_hint
        if hint is None and existing:
            hint = existing.get("hint")
        records_map[key] = {
            "text": candidate.text,
            "label": label,
            "verifier_response": response_text,
            "hint": hint,
        }
        persist()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    records_list = list(records_map.values())
    selected_cache = select_balanced_subset(records_list, desired_total)
    safe_write_json(out_path, [item["text"] for item in selected_cache])
    write_labeled_csv(labels_path, selected_cache)
    safe_write_json(checkpoint_path, records_list)
    return records_list, selected_cache


def select_balanced_subset(claims: Sequence[dict], total: int) -> List[dict]:
    valid = [c for c in claims if c.get("label") in {"true", "false"}]
    if not valid:
        return []

    target_true = total // 2
    target_false = total - target_true

    selected: List[dict] = []
    true_count = 0
    false_count = 0
    remainder: List[dict] = []

    for claim in valid:
        label = claim["label"]
        if label == "true" and true_count < target_true:
            selected.append(claim)
            true_count += 1
        elif label == "false" and false_count < target_false:
            selected.append(claim)
            false_count += 1
        else:
            remainder.append(claim)

    for claim in remainder:
        if len(selected) >= min(total, len(valid)):
            break
        selected.append(claim)

    return selected[: min(total, len(valid))]


def load_verified_records(path: str) -> "OrderedDict[str, dict]":
    records: "OrderedDict[str, dict]" = OrderedDict()
    if not os.path.exists(path):
        return records
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            return records
        for item in data:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            key = text.lower()
            if key not in records or records[key].get("label") == "error":
                records[key] = {
                    "text": text,
                    "label": item.get("label"),
                    "verifier_response": item.get("verifier_response"),
                    "hint": item.get("hint"),
                }
    except Exception as exc:
        print(f"Warning: could not load verifier checkpoint ({exc}). Starting fresh.", file=sys.stderr)
    return records


def write_labeled_csv(path: str, items: Sequence[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim", "label"])
        for item in items:
            writer.writerow([item.get("text", ""), item.get("label", "")])


# ---------------------------
# Prompts
# ---------------------------

SYSTEM_PROMPT = """You extract labelled factual claims from long texts.
Respond with JSON ONLY using the schema: {"claims": [{"text": "...", "label": "true|false"}]} with no additional text."""

USER_INSTRUCTION_TEMPLATE = """You are given a chunk of a PDF.
Produce up to {k} claims that obey ALL of the following rules:
1) Declarative sentences only; 12–40 words; atomic and unambiguous.
2) Stay on topic and avoid headings, lists, references, tables, or subjective opinions.
3) Include BOTH supported ("true") and intentionally incorrect ("false") claims:
   - Label "true" only if the text clearly supports the claim.
   - Label "false" for plausible-sounding statements that are contradicted or not supported by the text.
   - Aim for at least 40% true and 40% false claims.
4) Do not duplicate claims across labels.
5) Output valid JSON ONLY in the exact form:
   {{"claims": [{{"text": "...", "label": "true"}}, ...]}}

Text:
\"\"\"{chunk}\"\"\""""


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", default="claims.json", help="Output JSON file containing the selected claims")
    ap.add_argument("--labels-out", default="claims_with_labels.csv", help="CSV file with claims and true/false labels")
    ap.add_argument("--total-claims", type=int, default=525, help="Number of claims to include in the final dataset")
    ap.add_argument("--generator-model", default="deepseek/deepseek-chat", help="OpenRouter model for claim generation")
    ap.add_argument("--generator-api-key", default=None, help="API key for generation (defaults to OPENROUTER_API_KEY)")
    ap.add_argument("--generator-referer", default=None, help="Optional HTTP-Referer header for generation calls")
    ap.add_argument("--generator-title", default="Claim Generation", help="Optional X-Title header for generation calls")
    ap.add_argument("--generator-max-tokens", type=int, default=768, help="Max tokens for each generation response")
    ap.add_argument("--generator-temperature", type=float, default=0.2, help="Sampling temperature for generation")
    ap.add_argument("--generator-sleep", type=float, default=1.0, help="Delay between generation API calls (seconds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pages", type=int, default=None, help="Optionally limit pages for faster runs")
    ap.add_argument("--start-page", type=int, default=18, help="1-indexed page to start reading from")
    ap.add_argument("--verifier-model", default="deepseek/deepseek-chat", help="OpenRouter model for claim verification")
    ap.add_argument("--verifier-api-key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    ap.add_argument("--verifier-referer", default=None, help="Optional HTTP-Referer header for OpenRouter")
    ap.add_argument("--verifier-title", default="Claim Verification", help="Optional X-Title header for OpenRouter")
    ap.add_argument("--verifier-top-k", type=int, default=2, help="Number of context chunks to send to the verifier")
    ap.add_argument("--verifier-temperature", type=float, default=0.0, help="Sampling temperature for the verifier model")
    ap.add_argument("--verifier-max-tokens", type=int, default=64, help="Max tokens for the verifier completion")
    ap.add_argument("--verifier-sleep", type=float, default=1.0, help="Delay between verifier API calls (seconds)")
    args = ap.parse_args()

    random.seed(args.seed)

    raw_text = extract_pdf_text(args.pdf, start_page=args.start_page, max_pages=args.pages)
    if not raw_text.strip():
        print("ERROR: Could not extract any text from the PDF.", file=sys.stderr)
        sys.exit(1)

    text = clean_text(raw_text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)  # remove page numbers
    text = re.sub(r"(CHAPTER|SECTION|THE\s+[A-Z][A-Z\s]+)\s+\d*\b", "", text)

    chunks = chunk_text(text, chunk_chars=2600, overlap=200)
    if not chunks:
        print("ERROR: No chunks created from the PDF.", file=sys.stderr)
        sys.exit(1)

    print(f"Prepared {len(chunks)} chunks.")

    generator_api_key = (
        args.generator_api_key
        or args.verifier_api_key
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not generator_api_key:
        print(
            "ERROR: Missing OpenRouter API key for generation (set --generator-api-key or OPENROUTER_API_KEY).",
            file=sys.stderr,
        )
        sys.exit(1)

    generator_referer = args.generator_referer or args.verifier_referer
    generator_title = args.generator_title or "Claim Generation"

    # checkpoint + resume metadata
    desired_total = max(1, args.total_claims)
    collection_goal = max(desired_total, int(math.ceil(desired_total * 1.2)))
    per_chunk = 12
    base = os.path.splitext(args.out)[0]
    checkpoint_path = base + "_partial.json"   # rolling list of claims
    state_path = base + "_state.json"          # resume info
    raw_claims_json_path = base + "_claims_progress.json"
    raw_claims_csv_path = base + "_claims_progress.csv"
    verified_checkpoint_path = base + "_verified.json"

    # Try to resume
    claims_all: List[ClaimCandidate] = []
    start_idx = 1
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            claims_all = claims_from_serialized(st.get("claims", []))
            start_idx = int(st.get("next_idx", 1))

            # sanity check: if resume says skip far but we have almost no claims, don't trust it
            if start_idx > 1 and len(claims_all) < 5:
                print("State looks inconsistent (high next_idx but very few claims). Starting clean.")
                claims_all = []
                start_idx = 1

            print(f"Resuming from checkpoint: {len(claims_all)} claims, starting at chunk {start_idx}.")
        except Exception as e:
            print(f"Warning: could not load state ({e}); starting fresh.")

    try:
        # MAIN EXTRACTION LOOP
        for idx in range(start_idx, len(chunks) + 1):
            if len(claims_all) >= collection_goal:
                break

            chunk = chunks[idx - 1]

            # build prompt for the model
            k = per_chunk
            user_prompt = USER_INSTRUCTION_TEMPLATE.format(k=k, chunk=chunk)

            # generate model output via OpenRouter
            try:
                out = call_openrouter(
                    api_key=generator_api_key,
                    model=args.generator_model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=args.generator_temperature,
                    max_tokens=args.generator_max_tokens,
                    referer=generator_referer,
                    title=generator_title,
                )
            except Exception as e:
                print(f"[Chunk {idx}] Generation error: {e}", file=sys.stderr)

                # save progress even on error so we can resume
                safe_write_json(checkpoint_path, claims_to_serializable(claims_all))
                safe_write_json(state_path, {
                    "claims": claims_to_serializable(claims_all),
                    "next_idx": idx + 1
                })
                safe_write_json(raw_claims_json_path, claims_to_serializable(claims_all))
                write_plain_claims_csv(raw_claims_csv_path, claims_all)
                continue

            if args.generator_sleep > 0:
                time.sleep(args.generator_sleep)

            # DEBUG PRINT (helps diagnose if model is just echoing instructions)
            print(f"\n=== RAW MODEL OUTPUT FOR CHUNK {idx} (first 800 chars) ===")
            print(out[:800])
            print("=== END RAW MODEL OUTPUT ===\n")

            # parse + clean
            raw_claims = parse_claims_from_model_output(out)
            cleaned: List[ClaimCandidate] = []
            for candidate in raw_claims:
                text = postprocess_model_claim_text(candidate.text)
                if not text:
                    continue
                if is_atomic_claim(text):
                    normalized = normalize_claim(text)
                    cleaned.append(ClaimCandidate(text=normalized, label_hint=candidate.label_hint))

            # update collected claims
            before_ct = len(claims_all)
            claims_all.extend(cleaned)
            claims_all = dedupe_claims(claims_all)
            after_ct = len(claims_all)

            # save after each chunk
            safe_write_json(checkpoint_path, claims_to_serializable(claims_all))
            safe_write_json(state_path, {
                "claims": claims_to_serializable(claims_all),
                "next_idx": idx + 1
            })
            safe_write_json(raw_claims_json_path, claims_to_serializable(claims_all))
            write_plain_claims_csv(raw_claims_csv_path, claims_all)

            print(f"[Chunk {idx}] Parsed {len(raw_claims)} -> kept {len(cleaned)} -> total {after_ct} (added {after_ct - before_ct})")

        # TOP-UP PHASE if still short
        if len(claims_all) < collection_goal:
            print(f"Only {len(claims_all)} claims collected; topping up with stricter generation...")
            need = collection_goal - len(claims_all)
            big_chunks = sorted(chunks, key=len, reverse=True)[:min(10, len(chunks))]
            for jdx, chunk in enumerate(big_chunks, start=1):
                if need <= 0:
                    break

                k = 15
                user_prompt = USER_INSTRUCTION_TEMPLATE.format(k=k, chunk=chunk)

                try:
                    out = call_openrouter(
                        api_key=generator_api_key,
                        model=args.generator_model,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=max(0.1, args.generator_temperature - 0.05),
                        max_tokens=max(768, args.generator_max_tokens),
                        referer=generator_referer,
                        title=generator_title,
                    )
                except Exception as e:
                    print(f"[Top-up {jdx}] Generation error: {e}", file=sys.stderr)
                    safe_write_json(checkpoint_path, claims_to_serializable(claims_all))
                    safe_write_json(state_path, {
                        "claims": claims_to_serializable(claims_all),
                        "next_idx": len(chunks) + 1
                    })
                    safe_write_json(raw_claims_json_path, claims_to_serializable(claims_all))
                    write_plain_claims_csv(raw_claims_csv_path, claims_all)
                    continue

                if args.generator_sleep > 0:
                    time.sleep(args.generator_sleep)

                print(f"\n=== RAW MODEL OUTPUT FOR TOP-UP {jdx} (first 800 chars) ===")
                print(out[:800])
                print("=== END RAW MODEL OUTPUT ===\n")

                raw_claims = parse_claims_from_model_output(out)
                cleaned: List[ClaimCandidate] = []
                for candidate in raw_claims:
                    text = postprocess_model_claim_text(candidate.text)
                    if not text:
                        continue
                    if is_atomic_claim(text):
                        normalized = normalize_claim(text)
                        cleaned.append(ClaimCandidate(text=normalized, label_hint=candidate.label_hint))

                before_ct = len(claims_all)
                claims_all.extend(cleaned)
                claims_all = dedupe_claims(claims_all)
                after_ct = len(claims_all)
                need = collection_goal - len(claims_all)

                safe_write_json(checkpoint_path, claims_to_serializable(claims_all))
                safe_write_json(state_path, {
                    "claims": claims_to_serializable(claims_all),
                    "next_idx": len(chunks) + 1
                })
                safe_write_json(raw_claims_json_path, claims_to_serializable(claims_all))
                write_plain_claims_csv(raw_claims_csv_path, claims_all)

                print(f"[Top-up {jdx}] +{after_ct - before_ct} -> total {after_ct} (need {max(0, need)})")

    except KeyboardInterrupt:
        print("Interrupted by user; saving current progress...")

    finally:
        # Always persist the latest snapshot even if interrupted
        safe_write_json(checkpoint_path, claims_to_serializable(claims_all))
        safe_write_json(state_path, {
            "claims": claims_to_serializable(claims_all),
            "next_idx": max(start_idx, 1 if not len(claims_all) else len(chunks) + 1)
        })
        safe_write_json(raw_claims_json_path, claims_to_serializable(claims_all))
        write_plain_claims_csv(raw_claims_csv_path, claims_all)

    if not claims_all:
        print("No claims were generated; aborting.", file=sys.stderr)
        sys.exit(1)

    verifier_api_key = (
        args.verifier_api_key
        or args.generator_api_key
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not verifier_api_key:
        print("ERROR: Missing OpenRouter API key for verification (set --verifier-api-key or OPENROUTER_API_KEY).", file=sys.stderr)
        sys.exit(1)

    print(f"Running verification on {len(claims_all)} claims using {args.verifier_model}...")
    verified, selected = verify_claims(
        claims_all,
        chunks,
        api_key=verifier_api_key,
        model=args.verifier_model,
        top_k=args.verifier_top_k,
        temperature=args.verifier_temperature,
        max_tokens=args.verifier_max_tokens,
        sleep_seconds=args.verifier_sleep,
        referer=args.verifier_referer,
        title=args.verifier_title,
        checkpoint_path=verified_checkpoint_path,
        out_path=args.out,
        labels_path=args.labels_out,
        desired_total=desired_total,
    )

    if len(selected) < desired_total:
        print(
            f"Warning: Only {len(selected)} verified claims available (target was {desired_total}).",
            file=sys.stderr,
        )

    claims_plain = [item["text"] for item in selected]
    true_ct = sum(1 for item in selected if item["label"] == "true")
    false_ct = sum(1 for item in selected if item["label"] == "false")
    print(f"Saved {len(claims_plain)} claims to {args.out}")
    print(f"Saved labelled dataset to {args.labels_out} (true={true_ct}, false={false_ct})")
    print(
        f"(Checkpoints: claims -> {checkpoint_path}, verifier -> {verified_checkpoint_path}, resume state -> {state_path})"
    )


if __name__ == "__main__":
    main()
