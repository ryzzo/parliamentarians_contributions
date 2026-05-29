"""
Kenya Hansard PDF → RAG chunks parser.
Produces two chunk types per topic:
  - full_topic   : entire topic conversation as one block
  - speaker_turn : individual utterances with speaker metadata
"""

import re
import json
import pdfplumber

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKIP_TOPICS = frozenset({
    "THE HANSARD", "QUESTIONS AND STATEMENTS", "REQUESTS FOR STATEMENTS",
    "REQUEST FOR STATEMENT", "COMMUNICATION FROM THE CHAIR", "STATEMENTS",
    "MOTIONS", "PRAYERS", "NEXT ORDER", "BUSINESS FOR THE WEEK",
    "PAPERS", "BILLS", "NOTICES OF MOTION", "ADJOURNMENT",
    "NOTING OF REPORT OF KENYA",
    "QUESTIONS AND STATEMENTS REQUEST FOR STATEMENT",
    "QUESTIONS AND STATEMENTS REQUESTS FOR STATEMENTS",
})

ORDINALS = (
    "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH",
    "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH", "THIRTEENTH",
    "FOURTEENTH", "FIFTEENTH",
)

# Single compiled pattern — no duplicate definitions
SPEAKER_RE = re.compile(
    r"((?:The\s+)?(?:Hon\.|Temporary Speaker|Deputy Speaker)"
    r"[\w\s.\(\),\-]*?"
    r"(?:\([\w\s,]+\))?)"
    r"\s*:",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def load_pdf(path: str) -> str:
    """Extract and join all page text from a PDF."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------------------------------------------------------
# Pre-processing (called once on the full document)
# ---------------------------------------------------------------------------

_PREPROCESS_SUBS = [
    # Disclaimer blocks
    (re.compile(r"Disclaimer:.*?Hansard Editor\.", re.DOTALL), ""),
    # Running page headers  e.g. "30th April 2026 National Assembly Debates 12"
    (re.compile(r"\d{1,2}\w{2}\s+\w+\s+\d{4}\s+National Assembly Debates\s+\d+"), ""),
    # Stage directions
    (re.compile(r"\[.*?\]", re.DOTALL), ""),
    (re.compile(r"\((?!Hon\.|The Temp|The Dep)[^)]{0,80}\)"), ""),
    # "Thank you, Hon. Speaker." closing lines that bleed into the next label
    (re.compile(r"Thank you,?\s*\n+Hon\.\s+(?:Deputy\s+)?Speaker\.?\s*\n"), "\n"),
    # Standalone "Hon. Speaker." lines
    (re.compile(r"\nHon\.\s+(?:Temporary\s+|Deputy\s+)?Speaker\.\s*\n"), "\n"),
]

def preprocess(text: str) -> str:
    for pattern, repl in _PREPROCESS_SUBS:
        text = pattern.sub(repl, text)
    return text

# ---------------------------------------------------------------------------
# Metadata extraction (run once per document)
# ---------------------------------------------------------------------------

def extract_date(text: str) -> str:
    m = re.search(r"\b(\d{1,2}(?:st|nd|rd|th)\s+\w+\s+\d{4})\b", text)
    return m.group(1) if m else "Unknown Date"


def extract_parliament_info(text: str) -> dict:
    parliament_re = re.compile(
        rf"({'|'.join(ORDINALS)})\s+PARLIAMENT"
    )
    p = parliament_re.search(text)
    c = re.search(r"(NATIONAL ASSEMBLY|SENATE)", text)
    return {
        "parliament": p.group(0) if p else "Unknown Parliament",
        "chamber": c.group(1) if c else "Unknown Chamber",
    }

# ---------------------------------------------------------------------------
# Speaker helpers
# ---------------------------------------------------------------------------

def extract_clean_speaker(raw: str) -> str:
    candidates = re.findall(
        r"((?:The\s+)?(?:Temporary Speaker|Deputy Speaker|Hon\.)\s+[\w\s.\(\),]+)",
        raw,
    )
    if candidates:
        label = candidates[-1].strip().rstrip(".,").split("\n")[0].strip()
        return label
    return raw.split("\n")[0].strip()


def classify_role(speaker: str) -> str:
    s = speaker.lower()
    if any(kw in s for kw in ("temporary speaker", "deputy speaker", "speaker")):
        return "presiding_officer"
    if "nominated" in s:
        return "nominated_member"
    m = re.search(r"\([\w\s]+,\s*([\w]+)\)", speaker)
    party = m.group(1).lower() if m else "unknown"
    return f"elected_member_{party}"


def extract_constituency(speaker: str) -> str | None:
    m = re.search(r"\(([\w\s]+),\s*[\w]+\)", speaker)
    return m.group(1).strip() if m else None

# ---------------------------------------------------------------------------
# Topic validation / cleaning
# ---------------------------------------------------------------------------

_SKIP_NORMALIZED = frozenset(" ".join(s.split()) for s in SKIP_TOPICS)


def is_valid_topic(topic: str) -> bool:
    normalized = " ".join(topic.upper().split())
    if normalized in _SKIP_NORMALIZED:
        return False
    if any(normalized.startswith(s) for s in _SKIP_NORMALIZED):
        return False
    return len(normalized.replace(" ", "")) >= 10


def clean_topic(raw_topic: str) -> str:
    lines = [l.strip() for l in raw_topic.strip().splitlines() if l.strip()]
    filtered = [l for l in lines if " ".join(l.upper().split()) not in _SKIP_NORMALIZED]
    return " ".join(filtered) if filtered else " ".join(lines)

# ---------------------------------------------------------------------------
# Turn parsing
# ---------------------------------------------------------------------------

def parse_turns(text: str) -> list[tuple[str, str]]:
    matches = list(SPEAKER_RE.finditer(text))
    if not matches:
        return []

    turns = []
    for i, match in enumerate(matches):
        raw_speaker = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        utterance = text[start:end].strip()

        if len(utterance) < 40:
            continue

        turns.append((extract_clean_speaker(raw_speaker), utterance))

    return turns

# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^[A-Z][A-Z\s,\-\n]{8,}$")
_SECTION_SPLIT_RE = re.compile(r"\n((?:[A-Z][A-Z\s,\-]+\n?){1,4})\n")

def parse_hansard(text: str) -> list[dict]:
    date = extract_date(text)
    parliament_info = extract_parliament_info(text)
    base_meta = {
        "date": date,
        "parliament": parliament_info["parliament"],
        "chamber": parliament_info["chamber"],
        "source": "Kenya Hansard",
    }

    chunks: list[dict] = []
    current_topic: str | None = None

    for section in _SECTION_SPLIT_RE.split(text):
        stripped = section.strip()
        if not stripped:
            continue

        # Detect section heading
        if _HEADING_RE.match(stripped) and len(stripped) < 300:
            current_topic = stripped
            continue

        if not current_topic:
            continue

        topic_label = clean_topic(current_topic)
        current_topic = None  # reset regardless of validity

        if not is_valid_topic(topic_label):
            continue

        turns = parse_turns(stripped)
        if not turns:
            continue

        # --- full_topic chunk ---
        speakers_seen: list[str] = []
        body_lines: list[str] = [f"Topic: {topic_label}\n"]
        for speaker, utterance in turns:
            body_lines.append(f"{speaker}: {utterance}\n")
            speakers_seen.append(speaker)

        chunks.append({
            "text": "\n".join(body_lines).strip(),
            "chunk_type": "full_topic",
            "metadata": {**base_meta, "topic": topic_label, "speakers": list(set(speakers_seen))},
        })

        # --- speaker_turn chunks ---
        for speaker, utterance in turns:
            chunks.append({
                "text": f"Topic: {topic_label}\n{speaker}: {utterance}",
                "chunk_type": "speaker_turn",
                "metadata": {
                    **base_meta,
                    "topic": topic_label,
                    "speaker": speaker,
                    "constituency": extract_constituency(speaker),
                    "role": classify_role(speaker),
                },
            })

    return chunks

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str = "data/hansard_rag_chunks.jsonl") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    full = sum(1 for c in chunks if c["chunk_type"] == "full_topic")
    speaker = sum(1 for c in chunks if c["chunk_type"] == "speaker_turn")
    print(f"Saved {len(chunks)} chunks ({full} topic, {speaker} speaker turns) → {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Parse a Kenya Hansard PDF into RAG chunks.")
    parser.add_argument("pdf", help="Path to the Hansard PDF file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .jsonl path (default: same folder as PDF, named hansard_rag_chunks.jsonl)",
    )
    parser.add_argument(
        "-t", "--txt",
        default=None,
        help="Output .txt path for extracted raw text (default: same folder as PDF, same name as PDF)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = args.output or str(pdf_path.parent / "hansard_rag_chunks.jsonl")
    txt_path    = args.txt    or str(pdf_path.with_suffix(".txt"))

    print(f"Reading {pdf_path} ...")
    raw_text = load_pdf(str(pdf_path))
    save_text(raw_text, txt_path)
    print(f"Raw text saved → {txt_path}")

    clean_text = preprocess(raw_text)
    chunks = parse_hansard(clean_text)
    save_chunks(chunks, output_path)
