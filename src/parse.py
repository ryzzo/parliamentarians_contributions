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
# Processing helpers
# ---------------------------------------------------------------------------

def process_one(pdf_path: "Path", output_dir: "Path", txt_dir: "Path | None") -> int:
    """Process a single PDF, writing outputs to the given directories.
    Returns the number of chunks produced."""
    from pathlib import Path

    stem = pdf_path.stem
    jsonl_path = output_dir / f"{stem}.jsonl"
    txt_path   = (txt_dir or output_dir) / f"{stem}.txt"

    print(f"  Reading {pdf_path.name} ...")
    raw_text = load_pdf(str(pdf_path))
    save_text(raw_text, str(txt_path))
    print(f"  Raw text → {txt_path}")

    clean_text = preprocess(raw_text)
    chunks = parse_hansard(clean_text)
    save_chunks(chunks, str(jsonl_path))
    return len(chunks)


def process_folder(input_dir: "Path", output_dir: "Path", txt_dir: "Path | None") -> None:
    from pathlib import Path

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if txt_dir:
        txt_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    failed = []
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")
        try:
            total_chunks += process_one(pdf_path, output_dir, txt_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(pdf_path.name)

    print(f"\nDone. {len(pdfs) - len(failed)}/{len(pdfs)} PDFs processed, "
          f"{total_chunks} total chunks.")
    if failed:
        print("Failed:", ", ".join(failed))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Parse Kenya Hansard PDF(s) into RAG chunks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to a single Hansard PDF file")
    group.add_argument("--input-dir", help="Folder containing multiple Hansard PDFs")

    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output folder for .jsonl chunk files (one per PDF)",
    )
    parser.add_argument(
        "--txt-dir",
        default=None,
        help="Folder for raw .txt extracts (default: same as --output-dir)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    txt_dir    = Path(args.txt_dir) if args.txt_dir else None

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if txt_dir:
            txt_dir.mkdir(parents=True, exist_ok=True)
        process_one(pdf_path, output_dir, txt_dir)
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")
        process_folder(input_dir, output_dir, txt_dir)
