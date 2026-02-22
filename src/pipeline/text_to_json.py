"""
Legal Text RAG Preprocessing Pipeline — Sakhi Edition
=======================================================
Processes concatenated legal PDF text files into clean, structured chunks
ready for vector embedding. No API calls needed — runs fully offline.

Usage:
    python legal_rag_pipeline.py --single_file "data/Personal_&_Family_Laws.txt"
    python legal_rag_pipeline.py --input_dir data --output_file legal_chunks.jsonl

Requirements:
    pip install tiktoken   (optional — falls back to word count if missing)

Output: legal_chunks.jsonl — one JSON object per line, ready to embed.
"""

import re
import json
import os
import argparse
from pathlib import Path

# ── Token counter ──────────────────────────────────────────────────────────────
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text):
        return int(len(text.split()) * 1.3)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Split concatenated file into individual Acts
# ══════════════════════════════════════════════════════════════════════════════

PDF_SEPARATOR = re.compile(
    r'={40,}\s*\n\s*📄\s*(.+?)\s*\n.*?\n={40,}',
    re.DOTALL
)

def split_into_acts(raw_text):
    """Split one big .txt file into [(act_name, act_text), ...] pairs."""
    matches = list(PDF_SEPARATOR.finditer(raw_text))
    if not matches:
        return [("Unknown Act", raw_text)]
    parts = []
    for i, match in enumerate(matches):
        act_name = match.group(1).strip().replace('.pdf', '').strip()
        start    = match.end()
        end      = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        parts.append((act_name, raw_text[start:end]))
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clean text
# ══════════════════════════════════════════════════════════════════════════════

def clean_act_text(text):
    text = remove_toc(text)
    text = remove_footnotes(text)           # FIX: now strips gazette blocks too
    text = remove_inline_markers(text)
    text = re.sub(r'^\s*\d{1,3}\s*$',    '', text, flags=re.MULTILINE)   # lone page numbers
    text = re.sub(r'^\s*[\*\•]+\s*$',    '', text, flags=re.MULTILINE)   # orphan * lines
    text = re.sub(r'^\s*[_\-=]{3,}\s*$', '', text, flags=re.MULTILINE)   # decorative lines
    text = re.sub(r'^.*📄.*$',            '', text, flags=re.MULTILINE)   # leftover PDF headers
    text = re.sub(r'\n{3,}',          '\n\n', text)                       # collapse blank lines
    return text.strip()


def remove_toc(text):
    """Drop Table of Contents — find where the real legal body starts."""
    real_body = re.search(r'\nTHE\s+.+?ACT[,\s]+\d{4}\s*\nACT\s+NO', text, re.IGNORECASE)
    if real_body:
        return text[real_body.start():]
    toc_start = re.search(r'ARRANGEMENT OF SECTIONS', text, re.IGNORECASE)
    if toc_start:
        after_toc     = text[toc_start.end():]
        chapter_start = re.search(r'\n(CHAPTER\s+[IVX]+|THE\s+\w+\s+ACT)', after_toc, re.IGNORECASE)
        if chapter_start:
            return text[:toc_start.start()] + after_toc[chapter_start.start():]
    return text


def remove_footnotes(text):
    """
    Remove all footnote and gazette noise:
    1. Amendment footnotes  — '1. Ins. by Act 20 of 1983...'
    2. Gazette notification lines — 'S.O. 2754(E), dated 12th September...'
    3. Vide notification lines
    4. Date-prefixed gazette entries — '1st April 2014 – S. 2(2)...'
    5. Asterisk footnote lines — '*. Vide Notification No...'
    """
    # Standard amendment footnotes
    keywords = r'(Ins\.|Subs\.|Omitted|Added|Renumbered|Rep\.|see|vide|ibid|w\.e\.f\.|w\.r\.e\.f\.)'
    text = re.compile(
        r'^\s*\d+[\.\)]\s+' + keywords + r'.+$',
        re.MULTILINE | re.IGNORECASE
    ).sub('', text)

    # Gazette notification lines containing S.O. numbers
    text = re.sub(r'^.*?S\.O\.\s*\d+.*?Gazette.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^.*?vide\s+notification\s+No\..*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^.*?see\s+Gazette\s+of\s+India.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Date-prefixed gazette lines: "1st April 2014 – S. 2(2), (7)..."
    text = re.sub(
        r'^\s*\d{1,2}(?:st|nd|rd|th)\s+\w+[,\s]+\d{4}\s*[-–]\s*S[\.\s]*\d+.*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )

    # Asterisk footnote lines: "*. Vide Notification..." or "* Herein give particulars..."
    text = re.sub(r'^\s*\*[\.\s].+$', '', text, flags=re.MULTILINE)

    # Wide spacing that precedes footnote blocks
    text = re.sub(r'\s{20,}\n', '\n', text)
    return text


def remove_inline_markers(text):
    """
    Clean inline amendment artifacts:
      2***          → (removed)
      3[some text]  → some text
      date4 as      → date as  (superscript footnote refs)
    """
    text = re.sub(r'\d+\*+',                                        '',    text)
    text = re.sub(r'\d+\[([^\]]*)\]',                               r'\1', text)
    text = re.sub(r'(?<=[a-zA-Z])\d+(?=\s|[,\.;\:\)\-])',          '',    text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Parse Chapters and Sections
# ══════════════════════════════════════════════════════════════════════════════

CHAPTER_PATTERN = re.compile(
    r'^(CHAPTER\s+[IVXLCDM\d]+)\s*\n(.+?)$',
    re.MULTILINE | re.IGNORECASE
)

# FIX: Section pattern now rejects titles that look like footnotes
# A real section title starts with a capital letter and does NOT contain
# amendment keywords like "subs.", "ins.", "omitted", "w.e.f.", "Act X of YYYY"
SECTION_PATTERN = re.compile(
    r'(?m)^(\d+[A-Z]?)\.\s{1,5}([A-Z][^\n]{5,80}?)(?:\.?\s*[―—–-])'
)

FOOTNOTE_TITLE_PATTERN = re.compile(
    r'(subs\.|ins\.|omitted|w\.e\.f\.|act\s+\d+\s+of\s+\d{4}|dated|notification|gazette|vide)',
    re.IGNORECASE
)

def is_valid_section_title(title):
    """Return False if the title looks like a footnote rather than a real section heading."""
    if FOOTNOTE_TITLE_PATTERN.search(title):
        return False
    # Reject if it starts with lowercase (genuine section titles are Title Case)
    if title and title[0].islower():
        return False
    return True


def parse_chapters_and_sections(text):
    """Return list of {chapter, section_number, section_title, text} dicts."""

    # Collect chapter heading positions
    chapter_positions = []
    for m in CHAPTER_PATTERN.finditer(text):
        chapter_positions.append({
            "pos":     m.start(),
            "chapter": f"{m.group(1).strip()} - {m.group(2).strip()}"
        })

    # Collect section heading positions — FIX: filter out footnote-style titles
    section_positions = []
    for m in SECTION_PATTERN.finditer(text):
        title = m.group(2).strip().rstrip('.')
        if not is_valid_section_title(title):
            continue
        section_positions.append({
            "pos":            m.start(),
            "section_number": m.group(1).strip(),
            "section_title":  title,
        })

    if not section_positions:
        return [{"chapter": "General", "section_number": "0",
                 "section_title": "Full Text", "text": text}]

    def chapter_at(pos):
        result = "General"
        for cp in chapter_positions:
            if cp["pos"] <= pos:
                result = cp["chapter"]
            else:
                break
        return result

    results = []
    for i, sec in enumerate(section_positions):
        start = sec["pos"]
        end   = section_positions[i + 1]["pos"] if i + 1 < len(section_positions) else len(text)
        results.append({
            "chapter":        chapter_at(sec["pos"]),
            "section_number": sec["section_number"],
            "section_title":  sec["section_title"],
            "text":           text[start:end].strip()
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Chunk sections that exceed token limit
# ══════════════════════════════════════════════════════════════════════════════

MAX_TOKENS = 512


def split_large_section(section_text):
    """
    Split oversized sections intelligently:
    1. Sub-section boundaries: (1), (2), (a), (b) …
    2. Fall back to paragraph breaks.
    """
    parts = re.split(r'(?=\n\s*\([a-z0-9]{1,2}\)\s)', section_text)

    merged = []
    current = ""
    for part in parts:
        candidate = current + part
        if count_tokens(candidate) <= MAX_TOKENS:
            current = candidate
        else:
            if current.strip():
                merged.append(current.strip())
            current = part
    if current.strip():
        merged.append(current.strip())

    final = []
    for chunk in merged:
        if count_tokens(chunk) <= MAX_TOKENS:
            final.append(chunk)
        else:
            para_chunk = ""
            for para in chunk.split('\n\n'):
                candidate = (para_chunk + "\n\n" + para).strip() if para_chunk else para
                if count_tokens(candidate) <= MAX_TOKENS:
                    para_chunk = candidate
                else:
                    if para_chunk.strip():
                        final.append(para_chunk.strip())
                    para_chunk = para
            if para_chunk.strip():
                final.append(para_chunk.strip())

    return final if final else [section_text]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build final chunk objects
# ══════════════════════════════════════════════════════════════════════════════

def build_text(act_name, chapter, section_number, section_title, body):
    return (
        f"[Act: {act_name}]\n"
        f"[{chapter}]\n"
        f"[Section {section_number}: {section_title}]\n\n"
        f"{body}"
    )


def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', '_', text.strip())


def make_chunk(chunk_id, text, act_name, sec, chunk_index, source_file):
    return {
        "chunk_id": chunk_id,
        "text":     text,
        "metadata": {
            "act_name":       act_name,
            "chapter":        sec["chapter"],
            "section_number": sec["section_number"],
            "section_title":  sec["section_title"],
            "language":       "en",
            "chunk_index":    chunk_index,
            "token_count":    count_tokens(text),
            "source_file":    source_file,
        }
    }


def chunk_act(act_name, sections, source_file):
    chunks   = []
    act_slug = slugify(act_name)

    for sec in sections:
        body = sec["text"]
        if count_tokens(body) <= MAX_TOKENS:
            text = build_text(act_name, sec["chapter"], sec["section_number"], sec["section_title"], body)
            cid  = f"{act_slug}__s{sec['section_number']}__0"
            chunks.append(make_chunk(cid, text, act_name, sec, 0, source_file))
        else:
            for idx, sub in enumerate(split_large_section(body)):
                text = build_text(act_name, sec["chapter"], sec["section_number"], sec["section_title"], sub)
                cid  = f"{act_slug}__s{sec['section_number']}__{idx}"
                chunks.append(make_chunk(cid, text, act_name, sec, idx, source_file))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — File processing helpers
# ══════════════════════════════════════════════════════════════════════════════

def process_file(filepath):
    print(f"\n📂 Processing: {filepath}")
    raw_text = Path(filepath).read_text(encoding='utf-8', errors='replace')
    filename = os.path.basename(filepath)

    acts       = split_into_acts(raw_text)
    all_chunks = []

    print(f"   Found {len(acts)} act(s)")
    for act_name, act_text in acts:
        cleaned  = clean_act_text(act_text)
        sections = parse_chapters_and_sections(cleaned)
        chunks   = chunk_act(act_name, sections, filename)
        print(f"   📜 {act_name}: {len(sections)} sections → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


def process_directory(input_dir, output_file):
    txt_files = list(Path(input_dir).glob("*.txt"))
    if not txt_files:
        print(f"❌ No .txt files found in {input_dir}")
        return []

    print(f"Found {len(txt_files)} .txt file(s) to process\n")
    all_chunks = []
    for f in sorted(txt_files):
        all_chunks.extend(process_file(f))

    write_jsonl(all_chunks, output_file)
    return all_chunks


def write_jsonl(chunks, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"\n{'═'*55}")
    print(f"  ✅ Done!  {len(chunks):,} chunks → {output_file}")
    print(f"{'═'*55}")

    from collections import Counter
    act_counts = Counter(c["metadata"]["act_name"] for c in chunks)
    print("\n  Chunks per Act:")
    for act, count in act_counts.most_common():
        print(f"    {count:>5}  {act}")

    if chunks:
        print("\n─── Sample Chunk (index 5) ───")
        print(json.dumps(chunks[min(5, len(chunks)-1)], indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal RAG Pipeline — Sakhi Edition",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python legal_rag_pipeline.py --single_file "data/Personal_&_Family_Laws.txt"
  python legal_rag_pipeline.py --input_dir data --output_file legal_chunks.jsonl
        """
    )
    from src.config import TXT_DIR, DEFAULT_INPUT_JSONL
    parser.add_argument("--input_dir",   type=str, default=str(TXT_DIR),
                        help="Directory containing your .txt files")
    parser.add_argument("--single_file", type=str, default=None,
                        help="Process a single .txt file")
    parser.add_argument("--output_file", type=str, default=str(DEFAULT_INPUT_JSONL),
                        help="Output JSONL path (default: legal_chunks.jsonl)")
    args = parser.parse_args()

    if args.single_file:
        chunks = process_file(args.single_file)
        write_jsonl(chunks, args.output_file)
    elif args.input_dir:
        process_directory(args.input_dir, args.output_file)
    else:
        print("❌  Please provide --single_file or --input_dir\n")
        parser.print_help()
