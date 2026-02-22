"""
Chunk Preview Tool
==================
Pretty prints chunks from the legal_chunks.jsonl file.

Usage:
    python preview_chunks.py                          # preview first 10 chunks
    python preview_chunks.py --file legal_chunks.jsonl  # specify file
    python preview_chunks.py --n 20                   # show first 20 chunks
    python preview_chunks.py --act "Hindu Marriage Act"  # filter by act name
    python preview_chunks.py --section 27             # filter by section number
    python preview_chunks.py --all                    # print ALL chunks
"""

import json
import argparse
from pathlib import Path

# ── Terminal colors ──
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
GREY   = "\033[90m"
RED    = "\033[91m"
BLUE   = "\033[94m"


def print_chunk(chunk, index):
    meta = chunk["metadata"]
    text = chunk["text"]

    separator = f"{GREY}{'─' * 72}{RESET}"

    print(separator)
    print(f"{BOLD}{CYAN}  Chunk #{index + 1}  {RESET}  {GREY}ID: {chunk['chunk_id']}{RESET}")
    print(separator)

    # Metadata block
    print(f"  {YELLOW}📁 Act      :{RESET} {meta.get('act_name', 'N/A')}")
    print(f"  {YELLOW}📂 Chapter  :{RESET} {meta.get('chapter', 'N/A')}")
    print(f"  {YELLOW}§  Section  :{RESET} {meta.get('section_number', 'N/A')} — {meta.get('section_title', 'N/A')}")
    print(f"  {YELLOW}📄 Source   :{RESET} {meta.get('source_file', 'N/A')}")
    print(f"  {YELLOW}🔢 Tokens   :{RESET} {meta.get('token_count', 'N/A')}")
    print(f"  {YELLOW}🔢 Sub-chunk:{RESET} {meta.get('chunk_index', 0)}")

    print()
    print(f"{GREEN}  TEXT:{RESET}")
    print()

    # Print text with indentation
    for line in text.splitlines():
        print(f"    {line}")

    print()


def load_chunks(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"{RED}❌ File not found: {filepath}{RESET}")
        print(f"   Make sure you've run the pipeline first to generate the .jsonl file.")
        exit(1)

    chunks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    from src.config import DEFAULT_INPUT_JSONL
    parser = argparse.ArgumentParser(description="Preview legal RAG chunks")
    parser.add_argument("--file",    type=str, default=str(DEFAULT_INPUT_JSONL), help="Path to JSONL file")
    parser.add_argument("--n",       type=int, default=10,                   help="Number of chunks to show (default: 10)")
    parser.add_argument("--act",     type=str, default=None,                 help="Filter by act name (partial match)")
    parser.add_argument("--section", type=str, default=None,                 help="Filter by section number")
    parser.add_argument("--all",     action="store_true",                    help="Print all chunks")
    args = parser.parse_args()

    chunks = load_chunks(args.file)
    total = len(chunks)

    # Apply filters
    filtered = chunks
    if args.act:
        filtered = [c for c in filtered if args.act.lower() in c["metadata"].get("act_name", "").lower()]
    if args.section:
        filtered = [c for c in filtered if c["metadata"].get("section_number", "") == args.section]

    # Apply limit
    limit = len(filtered) if args.all else args.n
    to_show = filtered[:limit]

    print(f"\n{BOLD}{BLUE}=== CHUNK PREVIEW ==={RESET}")
    print(f"{GREY}File: {args.file} | Total chunks: {total} | Showing: {len(to_show)}{RESET}\n")

    if not to_show:
        print(f"{RED}No chunks matched your filters.{RESET}")
        return

    for i, chunk in enumerate(to_show):
        print_chunk(chunk, i)

    print(f"{GREY}{'─' * 72}{RESET}")
    print(f"\n{BOLD}Showed {len(to_show)} of {len(filtered)} matching chunks (total in file: {total}){RESET}\n")


if __name__ == "__main__":
    main()