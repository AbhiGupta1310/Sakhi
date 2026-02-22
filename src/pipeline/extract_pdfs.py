#!/usr/bin/env python3
"""
extract_pdfs.py — Extract text from all PDFs in sakhi_data/ and save
one .txt file per category folder into data/.

Usage:
    python extract_pdfs.py
"""

import os
import fitz  # PyMuPDF
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────
from src.config import PDF_DIR, TXT_DIR
SOURCE_DIR = str(PDF_DIR)
OUTPUT_DIR = str(TXT_DIR)

SEPARATOR = "\n" + "=" * 80 + "\n"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a single PDF file using PyMuPDF."""
    text_parts: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
        doc.close()
    except Exception as e:
        print(f"  ⚠  Error reading {os.path.basename(pdf_path)}: {e}")
    return "\n".join(text_parts)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect category folders (skip hidden files like .DS_Store)
    categories = sorted(
        entry
        for entry in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, entry)) and not entry.startswith(".")
    )

    if not categories:
        print("No category folders found in sakhi_data/. Nothing to do.")
        return

    print(f"\n📂 Found {len(categories)} category folder(s) in sakhi_data/\n")

    summary: list[tuple[str, int, int, int]] = []  # (name, pdfs, pages, size)

    for category in categories:
        category_path = os.path.join(SOURCE_DIR, category)

        # Gather PDFs in this folder
        pdf_files = sorted(
            f
            for f in os.listdir(category_path)
            if f.lower().endswith(".pdf")
        )

        if not pdf_files:
            print(f"  ⏭  Skipping '{category}' — no PDF files found.")
            continue

        print(f"📖 Processing: {category} ({len(pdf_files)} PDF(s))")

        combined_text_parts: list[str] = []
        total_pages = 0

        for pdf_name in tqdm(pdf_files, desc=f"   {category}", unit="file"):
            pdf_path = os.path.join(category_path, pdf_name)

            # Count pages
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                doc.close()
            except Exception:
                num_pages = 0

            total_pages += num_pages

            # Build header + extracted text
            header = (
                f"\n{'=' * 80}\n"
                f"  📄 {pdf_name}\n"
                f"     Pages: {num_pages}\n"
                f"{'=' * 80}\n"
            )
            text = extract_text_from_pdf(pdf_path)
            combined_text_parts.append(header + text)

        # Write the combined text to output
        output_filename = f"{category.strip()}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        combined_text = "\n".join(combined_text_parts)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        file_size = os.path.getsize(output_path)
        summary.append((category.strip(), len(pdf_files), total_pages, file_size))
        print(f"   ✅ Saved → data/{output_filename}\n")

    # ── Summary table ────────────────────────────────────────────
    if summary:
        print("\n" + "=" * 70)
        print("  📊  EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"  {'Category':<40} {'PDFs':>5} {'Pages':>7} {'Size':>10}")
        print(f"  {'-' * 40} {'-' * 5} {'-' * 7} {'-' * 10}")

        total_pdfs = total_pgs = total_sz = 0
        for name, pdfs, pages, size in summary:
            size_str = f"{size / 1024:.1f} KB" if size < 1_048_576 else f"{size / 1_048_576:.1f} MB"
            print(f"  {name:<40} {pdfs:>5} {pages:>7} {size_str:>10}")
            total_pdfs += pdfs
            total_pgs += pages
            total_sz += size

        total_size_str = f"{total_sz / 1024:.1f} KB" if total_sz < 1_048_576 else f"{total_sz / 1_048_576:.1f} MB"
        print(f"  {'-' * 40} {'-' * 5} {'-' * 7} {'-' * 10}")
        print(f"  {'TOTAL':<40} {total_pdfs:>5} {total_pgs:>7} {total_size_str:>10}")
        print("=" * 70)
        print(f"\n✅ All done! {len(summary)} text file(s) saved to data/\n")


if __name__ == "__main__":
    main()
