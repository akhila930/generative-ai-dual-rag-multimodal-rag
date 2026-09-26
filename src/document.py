import json
from pathlib import Path
import fitz

from .config import HANDBOOK_PATH, PAGES_JSON, PAGE_IMAGE_DIR

def extract_handbook(pdf_path=HANDBOOK_PATH):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Handbook not found: {pdf_path}")

    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    doc = fitz.open(pdf_path)
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        image_path = PAGE_IMAGE_DIR / f"page_{page_no:02d}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
        pix.save(image_path)

        records.append({
            "page": page_no,
            "text": text,
            "image_path": str(image_path.relative_to(PAGE_IMAGE_DIR.parent.parent)),
        })

    PAGES_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records

if __name__ == "__main__":
    records = extract_handbook()
    print(f"Processed {len(records)} pages.")
    print(f"Saved: {PAGES_JSON}")
