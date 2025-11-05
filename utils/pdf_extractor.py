import pymupdf
from pathlib import Path
from typing import List, Union


def extract_text_from_pdf(pdf_path: str, by_page: bool = False) -> Union[str, List[str]]:
    """
    Extract text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        by_page: If True, returns a list of texts per page. If False, returns a single combined string.

    Returns:
        str | List[str]: Extracted text from all pages, combined or split per page.
    """
    doc = pymupdf.open(pdf_path)
    all_pages = []
    full_text = ""

    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        if by_page:
            # Keep clean page text for page-wise extraction
            all_pages.append(page_text)
        else:
            # Add page delimiter for readability (combined mode)
            full_text += (
                f"\n\n{'=' * 50}\n"
                f"PAGE {page_num}\n"
                f"{'=' * 50}\n\n"
                f"{page_text}"
            )

    doc.close()

    return all_pages if by_page else full_text
