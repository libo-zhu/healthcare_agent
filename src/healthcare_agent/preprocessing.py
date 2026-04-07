from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image
from pypdf import PdfReader
import pytesseract

from healthcare_agent.schemas import PreprocessedInput


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".csv", ".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


async def build_preprocessed_input(
    medical_data: str | None = None,
    files: list[UploadFile] | None = None,
) -> PreprocessedInput:
    normalized_texts: list[str] = []
    source_parts: list[str] = []

    if medical_data and medical_data.strip():
        normalized_texts.append(medical_data.strip())
        source_parts.append("inline_text")

    for upload in files or []:
        extracted_text = await extract_text_from_upload(upload)
        normalized_texts.append(
            f"文件名: {upload.filename or 'unknown'}\n提取内容:\n{extracted_text}"
        )
        source_parts.append(upload.filename or "unknown")

    if not normalized_texts:
        raise HTTPException(
            status_code=400,
            detail="Please provide medical_data or upload at least one supported file.",
        )

    return PreprocessedInput(
        medical_data="\n\n".join(normalized_texts),
        source_summary=", ".join(source_parts),
    )


async def extract_text_from_upload(upload: UploadFile) -> str:
    filename = upload.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {suffix or 'unknown'}. "
                "Supported types: txt, csv, pdf, png, jpg, jpeg, bmp, tif, tiff, webp."
            ),
        )

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {filename}")

    if suffix == ".txt":
        return decode_text_bytes(raw_bytes, filename)
    if suffix == ".csv":
        return extract_text_from_csv(raw_bytes, filename)
    if suffix == ".pdf":
        return extract_text_from_pdf(raw_bytes, filename)
    return extract_text_from_image(raw_bytes, filename)


def decode_text_bytes(raw_bytes: bytes, filename: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            text = raw_bytes.decode(encoding)
            return validate_extracted_text(text, filename)
        except UnicodeDecodeError:
            continue
    text = raw_bytes.decode("utf-8", errors="ignore")
    return validate_extracted_text(text, filename)


def extract_text_from_csv(raw_bytes: bytes, filename: str) -> str:
    text = decode_text_bytes(raw_bytes, filename)
    rows: list[str] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        cleaned = [cell.strip() for cell in row if cell and cell.strip()]
        if cleaned:
            rows.append(" | ".join(cleaned))
    return validate_extracted_text("\n".join(rows), filename)


def extract_text_from_pdf(raw_bytes: bytes, filename: str) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF {filename}: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {index}]\n{text.strip()}")

    return validate_extracted_text("\n\n".join(pages), filename)


def extract_text_from_image(raw_bytes: bytes, filename: str) -> str:
    if shutil.which("tesseract") is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Image OCR requires the system dependency 'tesseract'. "
                "Please install it locally and retry."
            ),
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        text = pytesseract.image_to_string(image, lang="eng+chi_sim")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to OCR image {filename}: {exc}") from exc

    return validate_extracted_text(text, filename)


def validate_extracted_text(text: str, filename: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"No readable text could be extracted from {filename}.",
        )
    return normalized

