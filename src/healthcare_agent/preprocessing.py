from __future__ import annotations

import csv
import io
import shutil
import subprocess
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
    notes: list[str] = []

    if medical_data and medical_data.strip():
        normalized_texts.append(medical_data.strip())
        source_parts.append("inline_text")

    for upload in files or []:
        extracted_text, file_notes = await extract_text_from_upload(upload)
        normalized_texts.append(
            f"文件名: {upload.filename or 'unknown'}\n提取内容:\n{extracted_text}"
        )
        source_parts.append(upload.filename or "unknown")
        notes.extend(file_notes)

    if not normalized_texts:
        raise HTTPException(
            status_code=400,
            detail="Please provide medical_data or upload at least one supported file.",
        )

    return PreprocessedInput(
        medical_data="\n\n".join(normalized_texts),
        source_summary=", ".join(source_parts),
        notes=deduplicate_notes(notes),
    )


async def extract_text_from_upload(upload: UploadFile) -> tuple[str, list[str]]:
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
        return decode_text_bytes(raw_bytes, filename), []
    if suffix == ".csv":
        return extract_text_from_csv(raw_bytes, filename), []
    if suffix == ".pdf":
        return extract_text_from_pdf(raw_bytes, filename), []
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


def extract_text_from_image(raw_bytes: bytes, filename: str) -> tuple[str, list[str]]:
    if shutil.which("tesseract") is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Image OCR requires the system dependency 'tesseract'. "
                "Please install it locally and retry."
            ),
        )

    installed_languages = get_tesseract_languages()
    if not installed_languages:
        raise HTTPException(
            status_code=500,
            detail="Tesseract is installed, but no OCR languages were detected.",
        )

    notes: list[str] = []
    ocr_language = select_ocr_language(installed_languages)
    if "chi_sim" not in installed_languages:
        notes.append(
            "OCR warning: tesseract language pack 'chi_sim' is not installed; Chinese extraction quality may be poor."
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        prepared_variants = build_ocr_image_variants(image)
        candidates: list[str] = []
        for variant in prepared_variants:
            text = pytesseract.image_to_string(
                variant,
                lang=ocr_language,
                config="--oem 3 --psm 6",
            )
            normalized = text.strip()
            if normalized:
                candidates.append(normalized)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to OCR image {filename}: {exc}") from exc

    best_text = pick_best_ocr_text(candidates)
    notes.append(f"OCR languages used: {ocr_language}")
    notes.append("OCR preprocessing: grayscale, autocontrast, sharpen, upscale, threshold")
    return validate_extracted_text(best_text, filename), notes


def validate_extracted_text(text: str, filename: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"No readable text could be extracted from {filename}.",
        )
    return normalized


def get_tesseract_languages() -> set[str]:
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()

    languages: set[str] = set()
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("List of available languages"):
            languages.add(candidate)
    return languages


def select_ocr_language(installed_languages: set[str]) -> str:
    if {"chi_sim", "eng"}.issubset(installed_languages):
        return "eng+chi_sim"
    if "chi_sim" in installed_languages:
        return "chi_sim"
    if "eng" in installed_languages:
        return "eng"
    return "+".join(sorted(installed_languages))


def build_ocr_image_variants(image: Image.Image) -> list[Image.Image]:
    rgb_image = image.convert("RGB")
    grayscale = ImageOps.grayscale(rgb_image)
    autocontrast = ImageOps.autocontrast(grayscale)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)
    enlarged = sharpened.resize(
        (max(1, sharpened.width * 2), max(1, sharpened.height * 2)),
        Image.Resampling.LANCZOS,
    )
    boosted = ImageEnhance.Contrast(enlarged).enhance(1.8)
    threshold = boosted.point(lambda pixel: 255 if pixel > 170 else 0)
    return [boosted, threshold]


def pick_best_ocr_text(candidates: list[str]) -> str:
    if not candidates:
        return ""
    return max(candidates, key=score_ocr_text)


def score_ocr_text(text: str) -> tuple[int, int, int]:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    digits = sum(1 for char in text if char.isdigit())
    length = len(text)
    return chinese_chars, digits, length


def deduplicate_notes(notes: list[str]) -> list[str]:
    unique_notes: list[str] = []
    seen: set[str] = set()
    for note in notes:
        if note not in seen:
            unique_notes.append(note)
            seen.add(note)
    return unique_notes
