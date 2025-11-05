# app/services/ocr_service.py

import logging
import os
import fitz # PyMuPDF for PDF extraction
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Set Tesseract path for macOS/Homebrew
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


class OCRService:
    @staticmethod
    def extract_text_from_image(image_path: str) -> str:
        """Extract text from a single image."""
        try:
            print(f"🖼️ Extracting text from image: {os.path.basename(image_path)}")
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"❌ OCR failed for image {image_path}: {e}")
            return ""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF using PyMuPDF."""
        text = ""
        try:
            with fitz.open(pdf_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            logging.error(f"Error reading PDF {pdf_path}: {e}")
            return ""   

    @staticmethod
    def extract_all_from_directory(directory_path: str):
        """Extract all text from images and PDFs in the data directory."""
        file_texts = {}
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                    text = OCRService.extract_text_from_image(file_path)
                elif file.lower().endswith(".pdf"):
                    text = OCRService.extract_text_from_pdf(file_path)
                else:
                    continue

                if text:
                    file_texts[file_path] = text
                    logging.info(f"✅ Extracted text from: {file}")
                else:
                    logging.warning(f"⚠️ No text extracted from: {file}")
        return file_texts
