import os
import fitz  # PyMuPDF for PDF extraction
import pytesseract
from PIL import Image
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from app.services.ocr_service import OCRService
from app.services.ml_service import MLService

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
# ✅ Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/run_demo.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

DATA_DIR = "data"
SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg")
SUPPORTED_PDF_EXTS = (".pdf",)

# -------------------------------------------------------------------
# 🧠 Dynamic Training + Prediction
# -------------------------------------------------------------------
def train_dynamic_model(file_texts):
    """Train a simple text classifier dynamically."""
    if not file_texts:
        logging.warning("No text data found for training.")
        return None, None

    # Derive labels dynamically from file names or folders
    texts = list(file_texts.values())
    labels = [
        os.path.basename(os.path.dirname(path)) or os.path.splitext(os.path.basename(path))[0].split("_")[0]
        for path in file_texts.keys()
    ]

    logging.info(f"📊 Total samples: {len(texts)}, Classes: {set(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    logging.info(f"🏁 Training complete. Accuracy: {acc:.2f}")
    logging.info("\n" + report)

    # Save report
    with open("logs/training_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n\n")
        f.write(report)

    return model, acc

def train_demo_model(texts, labels):
    """Train a simple text classification model."""
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts, labels)
    return model

def main():
    logging.info("🚀 Starting OCR and ML demo...")

    # Step 1: Extract all text from data directory
    file_texts = OCRService.extract_all_from_directory(DATA_DIR)
    logging.info(f"📁 Extracted {len(file_texts)} files with text.")

    # Print/log extracted text per file
    for file_path, text in file_texts.items():
        header = f"--- Extracted from: {file_path} ---"
        logging.info(header)
        # Also print to stdout for immediate visibility in the console
        if text and text.strip():
            # Print full text; if texts are very long you can truncate here
            logging.info("\n" + text)
        else:
            logging.info("[NO TEXT EXTRACTED]")

    # Step 2: Train model dynamically
    model, acc = train_dynamic_model(file_texts)
    if not model:
        logging.error("❌ Model training failed.")
        return

    # Step 3: Run a sample prediction
    sample_text = next(iter(file_texts.values()))
    prediction = model.predict([sample_text])[0]
    logging.info(f"🔍 Sample Prediction: {prediction}")

if __name__ == "__main__":
    main()