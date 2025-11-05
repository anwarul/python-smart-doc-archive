# python-smart-doc-archive

A Python-based intelligent document archiving system using OCR, text analysis, and machine learning for automatic document classification. Converts scanned documents and PDFs into searchable, analyzable text for smarter data management.

## Features

- 📄 **PDF Text Extraction** using PyMuPDF
- 🖼️ **OCR for Images** using Tesseract and Pillow
- 🤖 **ML-based Classification** using scikit-learn (TF-IDF + Naive Bayes)
- 📊 **Training Reports** with accuracy and classification metrics
- 🔍 **Searchable Document Archive** from scanned/digital files

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Tesseract OCR engine

### Install Tesseract (macOS)

```bash
brew install tesseract
```

For other platforms, visit: https://github.com/tesseract-ocr/tesseract

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/anwarulislam/python-smart-doc-archive.git
cd python-smart-doc-archive
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Verify Tesseract Installation

```bash
tesseract --version
```

If Tesseract is installed in a custom location, update the path in `app/run_demo.py`:

```python
pytesseract.pytesseract.tesseract_cmd = "/your/custom/path/to/tesseract"
```

## Project Structure

```
python-smart-doc-archive/
├── app/
│   ├── services/
│   │   ├── ocr_service.py       # OCR extraction logic
│   │   └── ml_service.py        # ML model service
│   └── run_demo.py              # Main demo script
├── data/                        # Place your documents here (PDFs, images)
├── logs/                        # Training reports and logs
├── requirements.txt             # Python dependencies
└── README.md
```

## Usage

### 1. Add Documents

Place your documents (PDFs, PNG, JPG, JPEG) in the `data/` directory:

```bash
mkdir -p data
cp /path/to/your/documents/* data/
```

### 2. Run the Demo

```bash
python3 app/run_demo.py
```

### 3. Check Output

- **Console**: Real-time extraction and predictions
- **Log File**: `logs/run_demo.log`
- **Training Report**: `logs/training_report.txt`

## Example Output

```
🚀 Starting OCR and ML demo...
📁 Extracted 3 files with text.

--- Extracted from: data/invoice_2024.pdf ---
Invoice #12345
Date: 2024-11-05
Total: $1,234.56

📊 Total samples: 3, Classes: {'invoice', 'resume', 'report'}
🏁 Training complete. Accuracy: 0.95
🔍 Sample Prediction: invoice
```

## Supported File Types

- **PDFs**: `.pdf`
- **Images**: `.png`, `.jpg`, `.jpeg`

## Troubleshooting

### Error: `name 'fitz' is not defined`

Install PyMuPDF:
```bash
python3 -m pip install pymupdf
```

### Error: `TesseractNotFoundError`

Install Tesseract or update the path in `app/run_demo.py`:
```python
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
```

### Low Accuracy

- Add more diverse training documents to `data/`
- Organize files into category subdirectories (e.g., `data/invoices/`, `data/resumes/`)
- Ensure scanned images have good quality and resolution

## Development

### Run Tests (if applicable)

```bash
python3 -m pytest tests/
```

### Generate Requirements

```bash
python3 -m pip install pipreqs
pipreqs --force .
```

## Dependencies

- `pymupdf` - PDF text extraction
- `pytesseract` - OCR wrapper for Tesseract
- `Pillow` - Image processing
- `scikit-learn` - Machine learning models
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `pdf2image` - PDF to image conversion (optional)

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

Anwarul Islam

## Acknowledgments

- Tesseract OCR by Google
- PyMuPDF (fitz) for PDF processing
- scikit-learn for ML capabilities
