# PDF Intelligence System with Retrieval Augmented Generation (RAG)

## Overview

The goal of this project is to create a user-centric and intelligent system that enhances information retrieval from PDF documents through natural language queries. The project focuses on streamlining the user experience by developing an intuitive interface, allowing users to interact with PDF content using language they are comfortable with. To achieve this, we leverage the Retrieval Augmented Generation (RAG) methodology introduced by Meta AI researchers.


https://github.com/ArmaanSeth/ChatPDF/assets/99117431/2500f636-c66d-46ad-bb68-1d55f04ce753


## Retrieval Augmented Generation (RAG)

### Introduction

RAG is a method designed to address knowledge-intensive tasks, particularly in information retrieval. It combines an information retrieval component with a text generator model to achieve adaptive and efficient knowledge processing. Unlike traditional methods that require retraining the entire model for knowledge updates, RAG allows for fine-tuning and modification of internal knowledge without extensive retraining.

### Current workflow

1. The app extracts text from uploaded PDFs with PyPDF2.
2. If the PDF has no selectable text, optional OCR renders its pages and uses Tesseract to recognize English text.
3. The app keeps page and line boundaries while chunking text, then ranks chunks with word and character TF-IDF similarity plus query-term overlap.
4. It selects relevant source sentences, handles document-name questions from upload metadata, recognizes broad summary requests, and can extract a matching numbered section when the PDF provides one.
5. It formats an extractive answer with PDF page citations; the active app does not call GPT-3.5 or use FAISS embeddings.
6. Streamlit provides the user interface.

OCR may introduce recognition errors, so verify extracted text before relying on answers from scanned PDFs.

## Evaluating answers

The evaluation script compares app answers with manually prepared reference answers. Its exact-match and token-F1 results are dataset-specific automatic text-overlap metrics; they are not a general or human-verified accuracy guarantee. Use a representative set of questions with correct answers checked against the source PDF.

The CSV needs these columns and at least one labeled row:

```csv
question,ground_truth
"What does OCR mean?","Optical character recognition"
```

Run the evaluator from this folder:

```powershell
python evaluate_accuracy.py --pdf "path\to\document.pdf" --dataset "path\to\evaluation_dataset.csv" --output evaluation_results.csv
```

For scanned PDFs, install the optional Python packages:

```powershell
pip install -r requirements-ocr.txt
```

Also install Tesseract OCR for Windows and ensure `tesseract.exe` is on `PATH`, then restart the terminal and Streamlit. The app reports an actionable error if OCR is needed but unavailable.

The app also checks the standard Windows install path (`C:\Program Files\Tesseract-OCR\tesseract.exe`). If Tesseract is installed elsewhere, set `TESSERACT_CMD` to the full path of `tesseract.exe` before starting the app or evaluator.

### Benefits

- **Adaptability**: RAG adapts to situations where facts may evolve over time, making it suitable for dynamic knowledge domains.
- **Efficiency**: By combining retrieval and generation, RAG provides access to the latest information without the need for extensive model retraining.
- **Reliability**: The methodology ensures reliable outputs by leveraging both retrieval-based and generative approaches.

## Project Features

1. **User-friendly Interface**: An intuitive interface designed to accommodate natural language queries, simplifying the interaction with PDF documents.

2. **Seamless Navigation**: The system streamlines information retrieval, reducing complexity and enhancing the overall user experience.

## Getting Started

To use the PDF Intelligence System:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/ArmaanSeth/ChatPDF.git
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application.
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8000` to access the user interface.

### Voice input

Browser speech recognition needs microphone access from a secure origin. Use `http://localhost:8501` on the computer running Streamlit, or configure HTTPS before opening the app from another device on the network. A plain `http://<LAN-IP>:8501` address is not secure and browsers block microphone access there. Browser-based speech recognition may also require an internet connection.

## Contributing

We welcome contributions to enhance the PDF Intelligence System. If you're interested in contributing, please follow our [Contribution Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [Apache License](LICENSE).

## Acknowledgments

We would like to express our gratitude to the Hugging Face community for the all-MiniLM-L6-v2 Embeddings model, and OpenAI for providing the GPT-3.5 Turbo model through their API.

---

Feel free to explore and enhance the capabilities of the PDF Intelligence System. Happy querying!
