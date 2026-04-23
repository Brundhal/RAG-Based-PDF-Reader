# importing dependencies
import re

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from htmlTemplates import css, bot_template, user_template


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


# extracting text from pdf
def get_pdf_text(docs):
    text = ""
    pdf_data = []
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        pages = []
        for page in pdf_reader.pages:
            pages.append(page.extract_text() or "")
        pdf_text = clean_text(" ".join(pages))
        text += pdf_text + "\n"
        pdf_data.append({
            "name": pdf.name,
            "text": pdf_text,
            "page_count": len(pdf_reader.pages),
            "has_text": bool(pdf_text),
        })
    return clean_text(text), pdf_data


# converting text to chunks
def get_chunks(raw_text, chunk_size=1000, chunk_overlap=180):
    raw_text = clean_text(raw_text)
    if not raw_text:
        return []

    chunks = []
    start = 0
    while start < len(raw_text):
        end = start + chunk_size
        chunks.append(raw_text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def build_search_index(pdf_data):
    chunks = []
    for pdf in pdf_data:
        for chunk in get_chunks(pdf.get("text", "")):
            chunks.append({"document": pdf.get("name", "PDF"), "text": chunk})

    if not chunks:
        return {"chunks": [], "vectorizer": None, "matrix": None}

    if TfidfVectorizer:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
        return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}

    return {"chunks": chunks, "vectorizer": None, "matrix": None}


def retrieve_chunks(question, search_index, limit=6):
    chunks = search_index.get("chunks", [])
    if not chunks:
        return []

    vectorizer = search_index.get("vectorizer")
    matrix = search_index.get("matrix")
    if vectorizer is not None and matrix is not None and cosine_similarity:
        query_vector = vectorizer.transform([question])
        scores = cosine_similarity(query_vector, matrix).flatten()
        ranked = scores.argsort()[::-1][:limit]
        return [chunks[index] for index in ranked if scores[index] > 0]

    # Fallback: keyword matching
    question_words = set(re.findall(r"\w+", question.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk["text"].lower()))
        score = len(question_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:limit] if score > 0]


def summarize_text(raw_text, max_sentences=6):
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(raw_text))
    sentences = [sentence for sentence in sentences if len(sentence) > 35]
    if not sentences:
        return "The uploaded PDF was processed, but it does not contain enough readable text to summarize."
    return "\n".join([f"- {sentence}" for sentence in sentences[:max_sentences]])


def answer_from_chunks(question, search_index, pdf_data=None):
    q_lower = question.lower()
    
    # Metadata questions - generic patterns
    count_pattern = re.compile(r"\b(how\s+many|number\s+of|count|how\s+long|pages?)\b", re.IGNORECASE)
    name_pattern = re.compile(r"\b(name\s+of|file\s+name|document\s+name|called|what\s+is\s+.*name|title)\b", re.IGNORECASE)
    summary_pattern = re.compile(r"\b(summary|summarize|about|describe|explain|what\s+does|what's)\b", re.IGNORECASE)
    
    if pdf_data:
        if count_pattern.search(q_lower):
            total_pages = sum(pdf.get("page_count", 0) for pdf in pdf_data)
            return f"The document has {total_pages} page(s)."
        
        if name_pattern.search(q_lower):
            names = [pdf.get("name", "Unknown") for pdf in pdf_data]
            if len(names) == 1:
                return f"The document name is {names[0]}."
            return f"The uploaded documents are: {', '.join(names)}."
        
        if summary_pattern.search(q_lower):
            combined_pdf_text = " ".join(pdf.get("text", "") for pdf in pdf_data)
            return summarize_text(combined_pdf_text, max_sentences=5)
    
    # Get relevant chunks for any question
    matches = retrieve_chunks(question, search_index, limit=6)
    if not matches:
        return "I processed the PDF, but I could not find a relevant answer in the extracted text."

    combined = clean_text(" ".join(match["text"] for match in matches))

    # Extract sentences and score by relevance to question
    question_words = set(re.findall(r"\w+", q_lower))
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from"}
    key_words = question_words - stop_words
    
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    
    scored = []
    for sentence in sentences:
        sentence_words = set(re.findall(r"\w+", sentence.lower()))
        key_intersection = len(key_words & sentence_words)
        total_intersection = len(question_words & sentence_words)
        # Prefer sentences with key question words and longer meaningful content
        if key_intersection > 0:
            score = key_intersection * 2 + total_intersection
            scored.append((score, len(sentence), sentence))
    
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [sentence for _, _, sentence in scored[:5]]
    
    if not selected:
        # Fallback: return first part of combined text
        selected = [combined[:500]]

    source_docs = sorted({match["document"] for match in matches})
    return clean_text(" ".join(selected)) + f"\n\nSource: {', '.join(source_docs)}"


def get_document_summary(raw_text):
    return summarize_text(raw_text)


# generating response from user queries and displaying them accordingly
def handle_question(question):
    answer = answer_from_chunks(
        question,
        st.session_state.search_index,
        st.session_state.pdf_data,
    )
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", answer))


def render_chat_history():
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = []

    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    if "summary" not in st.session_state:
        st.session_state.summary = None

    if "search_index" not in st.session_state:
        st.session_state.search_index = {"chunks": [], "vectorizer": None, "matrix": None}

    if "pdf_data" not in st.session_state:
        st.session_state.pdf_data = []

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask question from your document:")
    if question:
        if st.session_state.document_processed:
            handle_question(question)
        else:
            st.warning("Please upload and process a PDF before asking questions.")

    st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
    render_chat_history()
    st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if docs:
            st.markdown(
                "<div class='selected-docs'><strong>Selected PDF files</strong>"
                + "".join([f"<div class='selected-doc'>{doc.name}</div>" for doc in docs])
                + "</div>",
                unsafe_allow_html=True,
            )

        if st.button("Process"):
            if not docs:
                st.warning("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Processing your document..."):
                    raw_text, pdf_data = get_pdf_text(docs)
                    if not raw_text:
                        st.error("The PDF was uploaded, but no readable text was found.")
                    else:
                        search_index = build_search_index(pdf_data)
                        st.session_state.raw_text = raw_text
                        st.session_state.pdf_data = pdf_data
                        st.session_state.search_index = search_index
                        st.session_state.processed_docs = [doc.name for doc in docs]
                        st.session_state.document_processed = True
                        st.session_state.summary = None
                        st.session_state.chat_history = []

        if st.session_state.document_processed:
            processed_names = ", ".join(st.session_state.processed_docs)
            st.markdown(
                f"<div class='processed-box'>Document processed successfully.<br><span>{processed_names}</span></div>",
                unsafe_allow_html=True,
            )

        if st.session_state.document_processed:
            if st.button("Summarize Document"):
                with st.spinner("Creating summary..."):
                    st.session_state.summary = get_document_summary(st.session_state.raw_text)

        if st.session_state.summary:
            st.markdown("<div class='summary-display'>", unsafe_allow_html=True)
            st.markdown("<h3>Document Summary</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='summary-content'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
