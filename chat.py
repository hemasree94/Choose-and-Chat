import os
import groq
import faiss
import numpy as np
import fitz  # PyMuPDF
from typing import List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Load .env file
load_dotenv()


# --- Step 1: Setup ---
api_key = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=api_key)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 2: Extract text from PDF ---
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Step 3: Document Chunking ---
def chunk_document(document_text: str, chunk_size: int = 100) -> List[str]:
    words = document_text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# --- Step 4: Embed Chunks and Store in FAISS ---
def build_faiss_index(chunks: List[str]):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# --- Step 5: Retrieve Top Relevant Chunks ---
def retrieve_context(question: str, index, chunks, embeddings, top_k=3):
    question_embedding = embedding_model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    return [chunks[i] for i in I[0]]

# --- Step 6: Ask question using Groq + Retrieved Context ---
def ask_question_with_rag(question: str, context_chunks: List[str], chat_history: List[dict]) -> str:
    context = "\n\n".join(context_chunks)
    system_prompt = {
        "role": "system",
        "content": f"You are a helpful assistant. Use the context below to answer the user's questions.\n\nContext:\n{context}"
    }
    
    # Include system prompt and full chat history
    messages = [system_prompt] + chat_history + [{"role": "user", "content": question}]
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return response.choices[0].message.content

# --- Step 7: Full RAG Pipeline with Chat Memory ---
def rag_pipeline(pdf_path: str, question):
    document_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_document(document_text)
    index, embeddings, chunks = build_faiss_index(chunks)

    chat_history = []

    while True:
        
        if question.lower() == 'quit':
            break
        top_chunks = retrieve_context(question, index, chunks, embeddings)
        answer = ask_question_with_rag(question, top_chunks, chat_history)
        

        # Save interaction in chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return answer


