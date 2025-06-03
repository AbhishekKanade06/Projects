import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Extract text from PDFs
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

# Embed and store using FAISS (with SentenceTransformer for embeddings)
def create_vector_store(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a local embedding model
    embeddings = [model.encode(chunk) for chunk in chunks]
    vector_store = FAISS.from_embeddings(embeddings, FAISS.index_factory(len(embeddings[0]), "Flat"))
    vector_store.save_local("faiss_index")
    return vector_store

# Load vector store
def load_vector_store():
    return FAISS.load_local("faiss_index")

# Create Q&A chain with a simple prompt
def create_qa_chain():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant answering based on provided context only.

If the answer is not found in the context, say:
"The answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""
    )
    chain = load_qa_chain(chain_type="stuff", prompt=prompt_template)
    return chain

# Answer the question
def answer_question(question, vector_store):
    docs = vector_store.similarity_search(question)
    chain = create_qa_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response.get("output_text", "No answer returned.")

# Streamlit UI
def main():
    st.set_page_config(page_title="Local PDF Chatbot", page_icon="📄")
    st.title("🤖 Local PDF Chatbot (No API)")
    
    # Add some instructions for better UX
    st.markdown("""
    Upload PDFs and ask questions — powered by **SentenceTransformers** locally!
    
    **How it works:**
    1. Upload one or more PDFs.
    2. Once processed, type your questions and get answers from the content of your uploaded PDFs.
    """)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.subheader("📁 Upload PDFs")
        pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        # Button to trigger PDF processing
        if st.button("🔄 Process PDFs"):
            if not pdfs:
                st.warning("Please upload at least one PDF.")
                return
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdfs(pdfs)
                if raw_text.strip():
                    chunks = split_text_into_chunks(raw_text)
                    create_vector_store(chunks)
                    st.success("PDFs processed and indexed successfully!")
                else:
                    st.error("No readable text found in the PDFs.")
    
    # Main content section
    if os.path.exists("faiss_index"):
        st.markdown("### Ask questions based on your PDFs:")

        # Create a dynamic chat-like UI
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Display the history of the conversation
        for i, chat in enumerate(st.session_state.history):
            if chat['role'] == 'user':
                st.write(f"**You:** {chat['message']}")
            else:
                st.write(f"**Bot:** {chat['message']}")

        # Text input for user questions
        question = st.text_input("💬 Type your question here")

        if question:
            # Store the user's question in the history
            st.session_state.history.append({"role": "user", "message": question})

            # Show processing spinner
            with st.spinner("Thinking..."):
                vector_store = load_vector_store()
                answer = answer_question(question, vector_store)

                # Store the bot's answer in the history
                st.session_state.history.append({"role": "bot", "message": answer})

                # Display the answer in the UI
                st.write(f"**Bot:** {answer}")

    st.markdown("---")
    st.caption("Made with ❤️ using **SentenceTransformers** & **FAISS** locally")

if __name__ == "__main__":
    main()
