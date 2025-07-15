import streamlit as st
import io
import os
import tempfile
from pypdf import PdfReader # Still useful for raw text extraction as a fallback or for simple checks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # OpenAI models for embeddings and chat
from langchain_community.vectorstores import FAISS # Vector store for similarity search
#from langchain.chains import create_retrieval_qa_chain # For the RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os # To access environment variables

# --- Configuration and Constants ---
APP_TITLE = "Medical Report AI Assistant (Informational Only)"
PERMANENT_DISCLAIMER = """
**Important Disclaimer:** This AI assistant provides general information, summaries, and explanations of medical terminology from your uploaded reports. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment. Do not disregard professional medical advice or delay in seeking it because of something you have read here.
"""
AI_DISCLAIMER_PREFIX = "Please remember, this information is for discussion with your doctor and is not a diagnosis."

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide"
)

st.title(APP_TITLE)

# Display the permanent disclaimer prominently at the top
st.warning(PERMANENT_DISCLAIMER)

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialize a place to store all processed document chunks and processed file IDs
if "all_document_chunks" not in st.session_state:
    st.session_state.all_document_chunks = []
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set() # Use a set to store unique file IDs
# Initialize the FAISS vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize a key for the file uploader to allow programmatic clearing
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# --- OpenAI API Key Check ---
# Check if OPENAI_API_KEY environment variable is set
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY environment variable not found. Please set it to use the AI features. Refer to the instructions for setting it up.")
    st.stop() # Stop the app if key is not set

# --- File Uploader & Processing ---
st.header("Upload Your Medical Reports (PDFs)")

# Use the key to allow clearing the uploader widget
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more medical reports in PDF format for analysis.",
    key=f"pdf_uploader_{st.session_state.file_uploader_key}" # Unique key for resetting
)

# Button to clear all processed documents
if st.button("Clear All Processed Documents"):
    st.session_state.all_document_chunks = []
    st.session_state.processed_file_ids = set()
    st.session_state.vector_store = None # Clear the vector store as well
    st.session_state.messages.append({"role": "assistant", "content": "All processed documents have been cleared. You can now upload new files."})
    st.session_state.file_uploader_key += 1 # Change key to reset the uploader widget
    st.rerun() # Rerun to apply changes and clear the uploader display

if uploaded_files:
    newly_processed_count = 0
    newly_added_chunks = [] # Collect chunks from newly processed files

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        if uploaded_file.file_id not in st.session_state.processed_file_ids:
            st.info(f"Processing '{uploaded_file.name}'...")
            
            with st.spinner(f"Extracting text from '{uploaded_file.name}' and preparing for AI..."):
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_path = tmp_file.name

                    loader = PyPDFLoader(temp_file_path)
                    
                    # We are loading the document first, then splitting it
                    # This ensures metadata like 'source' is correctly attached by PyPDFLoader
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    # Split the loaded documents into chunks
                    chunks = text_splitter.split_documents(documents)
                    
                    if not chunks:
                        raise ValueError("Could not extract any meaningful text or documents from the PDF using Langchain's PyPDFLoader. It might be a scanned image without OCR, corrupted, or empty.")

                    st.session_state.all_document_chunks.extend(chunks) # Accumulate chunks
                    st.session_state.processed_file_ids.add(uploaded_file.file_id) # Mark as processed
                    newly_processed_count += 1
                    newly_added_chunks.extend(chunks)

                except Exception as e:
                    st.error(f"Error processing '{uploaded_file.name}': The document might be unreadable, improperly formatted, or an issue occurred during text extraction. Skipping this file. Details: {e}")

                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
    
    # After processing all new files, update the vector store if new chunks were added
    if newly_processed_count > 0:
        with st.spinner("Building vector store for AI retrieval..."):
            # If no vector store exists or if we're adding to an existing one
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") # Using a common embedding model
            if st.session_state.vector_store is None:
                # Create a new FAISS vector store from the first set of chunks
                st.session_state.vector_store = FAISS.from_documents(st.session_state.all_document_chunks, embeddings)
            else:
                # Add new chunks to the existing FAISS vector store
                # Note: This .add_documents() is efficient for FAISS.
                st.session_state.vector_store.add_documents(newly_added_chunks)
            
        st.success(f"Successfully processed {newly_processed_count} new file(s)! Total text chunks accumulated: {len(st.session_state.all_document_chunks)}. AI is ready to answer questions based on all uploaded reports.")
        st.session_state.messages.append({"role": "assistant", "content": f"I have processed {newly_processed_count} new report(s). There are now {len(st.session_state.all_document_chunks)} text chunks ready for your questions across all uploaded documents."})
    elif uploaded_files and newly_processed_count == 0 and len(st.session_state.all_document_chunks) > 0:
         st.info("All selected files were already processed, or no new readable content was found among the new selections. You can still ask questions based on the previously uploaded documents.")


# Display current processed status
if st.session_state.all_document_chunks:
    st.info(f"Currently analyzing {len(st.session_state.all_document_chunks)} text chunks from your reports.")
else:
    st.info("No documents uploaded or processed yet. Please upload a PDF to begin.")


# --- Chat Interface ---
st.header("Chat with the AI Assistant")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your reports..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_store: # Check if vector store is ready
            response = "Please upload one or more medical reports first so I can analyze their content."
        else:
            with st.spinner("Thinking..."):
                try:
                    # --- Langchain RAG Chain ---
                    # 1. Initialize LLM
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2) # You can change to "gpt-4o" or other models

                    # 2. Define the prompt for the LLM
                    # This prompt guides the AI to use the context and adhere to the "informational only" rule
                    # We pass the context here so the LLM uses it.
                    template = """
                    You are a helpful medical assistant. Use ONLY the following context to answer the user's question.
                    If you don't know the answer from the context, state that you don't have enough information.
                    Your goal is to summarize findings, explain medical terminology, and provide general informational context about potential issues, precautions, and widely applicable remedies based STRICTLY on the provided context.
                    Do NOT provide any specific medical diagnosis or personalized treatment advice. Always remind the user to consult a medical professional.

                    Context: {context}

                    Question: {question}

                    Answer:
                    """
                    prompt_template = ChatPromptTemplate.from_template(template)

                    # 3. Create the RAG chain
                    # Retriever gets relevant chunks from the vector store
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

                    # Define the RAG chain
                    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )

                    # 4. Invoke the chain with the user's prompt
                    raw_ai_response = rag_chain.invoke(prompt)

                    # Always prepend the AI-specific disclaimer
                    response = f"{AI_DISCLAIMER_PREFIX}\n\n{raw_ai_response}"
                    
                except Exception as e:
                    response = f"An error occurred while processing your request with the AI: {e}. Please try again later or upload a different document. This issue is observable via Langsmith."
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})