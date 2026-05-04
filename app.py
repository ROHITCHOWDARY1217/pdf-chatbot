import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import tempfile
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Page Config ----
st.set_page_config(
    page_title="Chat With PDF",
    page_icon="📄",
    layout="wide"
)

# ---- Title ----
st.title("📄 Chat With Your PDF")
st.markdown("Upload a PDF and ask anything about it!")

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vectorstore_path" not in st.session_state:
    st.session_state.vectorstore_path = None

# ---- Sidebar ----
with st.sidebar:
    st.header("📁 Upload Your PDFs")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,  # ✅ Added this
        help="You can select multiple PDF files"
    )

    if uploaded_files:  # ✅ Changed from uploaded_file to uploaded_files
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Show uploaded file names
        with st.expander("📄 Selected Files"):
            for file in uploaded_files:
                st.write(f"• {file.name}")
        
        if st.button("Process PDFs", type="primary"):
            with st.spinner("Reading your PDFs..."):

                # STEP 1: Clear everything
                st.session_state.chat_history = []
                st.session_state.chain = None

                # Clear old vectorstore
                if st.session_state.vectorstore_path is not None:
                    shutil.rmtree(
                        st.session_state.vectorstore_path,
                        ignore_errors=True
                    )
                    st.session_state.vectorstore_path = None

                all_documents = []  # ✅ Store all documents here
                total_pages = 0
                
                # ✅ Process EACH uploaded file
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    # Add source filename to metadata
                    for doc in documents:
                        doc.metadata['source_file'] = uploaded_file.name
                    
                    all_documents.extend(documents)  # ✅ Add to combined list
                    total_pages += len(documents)

                    # Clean temp PDF file
                    os.unlink(tmp_path)

                # ✅ Now process ALL documents together
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(all_documents)

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                # Create fresh temp directory
                temp_dir = tempfile.mkdtemp()

                # Create FRESH vectorstore with ALL PDFs
                vectorstore = Chroma.from_documents(
                    chunks,
                    embeddings,
                    persist_directory=temp_dir
                )

                # Store temp directory path
                st.session_state.vectorstore_path = temp_dir

                # Create LLM
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model="llama-3.1-8b-instant",
                    temperature=0.5
                )

                # Create FRESH memory
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                    max_token_limit=2000
                )

                # Create NEW chain with fresh memory
                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    memory=memory,
                    return_source_documents=True  # ✅ Added this
                )

                st.session_state.pdf_processed = True

            st.success(f"✅ All PDFs Processed!")
            st.info(f"📁 Files: {len(uploaded_files)}")
            st.info(f"📄 Total Pages: {total_pages}")
            st.info(f"🔢 Chunks: {len(chunks)}")

    # Clear button
    if st.session_state.pdf_processed:
        if st.button("🗑️ Clear Chat"):
            # Clear vectorstore
            if st.session_state.vectorstore_path is not None:
                shutil.rmtree(
                    st.session_state.vectorstore_path,
                    ignore_errors=True
                )
            # Reset everything
            st.session_state.chat_history = []
            st.session_state.chain = None
            st.session_state.pdf_processed = False
            st.session_state.vectorstore_path = None
            st.rerun()

# ---- Main Chat Area ----
if not st.session_state.pdf_processed:
    # Show instructions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 Step 1: Upload PDF from sidebar")
    with col2:
        st.info("⚙️ Step 2: Click Process PDF button")
    with col3:
        st.info("💬 Step 3: Start asking questions")

else:
    # Show chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    # ✅ INSIDE else block (correct indentation)
    user_question = st.chat_input(
        "Ask anything about your PDF..."
    )

    # ✅ INSIDE else block (correct indentation)
    if user_question:
        if st.session_state.chain is None:
            st.error("❌ Please upload and process a PDF first!")
        else:
            # Show user message
            with st.chat_message("user"):
                st.write(user_question)

            # Add to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })

            # Get AI response
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chain({
                        "question": user_question
                    })

                    answer = response.get(
                        "answer",
                        "No answer generated"
                    )
                    st.write(answer)
                
                # ✅ MOVED OUTSIDE spinner but INSIDE chat_message
                # Show sources
                sources = response.get(
                    "source_documents", []
                )
                if sources:
                    with st.expander("📍 Source Pages"):
                        for doc in sources:
                            page = doc.metadata.get('page', 'Unknown')
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            
                            st.write(f"**📄 {source_file}** - Page {page + 1}")
                            st.write(doc.page_content[:200])
                            st.divider()

            # Add response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })