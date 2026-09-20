import os
import time
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #fafafa; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        color: #fafafa !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #30363d;
        border-radius: 8px;
        padding: 8px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }

    /* Spinner */
    .stSpinner { color: #58a6ff; }

    /* Success / info banners */
    .stSuccess { background-color: #1a3a2a; border-color: #2ea043; }
    .stInfo    { background-color: #1a2a3a; border-color: #58a6ff; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Groq API key ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not set. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

# ── Load embedding model once ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ── Load LLM once ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )

# ── Build FAISS index from uploaded PDF ───────────────────────────────────────
def build_index(pdf_bytes, pdf_name, embeddings, status_text, progress_bar):

    # Step 1 — Save PDF to temp file
    status_text.markdown("**Step 1/4** — Loading PDF...")
    progress_bar.progress(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    progress_bar.progress(20)
    status_text.markdown(f"**Step 1/4** — Loaded {len(documents)} pages ✓")
    time.sleep(0.3)

    # Step 2 — Chunking
    status_text.markdown("**Step 2/4** — Splitting into chunks...")
    progress_bar.progress(30)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    progress_bar.progress(45)
    status_text.markdown(f"**Step 2/4** — Created {len(chunks)} chunks ✓")
    time.sleep(0.3)

    # Step 3 — Embedding (slowest step)
    status_text.markdown(f"**Step 3/4** — Embedding {len(chunks)} chunks into vectors... *(this is the slow part)*")
    progress_bar.progress(50)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    progress_bar.progress(85)
    status_text.markdown("**Step 3/4** — Vectors embedded ✓")
    time.sleep(0.3)

    # Step 4 — Build retriever
    status_text.markdown("**Step 4/4** — Building FAISS retriever...")
    progress_bar.progress(92)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    progress_bar.progress(100)
    status_text.markdown("**Step 4/4** — Index ready ✓")

    os.unlink(tmp_path)
    return retriever, len(documents), len(chunks)

# ── Format retrieved docs ──────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in docs
    )

# ── RAG chain ─────────────────────────────────────────────────────────────────
def ask(question, retriever, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions strictly based on the provided document context.
Always mention the page number(s) when citing information.
If the answer is not found in the context, say: 'I could not find this in the uploaded document.'
Be concise, accurate, and helpful.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": st.session_state.chat_history,
        "question": question,
    })

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))

    return answer, docs

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Document Assistant")
    st.markdown("Upload any PDF and ask questions about it. Answers are grounded in the document with page citations.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Any PDF — textbooks, papers, reports, manuals.",
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            st.markdown("**Building index...**")
            progress_bar = st.progress(0)
            status_text = st.empty()

            embeddings = load_embeddings()
            start = time.time()
            retriever, num_pages, num_chunks = build_index(
                uploaded_file.read(),
                uploaded_file.name,
                embeddings,
                status_text,
                progress_bar,
            )
            elapsed = time.time() - start

            progress_bar.empty()
            status_text.empty()

            st.session_state.retriever = retriever
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.num_pages = num_pages
            st.session_state.num_chunks = num_chunks
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.success(f"✅ Ready in {elapsed:.1f}s — ask your first question!")

    st.divider()

    if st.session_state.pdf_name:
        st.markdown("**Current document**")
        st.markdown(f"📘 `{st.session_state.pdf_name}`")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state.num_pages)
        with col2:
            st.metric("Chunks", st.session_state.num_chunks)

        st.divider()
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    st.markdown("""
**Tech stack**
- 🔍 FAISS vector search
- 🤗 MiniLM-L6-v2 embeddings
- ⚡ Groq — Llama 3.3 70B
- 🦜 LangChain LCEL
""")
    st.markdown("Built by [Aashish](https://github.com/aashish01784/Technical-Document-RAG-Assistant) · IIT Madras")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 📄 RAG Document Assistant")
st.caption("Ask questions about your PDF — answers grounded in the document with page citations.")

if not st.session_state.pdf_name:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #8b949e;'>
        <h3>👈 Upload a PDF from the sidebar to get started</h3>
        <p>Supports any PDF — textbooks, research papers, reports, manuals.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for doc in msg["sources"]:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page}:**")
                        st.caption(doc.page_content[:300] + "...")

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = load_llm()
                answer, sources = ask(question, st.session_state.retriever, llm)
            st.markdown(answer)
            with st.expander("📚 Sources"):
                for doc in sources:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page}:**")
                    st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
