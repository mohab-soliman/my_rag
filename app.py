import streamlit as st
import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Library Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ── MODERN UI / CSS ─────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;700&display=swap');

:root{
    --bg:#0B0B0C;
    --card:#151515;
    --border:#262626;
    --text:#E8E2D9;
    --muted:#8B8B8B;
    --accent:#C9A96E;
}

html, body, [class*="css"]{
    font-family:'DM Sans', sans-serif;
    background:var(--bg);
    color:var(--text);
}

body{
    background:
        radial-gradient(circle at top left, rgba(201,169,110,.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(201,169,110,.06), transparent 25%),
        #0B0B0C;
}

#MainMenu,
footer,
header{
    visibility:hidden;
}

.stApp{
    background:transparent;
}

.block-container{
    max-width:920px;
    padding-top:2rem;
    padding-bottom:2rem;
}


/* ── HERO ───────────────────────────────────────────── */

.hero{
    position:relative;
    overflow:hidden;

    padding:3rem 2rem;
    margin-bottom:2rem;

    border-radius:28px;

    background:
        linear-gradient(
            145deg,
            rgba(28,28,28,.95),
            rgba(14,14,14,.98)
        );

    border:1px solid rgba(201,169,110,.15);

    box-shadow:
        0 10px 40px rgba(0,0,0,.45),
        inset 0 1px 0 rgba(255,255,255,.03);
}

.hero::before{
    content:"";
    position:absolute;

    width:260px;
    height:260px;

    background:rgba(201,169,110,.08);

    border-radius:50%;

    top:-120px;
    right:-100px;

    filter:blur(40px);
}

.hero-grid{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:2rem;

    position:relative;
    z-index:2;
}

.hero-text h1{
    margin:0;

    font-family:'DM Serif Display', serif;
    font-size:3.2rem;
    line-height:1;

    letter-spacing:-1px;
    color:var(--text);
}

.hero-text p{
    margin-top:1rem;

    color:var(--muted);

    font-size:1rem;
    line-height:1.8;

    max-width:580px;
}

.badge{
    display:inline-flex;
    align-items:center;
    gap:.45rem;

    padding:.45rem .9rem;
    margin-bottom:1.3rem;

    border-radius:999px;

    background:rgba(201,169,110,.08);

    border:1px solid rgba(201,169,110,.12);

    color:#D8C29A;

    font-size:.82rem;
    font-weight:500;
}


/* ── LOGO ───────────────────────────────────────────── */

.hero-logo{
    width:120px;
    height:120px;

    border-radius:24px;

    background:#111;

    border:1px solid rgba(201,169,110,.18);

    display:flex;
    align-items:center;
    justify-content:center;

    overflow:hidden;
}

/*
=========================================
PUT YOUR LOGO HERE
=========================================

1) ضع صورة باسم:
logo.png

2) جنب ملف:
app.py

=========================================
*/

.hero-logo img{
    width:100%;
    height:100%;
    object-fit:contain;
}


/* ── CHAT ───────────────────────────────────────────── */

[data-testid="stChatMessage"]{
    background:transparent !important;
    border:none !important;
    padding:.45rem 0 !important;
}

[data-testid="stChatMessageAvatarUser"]{
    background:linear-gradient(135deg,#C9A96E,#9F7B45) !important;
}

[data-testid="stChatMessageAvatarAssistant"]{
    background:#151515 !important;
}

[data-testid="stChatMessage"] .stMarkdown{
    padding:1rem 1.2rem;
    border-radius:18px;
    line-height:1.8;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown{
    background:#1A1A1A;
    border:1px solid #2A2A2A;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown{
    background:#141414;
    border:1px solid rgba(201,169,110,.12);
}


/* ── INPUT ───────────────────────────────────────────── */

[data-testid="stChatInput"]{
    padding-top:1rem;
}

[data-testid="stChatInputTextArea"]{
    background:#161616 !important;

    color:var(--text) !important;

    border-radius:18px !important;

    border:1px solid #2A2A2A !important;

    padding:.9rem 1rem !important;
}

[data-testid="stChatInputTextArea"]:focus{
    border:1px solid rgba(201,169,110,.6) !important;

    box-shadow:
        0 0 0 4px rgba(201,169,110,.12) !important;
}


/* ── REFERENCES ─────────────────────────────────────── */

.reference-card{
    margin-top:.8rem;

    padding:1rem;

    border-radius:16px;

    background:#121212;

    border:1px solid rgba(201,169,110,.08);
}

[data-testid="stExpander"]{
    border-radius:16px !important;

    border:1px solid rgba(201,169,110,.08) !important;

    background:#121212 !important;
}


/* ── SCROLLBAR ──────────────────────────────────────── */

::-webkit-scrollbar{
    width:5px;
}

::-webkit-scrollbar-thumb{
    background:#2C2C2C;
    border-radius:20px;
}

</style>


<div class="hero">

    <div class="hero-grid">

        <div class="hero-text">

            <div class="badge">
                ✨ AI Powered Library Retrieval System
            </div>

            <h1>
                📚 Library Assistant
            </h1>

            <p>
                Search your library collection intelligently using
                Retrieval-Augmented Generation (RAG),
                semantic search,
                and Gemini AI.
            </p>

        </div>

        <div class="hero-logo">
            <img src="logo.png">
        </div>

    </div>

</div>

""", unsafe_allow_html=True)


# ── APA Citation Helper ──────────────────────────────────────
def build_apa_citation(metadata):

    author = metadata.get("author", "Unknown Author")
    year = metadata.get("year", "n.d.")
    title = metadata.get("title", "Untitled")
    publisher = metadata.get("publisher", "")
    page = metadata.get("page", None)

    citation = f"{author}. ({year}). *{title}*."

    if publisher:
        citation += f" {publisher}."

    if page:
        try:
            citation += f" p. {int(page) + 1}"
        except:
            pass

    return citation


# ── Build RAG Chain ──────────────────────────────────────────
@st.cache_resource
def build_rag_chain():

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        persist_directory="vector_DB",
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    system_prompt = (
        "You are a professional Library Assistant. "
        "Use the retrieved context about books to answer the user's question. "
        "If the answer is not in the context, say that you don't know. "
        "\\n\\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    return create_retrieval_chain(
        retriever,
        combine_docs_chain
    )


# ── SESSION STATE ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── LOAD RAG CHAIN ───────────────────────────────────────────
rag_chain = build_rag_chain()


# ── DISPLAY OLD MESSAGES ─────────────────────────────────────
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ── USER INPUT ───────────────────────────────────────────────
query = st.chat_input(
    "Ask about a book, author, or topic..."
)


# ── PROCESS QUERY ────────────────────────────────────────────
if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Searching library collection..."):

        result = rag_chain.invoke({
            "input": query
        })

    answer = result.get(
        "answer",
        "No answer found."
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):

        st.write(answer)

        # ── REFERENCES ─────────────────────────────
        st.markdown("---")
        st.markdown(
            "<small>📎 <b>References</b></small>",
            unsafe_allow_html=True
        )

        seen_citations = set()

        for doc in result.get("context", []):

            citation = build_apa_citation(
                doc.metadata
            )

            if citation not in seen_citations:

                seen_citations.add(citation)

                st.markdown(
                    f'''
                    <div class="reference-card">
                        {citation}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        # ── SOURCE CHUNKS ─────────────────────────
        with st.expander("View Retrieved Chunks"):

            for i, doc in enumerate(
                result.get("context", []),
                1
            ):

                st.markdown(f"### Chunk {i}")

                st.write(doc.page_content)

                st.divider()
