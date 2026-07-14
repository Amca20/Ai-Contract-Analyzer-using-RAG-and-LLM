import streamlit as st
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import re

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ==========================================
# BAHAGIAN 1: SETUP UI & FUNGSI BANTUAN
# ==========================================
st.set_page_config(page_title="AI Contract Risk Analyzer", page_icon="⚖️", layout="wide")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def papar_hasil_ui(hasil_teks, source_docs=None):
    st.success("🎯 Analisis Selesai!")
    senarai_klausa = hasil_teks.split("---")

    for idx, klausa in enumerate(senarai_klausa):
        if len(klausa.strip()) > 10:
            with st.container(border=True):
                st.markdown(klausa.strip())
                is_irrelevant = "⚪ UNKNOWN" in klausa or "IRRELEVANT_CONTEXT" in klausa

                if source_docs and not is_irrelevant:
                    doc_idx = min(idx, len(source_docs) - 1)
                    senarai_doc_untuk_klausa = source_docs[doc_idx]
                    
                    if senarai_doc_untuk_klausa:
                        st.divider()
                        st.subheader("🟢 Rujukan Akta Disemak Oleh AI")
                        with st.expander("📚 Lihat Sumber Rujukan Akta"):
                            if sumber_dokumen:
                                for i, doc in enumerate(sumber_dokumen):
                                    st.markdown(f"**Rujukan {i+1}**")
                                    
                                    # ⚠️ PASTIKAN EJAAN KEY NI SEBIJIK MACAM NI:
                                    nama_akta = doc.metadata.get('nama_akta', 'Akta Tidak Diketahui')
                                    seksyen = doc.metadata.get('section_num', 'Umum')
                                    bahagian = doc.metadata.get('part_name', 'Umum')
                                    
                                    st.caption(f"📍 Sumber: {nama_akta} | Bahagian: {bahagian} | Seksyen: {seksyen}")
                                    st.info(doc.page_content)
                            else:
                                st.write("Tiada dokumen rujukan dijumpai.")

# ==========================================
# BAHAGIAN 2: PANGGIL DATABASE & ARAHAN AI
# ==========================================
@st.cache_resource
def load_rag_system():
    embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    database = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    vector_retriever = database.as_retriever(search_kwargs={"k": 25})
    
    db_data = database.get()
    senarai_dokumen = []
    if db_data and len(db_data['documents']) > 0:
        metadatas = db_data['metadatas'] if db_data['metadatas'] else [{}] * len(db_data['documents'])
        for teks, meta in zip(db_data['documents'], metadatas):
            senarai_dokumen.append(Document(page_content=teks, metadata=meta))

    bm25_retriever = BM25Retriever.from_documents(senarai_dokumen)
    bm25_retriever.k = 25
    
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.85, 0.15])  
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=7) 
    final_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid_retriever)

    llm = Ollama(model="llama3.1", temperature=0.0)
    
     # --- PROMPT EKSTRAK KATA KUNCI (VERSI HIBRID GLOBAL) ---
    template_keyword = """[INST] Extract the core legal concepts from the following contract clause. 
Provide 3 to 5 English keywords. If the clause uses local terms (like 'direct bank transfer' or 'overtime'), include the standard legal equivalents (like 'financial institution', 'working hours', 'remuneration').
Do not write full sentences. Separate keywords with commas. Ignore brackets like [FAKTA].

Clause: {klausa}
Keywords: [/INST]"""
    prompt_keyword = PromptTemplate(template=template_keyword, input_variables=["klausa"])
    keyword_chain = prompt_keyword | llm 

    # --- PROMPT BERSIH & SIMPLE (TANPA VARIABEL GAJI LUAR) ---
    # ==========================================
    # --- PROMPT MUKTAMAD: ANTI-HALUSINASI & FIRST SCHEDULE ---
    # ==========================================
    template_arahan = """You are a meticulous Malaysia Legal AI Assistant evaluating an employment contract clause.
Your ONLY task is to compare the 'USER CONTRACT' clause against the provided 'LEGAL CONTEXT'.

CRITICAL RULES:
1. STRICT RELEVANCE: If the LEGAL CONTEXT does not explicitly mention the subject matter of the USER CONTRACT, you MUST classify the risk as ⚪ UNKNOWN.
2. NO HALLUCINATION: You MUST NOT infer, assume, or guess any legal rules that are not explicitly stated in the LEGAL CONTEXT. If the clause cannot be evaluated based on the provided context, classify it as ⚪ UNKNOWN.
3. ZERO OUTSIDE KNOWLEDGE: Do not use external legal knowledge. 
4. MALAYSIAN EMPLOYMENT LAW (2022 AMENDMENT) OVERRIDE: Under the First Schedule of the Employment Act 1955, Overtime pay provisions (such as Section 60) DO NOT APPLY to employees earning more than RM 4,000 per month. If the user's declared salary is above RM 4,000, any clause denying overtime pay must be evaluated as 🟢 SAFE.
5. Under NO circumstances are you allowed to quote laws from your own memory. The 'Reference Evidence' MUST literally exist inside the provided LEGAL CONTEXT string. If you cannot find the exact words inside the LEGAL CONTEXT, you MUST fail it and output UNKNOWN. Do not try to be helpful by inventing evidence.

RISK STATUS DEFINITIONS (DO NOT GUESS):
* 🟢 SAFE: The clause fully complies with or provides better terms than the LEGAL CONTEXT.
* 🔴 DANGER: The clause directly and explicitly violates a clear rule stated in the LEGAL CONTEXT.
* 🟡 WARNING: The clause is legally ambiguous, lacks transparency, or heavily favors the employer without explicitly breaking the rules in the context.
* ⚪ UNKNOWN: The LEGAL CONTEXT provided is irrelevant to the clause.

FORMAT TO FOLLOW STRICTLY:
#### 📝 Original Clause
"{question}"

#### 🔎 Reference Evidence
[Copy 1 or 2 sentences exactly from the LEGAL CONTEXT above. Do not alter or paraphrase this text.]

#### 🚦 Risk Status
[Choose ONLY ONE: 🟢 SAFE / 🟡 WARNING / 🔴 DANGER / ⚪ UNKNOWN]

#### 🧠 AI Analysis
[If UNKNOWN: Write EXACTLY "IRRELEVANT_CONTEXT: The provided legal context does not cover the subject matter of this clause."
Otherwise: Explain precisely how the clause compares to the context. State facts only. Do not hallucinate.]

#### 💡 Recommendation
[If UNKNOWN: Write EXACTLY "No recommendation."
Otherwise: Provide a specific, actionable suggestion to amend the clause to comply with the context.]
---
LEGAL CONTEXT: {context}

USER CONTRACT: {question}

Analysis:"""
    prompt_utama = PromptTemplate(template=template_arahan, input_variables=["context", "question"])
    main_chain = prompt_utama | llm
    
    return keyword_chain, main_chain, final_retriever

keyword_chain, main_chain, final_retriever = load_rag_system()

# ==========================================
# BAHAGIAN 3: INTERFACE UTAMA
# ==========================================
st.title("⚖️ AI Contract Risk Analyzer")
st.divider()

tab1, tab2 = st.tabs(["✍️ Taip/Tampal Klausa", "📁 Muat Naik PDF"])

with tab1:
    soalan_teks = st.text_area("Masukkan klausa kontrak di sini (Jarakkan klausa dengan butang Enter 2 kali):", height=150)
    btn_teks = st.button("Analisis Teks", type="primary")

with tab2:
    fail_pdf = st.file_uploader("Upload Contract (PDF)", type=["pdf"])
    btn_pdf = st.button("Analyze PDF", type="primary")

# ==========================================
# BAHAGIAN 4: LOGIK BUTANG
# ==========================================
if btn_teks:
    if soalan_teks:
        with st.spinner("Sedang membedah klausa satu per satu..."):
            senarai_input = soalan_teks.split("\n\n") 
            hasil_akhir = []
            semua_rujukan = []

            for klausa_tunggal in senarai_input:
                if len(klausa_tunggal.strip()) > 5:
                    klausa_bersih = klausa_tunggal.strip()
                    
                    kata_kunci = keyword_chain.invoke({"klausa": klausa_bersih})
                    st.caption(f"🔍 *Mencari akta menggunakan kata kunci:* `{kata_kunci}`")
                    
                    sumber_dokumen = final_retriever.invoke(kata_kunci)
                    teks_akta = "\n\n".join([doc.page_content for doc in sumber_dokumen])
                    
                    jawapan = main_chain.invoke({
                        "context": teks_akta, 
                        "question": klausa_bersih
                    })
                    
                    hasil_akhir.append(jawapan)
                    semua_rujukan.append(sumber_dokumen if sumber_dokumen else [])

            teks_gabungan = "---".join(hasil_akhir)
            papar_hasil_ui(teks_gabungan, source_docs=semua_rujukan)

if btn_pdf:
    if fail_pdf:
        with st.spinner("Menyusun dan mengekstrak klausa kontrak (Regex Mode)..."):
            teks_penuh = extract_text_from_pdf(fail_pdf)
            senarai_klausa_pdf = re.split(r'\n(?=\d{1,2}\.\s+[A-Z])', teks_penuh)
            
            hasil_akhir = []
            semua_rujukan = []
            klausa_sah = []

            for k in senarai_klausa_pdf:
                k_bersih = k.strip()
                if re.match(r'^\d{1,2}\.\s+[A-Z]', k_bersih):
                    klausa_sah.append(k_bersih)

            if not klausa_sah:
                st.error("Gagal mengesan susunan nombor klausa.")
            else:
                bar_progres = st.progress(0)
                status_teks = st.empty()

                for i, klausa in enumerate(klausa_sah):
                    status_teks.text(f"Sedang menganalisis klausa {i+1} dari {len(klausa_sah)}...")
                    klausa_bersih = klausa.strip()
                    
                    kata_kunci = keyword_chain.invoke({"klausa": klausa_bersih})
                    sumber_dokumen = final_retriever.invoke(kata_kunci)
                    teks_akta = "\n\n".join([doc.page_content for doc in sumber_dokumen])
                    
                    jawapan = main_chain.invoke({
                        "context": teks_akta, 
                        "question": klausa_bersih
                    })
                    
                    hasil_akhir.append(jawapan)
                    semua_rujukan.append(sumber_dokumen if sumber_dokumen else [])
                        
                    bar_progres.progress((i + 1) / len(klausa_sah))

                status_teks.empty() 
                teks_gabungan = "---".join(hasil_akhir)
                papar_hasil_ui(teks_gabungan, source_docs=semua_rujukan)
