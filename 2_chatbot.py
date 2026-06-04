import streamlit as st
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import re

# --- IMPORTS BARU UNTUK HYBRID SEARCH & RERANKER ---
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- BAHAGIAN 1: SETUP UI & FUNGSI BANTUAN ---
st.set_page_config(page_title="AI Contract Risk Analyzer", page_icon="⚖️", layout="centered")

def extract_text_from_pdf(file):
    """Fungsi untuk tukar PDF ke teks biasa"""
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
                
                # Cuma cek: Adakah Llama rasa benda ni relevan?
                is_irrelevant = "⚪ UNKNOWN" in klausa or "IRRELEVANT_CONTEXT" in klausa

                # Kalau relevan (Llama bagi SAFE/WARNING/DANGER), WAJIB tunjuk rujukan
                if source_docs and not is_irrelevant:
                    doc_idx = min(idx, len(source_docs) - 1)
                    doc = source_docs[doc_idx]
                    
                    st.divider()
                    
                    # Cuba tarik skor (sekadar nak tahu kalau dia wujud)
                    score = doc.metadata.get('relevance_score') or doc.metadata.get('score') or 0.0
                    
                    # Logik Paparan Skor yang Bijak
                    if score > 0.0:
                        st.subheader(f"🟢 Rujukan Relevan (Confidence: {score:.2f})")
                    else:
                        # Kalau skor rosak/hilang (0.00), kita tayang ayat ni supaya tak nampak pelik
                        st.subheader("🟢 Rujukan Disahkan Oleh AI")
                        
                    with st.expander("📖 Lihat Rujukan Akta Dikesan"):
                        st.info(f"**Sumber:** {doc.metadata.get('source', 'Undang-undang Malaysia')}")
                        st.write(doc.page_content)
                        
    # BUTANG MUAT TURUN
    st.download_button(
        label="📥 Muat Turun Laporan Analisis (TXT)",
        data=hasil_teks,
        file_name="Analisis_Kontrak.txt",
        mime="text/plain"
    )

# --- BAHAGIAN 2: PANGGIL DATABASE & ARAHAN ---
@st.cache_resource
def load_rag_system():
    # 1. Load Embeddings & Vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    database = Chroma(persist_directory="./Database_Akta", embedding_function=embedding_model)
    
    # Tarik 10 Dokumen dari Vector
    vector_retriever = database.as_retriever(search_kwargs={"k": 10})
    
    # 2. Setup BM25 (Keyword Search)
    db_data = database.get()
    senarai_dokumen = [Document(page_content=teks, metadata=meta) for teks, meta in zip(db_data['documents'], db_data['metadatas'])]
    bm25_retriever = BM25Retriever.from_documents(senarai_dokumen)
    bm25_retriever.k = 10 
    
    # 3. Gabungkan Hybrid (50% Vector + 50% Keyword)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )
    
    # 4. Setup Reranker (Penapis Terakhir)
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )

    # Set temperature=0 untuk elak halusinasi/AI terlalu kreatif
    llm = Ollama(model="llama3.1", temperature=0)
    
    # --- PROMPT KUKU BESI TERBARU ---
    template_arahan = """You are a strict Malaysian Legal AI. Analyze the USER CONTRACT based ONLY on the LEGAL CONTEXT.

    CRITICAL RULE FOR IRRELEVANT CONTEXT: 
    If the LEGAL CONTEXT is completely unrelated to the USER CONTRACT (e.g., context is corporate law but contract is employment), you MUST set Risk Status to ⚪ UNKNOWN and write ONLY "IRRELEVANT_CONTEXT" in the AI Analysis.
    HOWEVER, you MUST strictly copy the exact text of the USER CONTRACT into the 'Original Clause' section. Never write 'None' or 'Not relevant' there.

    FORMAT TO FOLLOW STRICTLY:
    #### 📝 Original Clause
    "{question}"
    #### 🚦 Risk Status
    [🟢 SAFE / 🟡 WARNING / 🔴 DANGER / ⚪ UNKNOWN]
    #### 🧠 AI Analysis
    [Your analysis here. If irrelevant, just write IRRELEVANT_CONTEXT]
    #### 💡 Recommendation
    [Your advice or state 'No recommendation']
    ---
    LEGAL CONTEXT: {context}
    USER CONTRACT: {question}
    
    Analysis:"""
    
    PROMPT = PromptTemplate(template=template_arahan, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=final_retriever, 
        return_source_documents=True, # Wajib untuk UI tarik rujukan akta
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_system = load_rag_system()

# --- BAHAGIAN 3: RUANGAN INTERFACE (TABS & BUTANG) ---
st.title("⚖️ AI Contract Risk Analyzer")
st.write("Sistem RAG On-Premise. Dikuasakan oleh Llama 3.1 & Legal-BERT.")
st.divider()

st.markdown("### Contract Analysis Module")
st.info("Sila pastikan kontrak anda dalam Bahasa Inggeris untuk ketepatan carian akta yang optimum.")

tab1, tab2 = st.tabs(["✍️ Taip/Tampal Klausa", "📁 Muat Naik PDF"])

# TAB 1: KOTAK TEKS
with tab1:
    soalan_teks = st.text_area("Masukkan klausa kontrak di sini:", height=150)
    btn_teks = st.button("Analisis Teks", type="primary", key="btn1")

# TAB 2: MUAT NAIK PDF
with tab2:
    fail_pdf = st.file_uploader("Upload Contract (PDF)", type=["pdf"])
    if st.button("Analyze PDF", type="primary"):
        if fail_pdf:
            with st.spinner("Menyusun dan mengekstrak klausa kontrak (Regex Mode)..."):
                    teks_penuh = extract_text_from_pdf(fail_pdf)

                    # 1. THE REGEX MAGIC
                    # Dia potong TEPAT bila jumpa: [New Line] + [Nombor 1-99] + [Titik] + [Space] + [Huruf Besar]
                    # Contoh: "1. PARTIES", "10. UNILATERAL MODIFICATION"
                    senarai_klausa_pdf = re.split(r'\n(?=\d{1,2}\.\s+[A-Z])', teks_penuh)
                    
                    hasil_akhir = []
                    semua_rujukan = []
                    klausa_sah = []

                    # 2. TAPIS BAHAGIAN TAK MASUK AKAL (Contoh: Tajuk, Ruang Tandatangan)
                    for k in senarai_klausa_pdf:
                        k_bersih = k.strip()
                        # Hanya ambil blok yang BERMULA dengan format "Nombor. Huruf Besar"
                        if re.match(r'^\d{1,2}\.\s+[A-Z]', k_bersih):
                            klausa_sah.append(k_bersih)

                    if not klausa_sah:
                        st.error("Gagal mengesan susunan nombor klausa dalam PDF.")
                    else:
                        st.info(f"⏳ Dijumpai {len(klausa_sah)} klausa utama. Memulakan analisis spesifik...")
                        bar_progres = st.progress(0)
                        status_teks = st.empty()

                        for i, klausa in enumerate(klausa_sah):
                            status_teks.text(f"Sedang menganalisis klausa {i+1} dari {len(klausa_sah)}...")
                            
                            res = qa_system.invoke({"query": klausa})
                            hasil_akhir.append(res['result'])
                            
                            if res.get('source_documents'):
                                semua_rujukan.append(res['source_documents'][0])
                                
                            bar_progres.progress((i + 1) / len(klausa_sah))

                        status_teks.empty() 
                        
                        teks_gabungan = "---".join(hasil_akhir)
                        papar_hasil_ui(teks_gabungan, source_docs=semua_rujukan)

# --- BAHAGIAN 4: LOGIK EKSEKUSI BUTANG ---

# --- LOGIK BUTANG TEKS ---
# --- BAHAGIAN LOGIK BUTANG TEKS ---
if btn_teks:
    if soalan_teks:
        with st.spinner("Sedang membedah klausa satu per satu..."):
            # 1. PECAHKAN INPUT KEPADA KLAUSA INDIVIDU
            # Kita andaikan user guna nombor (1., 2.) atau baris baru
            senarai_input = soalan_teks.split("\n\n") 
            
            hasil_akhir = []
            semua_rujukan = []

            for klausa_tunggal in senarai_input:
                if len(klausa_tunggal.strip()) > 5:
                    # 2. CARI RUJUKAN KHAS UNTUK KLAUSA INI SAHAJA (Reset Search)
                    res = qa_system.invoke({"query": klausa_tunggal})
                    
                    # Simpan hasil analisis & rujukan yang sepadan
                    hasil_akhir.append(res['result'])
                    # Kita cuma ambil rujukan pertama yang paling tepat untuk klausa ini
                    if res.get('source_documents'):
                        semua_rujukan.append(res['source_documents'][0]) 

            # 3. GABUNGKAN SEMUA UNTUK PAPARAN UI
            teks_gabungan = "---".join(hasil_akhir)
            papar_hasil_ui(teks_gabungan, source_docs=semua_rujukan)

# --- BAHAGIAN LOGIK BUTANG TEKS ---
    if fail_pdf:
        with st.spinner("Sedang membedah klausa satu per satu..."):
            # 1. PECAHKAN INPUT KEPADA KLAUSA INDIVIDU
            # Kita andaikan user guna nombor (1., 2.) atau baris baru
            senarai_input = soalan_teks.split("\n\n") 
            
            hasil_akhir = []
            semua_rujukan = []

            for klausa_tunggal in senarai_input:
                if len(klausa_tunggal.strip()) > 5:
                    # 2. CARI RUJUKAN KHAS UNTUK KLAUSA INI SAHAJA (Reset Search)
                    res = qa_system.invoke({"query": klausa_tunggal})
                    
                    # Simpan hasil analisis & rujukan yang sepadan
                    hasil_akhir.append(res['result'])
                    # Kita cuma ambil rujukan pertama yang paling tepat untuk klausa ini
                    if res.get('source_documents'):
                        semua_rujukan.append(res['source_documents'][0]) 

            # 3. GABUNGKAN SEMUA UNTUK PAPARAN UI
            teks_gabungan = "---".join(hasil_akhir)
            papar_hasil_ui(teks_gabungan, source_docs=semua_rujukan)