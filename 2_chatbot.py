import streamlit as st
import PyPDF2
import re
import time

# Database & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLM & Prompt
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Hybrid Search & Reranker
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ==========================================
# PART 1: UI SETUP (LIGHT & CLEAN)
# ==========================================
st.set_page_config(
    page_title="AI Contract Risk Analyzer", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar setup
with st.sidebar:
    st.header("⚙️ Evaluation Settings")
    st.markdown("Configure the employee profile to ensure accurate analysis under the **Employment Act 1955**.")
    
    user_salary = st.number_input("Employee Monthly Salary (RM):", min_value=1000, max_value=50000, value=3000, step=500)
    st.info("ℹ️ **First Schedule 2022 Amendment:**\nEmployees earning > RM4,000 are not entitled to statutory overtime pay (Section 60).")
    
    st.divider()
    st.markdown("### 🧠 System Architecture")
    st.caption("• **LLM:** Llama 3.1\n• **Embeddings:** M-Legal BERT\n• **Retrieval:** Hybrid (BM25 + Vector)\n• **Reranker:** BGE-Reranker (CUDA)")

# Helper Functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def display_results(results_text, source_docs=None):
    st.success("✅ Analysis Complete!")
    clauses = results_text.split("---")

    for idx, clause in enumerate(clauses):
        if len(clause.strip()) > 10:
            with st.container(border=True):
                st.markdown(clause.strip())
                is_irrelevant = "⚪ UNKNOWN" in clause or "IRRELEVANT_CONTEXT" in clause

                if source_docs and not is_irrelevant:
                    doc_idx = min(idx, len(source_docs) - 1)
                    clause_docs = source_docs[doc_idx]
                    
                    if clause_docs:
                        st.divider()
                        st.markdown("#### 📚 Verified Legal Sources")
                        with st.expander("View Statutory Evidence Extracted by RAG"):
                            for i, doc in enumerate(clause_docs):
                                act_name = doc.metadata.get('nama_akta', 'Unknown Act')
                                section = doc.metadata.get('section_num', 'General')
                                part = doc.metadata.get('part_name', 'General')
                                
                                st.markdown(f"**Source {i+1}: {act_name}**")
                                st.caption(f"📍 Part: {part} | Section: {section}")
                                st.info(doc.page_content)
                    else:
                        st.write("No matching reference documents found.")

# ==========================================
# PART 2: INITIALIZE RAG PIPELINE
# ==========================================
@st.cache_resource(show_spinner="Initializing AI Models and Vector Database (CUDA)...")
def load_rag_system():
    # 1. SETUP LLM
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0,
        seed=42  # <-- Ini akan pastikan soalan dijana tak berubah-ubah
    )

# 2. SETUP QUERY TRANSFORMER (UNBIASED & GENERALIZED)
    transform_prompt = PromptTemplate.from_template(
        """You are an expert Malaysian employment lawyer. 
Read the following contract clause and identify the core legal rules regarding the Employment Act 1955 or Contracts Act 1950.
Generate ONLY ONE comprehensive natural language query to search for these rules in a semantic legal database.

CRITICAL RULES (DO NOT IGNORE):
1. ELEVATE TO LEGAL TERMINOLOGY: Translate any layman business terms in the clause into their formal statutory equivalents under Malaysian law. Rely on your general legal knowledge to find the right jargon.
2. MULTIPLE ISSUES: If the clause contains multiple legal mechanisms (e.g., timing of payment AND method of payment), combine them into a single broad search question.
3. ABSTRACT DETAILS: Remove specific names, exact dates, or exact currency amounts. Focus ONLY on the precise legal mechanisms.
4. ENGLISH ONLY: The output question must be in English.
5. STRICT OUTPUT: Output ONLY the generated question. No introductory text, no explanations.

<contract_clause>
{clause}
</contract_clause>

Search Query:"""
    )
    query_chain = transform_prompt | llm | StrOutputParser()

    # 3. SETUP EMBEDDINGS & CHROMADB (GPU/CUDA)
    embeddings_model = HuggingFaceEmbeddings(
        model_name="./custom-malaysian-legal-bert/checkpoint-5090",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

    # 4. SETUP HYBRID SEARCH (BM25 + VECTOR)
    all_data = db.get()
    bm25_retriever = BM25Retriever.from_texts(
        texts=all_data['documents'],
        metadatas=all_data['metadatas']
    )
    bm25_retriever.k = 8

    vector_retriever = db.as_retriever(search_kwargs={"k": 8})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 5. SETUP RERANKER (BGE-Reranker)
    reranker_model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={'device': 'cuda'}
    )
    compressor = CrossEncoderReranker(model=reranker_model, top_n=4)

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )

    # 6. SETUP GENERATOR MAIN PROMPT (FIXED NO PROMPT BLEEDING)
    eval_prompt = PromptTemplate.from_template(
        """You are a meticulous Malaysian Legal AI Assistant auditing an employment contract clause.
Your task is to strictly evaluate the <user_clause> against the provided <legal_context>.

CRITICAL RULES (DO NOT IGNORE):
CRITICAL RULES (DO NOT IGNORE):
1. STRICT RELEVANCE (ZERO TOLERANCE): If the <legal_context> does not directly and explicitly regulate the exact action described in the <user_clause>, you MUST immediately output ⚪ UNKNOWN. Do not force logical connections.
2. NO EXTERNAL KNOWLEDGE: Base your analysis ONLY on the <legal_context> provided.
3. OVERTIME SALARY OVERRIDE: The system's declared salary is RM {salary}. IF AND ONLY IF the <user_clause> explicitly mentions "overtime" AND the declared salary is > RM 4000, then classify as 🟢 SAFE. If the clause does NOT mention overtime, you MUST ignore this rule completely.
4. EXACT QUOTATION: Your 'Reference Evidence' MUST be a direct, literal copy-paste from the <legal_context>. Do not paraphrase or add words like "According to...".
5. EXACT QUOTATION: Your 'Reference Evidence' MUST be a direct quotation from the <legal_context>.

RISK CLASSIFICATION DEFINITIONS:
* 🟢 SAFE: Clause complies with or provides better benefits than the law.
* 🔴 DANGER: Clause directly violates the law, or provides less than the statutory minimum.
* 🟡 WARNING: Clause is legally ambiguous, heavily favors the employer unfairly, or deliberately omits mandatory statutory requirements.
* ⚪ UNKNOWN: The provided legal context is completely irrelevant to the clause.

FORMAT STRICTLY AS FOLLOWS:
### 📝 Original Clause
<Copy the exact clause>

### 🔎 Reference Evidence
<Extract 1-2 exact sentences from the context. Do not include this instruction. If UNKNOWN, write exactly: No relevant evidence found.>

### 🚦 Risk Status
<Choose ONLY ONE: 🟢 SAFE / 🟡 WARNING / 🔴 DANGER / ⚪ UNKNOWN>

### 🧠 AI Analysis
<If UNKNOWN: Write EXACTLY "IRRELEVANT_CONTEXT: The retrieved documents do not address the subject matter of this clause."
Otherwise, provide a 2-part reasoning analysis:
1. Legal Principle: Briefly state the rule from the reference evidence. Caution: Check carefully if the evidence mentions exceptions or items NOT included as mandatory wages.
2. Application: Explain precisely how the clause aligns or conflicts with the principle.>

### 💡 Recommendation
<If UNKNOWN: Write EXACTLY "No recommendation."
Otherwise: Provide a specific, actionable amendment to make the clause compliant and fair.>

--------------------------------------------------
<legal_context>
{context}
</legal_context>

<user_clause>
{clause}
</user_clause>

Analysis:"""
    )
    main_chain = eval_prompt | llm | StrOutputParser()
    
    return query_chain, main_chain, final_retriever

# Load models on startup
query_chain, main_chain, final_retriever = load_rag_system()

# ==========================================
# PART 3: MAIN INTERFACE
# ==========================================
st.title("⚖️ AI Employment Contract Risk Analyzer")
st.markdown("Automated contract auditing powered by **Hybrid RAG (BM25 + Semantic Vector)**, **Query Transformation**, and **Cross-Encoder Reranking**.")
st.divider()

tab1, tab2 = st.tabs(["✍️ Paste Contract Text", "📁 Upload PDF Document"])

with tab1:
    text_input = st.text_area("Paste contract clauses here (Separate multiple clauses with a double Enter):", height=200)
    btn_text = st.button("Analyze Text", type="primary", use_container_width=True)

with tab2:
    pdf_file = st.file_uploader("Upload Employment Contract (PDF)", type=["pdf"])
    btn_pdf = st.button("Analyze PDF Document", type="primary", use_container_width=True)

# ==========================================
# PART 4: LOGIC & PROCESSING
# ==========================================
def generate_report_txt(combined_results):
    return f"AI CONTRACT AUDIT REPORT\n=========================\nDeclared Salary: RM {user_salary}\n\n{combined_results}"

if btn_text:
    if text_input:
        start_time = time.time()
        with st.status("🔍 Processing Clauses via Hybrid Pipeline...", expanded=True) as status:
            input_clauses = text_input.split("\n\n") 
            final_results = []
            all_references = []

            for i, single_clause in enumerate(input_clauses):
                if len(single_clause.strip()) > 5:
                    clean_clause = single_clause.strip()
                    st.write(f"⚙️ Analyzing Clause {i+1}...")
                    
                    # 1. Query Transformation
                    generated_query = query_chain.invoke({"clause": clean_clause})
                    st.info(f"**Generated Semantic Query:** {generated_query.strip()}")
                    
                    # 2. Hybrid Search & Reranking
                    retrieved_docs = final_retriever.invoke(generated_query)
                    legal_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # 3. Final Evaluation
                    analysis = main_chain.invoke({
                        "context": legal_context, 
                        "clause": clean_clause,
                        "salary": user_salary
                    })
                    
                    final_results.append(analysis)
                    all_references.append(retrieved_docs if retrieved_docs else [])

            status.update(label=f"Analysis Complete in {round(time.time() - start_time, 2)} seconds!", state="complete", expanded=False)

        combined_text = "---".join(final_results)
        display_results(combined_text, source_docs=all_references)
        
        st.download_button(
            label="📥 Download Full Audit Report",
            data=generate_report_txt(combined_text),
            file_name="ai_contract_audit_report.txt",
            mime="text/plain"
        )

if btn_pdf:
    if pdf_file:
        start_time = time.time()
        with st.status("📄 Extracting and Processing PDF Clauses...", expanded=True) as status:
            full_text = extract_text_from_pdf(pdf_file)
            pdf_clauses = re.split(r'\n(?=\d{1,2}(?:\.\d{1,2})*\.\s+[A-Z])', full_text)
            
            final_results = []
            all_references = []
            valid_clauses = []

            for k in pdf_clauses:
                clean_k = k.strip()
                if re.match(r'^\d{1,2}(?:\.\d{1,2})*\.', clean_k):
                    valid_clauses.append(clean_k)

            if not valid_clauses:
                st.error("Failed to detect numbered clauses. Ensure the PDF uses standard numbering formatting (e.g., '1. Clause').")
                status.update(label="Extraction Failed", state="error")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress_bar = st.progress(0)
                with col2:
                    st.metric(label="Total Clauses Detected", value=len(valid_clauses))
                
                for i, clause in enumerate(valid_clauses):
                    clean_clause = clause.strip()
                    st.write(f"⚙️ Processing Clause {i+1}/{len(valid_clauses)}...")
                    
                    # 1. Query Transformation
                    generated_query = query_chain.invoke({"clause": clean_clause})
                    
                    # 2. Hybrid Search & Reranking
                    retrieved_docs = final_retriever.invoke(generated_query)
                    legal_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # 3. Final Evaluation
                    analysis = main_chain.invoke({
                        "context": legal_context, 
                        "clause": clean_clause,
                        "salary": user_salary
                    })
                    
                    final_results.append(analysis)
                    all_references.append(retrieved_docs if retrieved_docs else [])
                        
                    progress_bar.progress((i + 1) / len(valid_clauses))

                status.update(label=f"PDF Audited in {round(time.time() - start_time, 2)} seconds!", state="complete", expanded=False)
                
                combined_text = "---".join(final_results)
                display_results(combined_text, source_docs=all_references)
                
                st.download_button(
                    label="📥 Download Full PDF Audit Report",
                    data=generate_report_txt(combined_text),
                    file_name="pdf_contract_audit_report.txt",
                    mime="text/plain"
                )