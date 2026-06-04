from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. SETUP TUKANG CARI (Guna LEGAL-BERT)
# Nota: Kali pertama run, dia akan guna internet untuk download fail model (~400MB)
print("[1] Memanaskan enjin model embedding Legal-BERT...")
embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

# 2. BUKA BUKU (Baca fail PDF)
nama_fail_pdf = "Companies Act 2016 Akta 777.pdf"
print(f"[2] Sedang membaca dokumen: {nama_fail_pdf}...")
loader = PyPDFLoader(nama_fail_pdf)
dokumen = loader.load()

# 3. POTONG TEKS (Chunking)
print("[3] Sedang mencincang teks kepada blok-blok kecil...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(dokumen)
print(f"    -> Berjaya dipotong kepada {len(chunks)} bahagian.")

# 4. SIMPAN DALAM CHROMA DB
print("[4] Sedang menukar teks kepada vector dan menyimpan ke dalam ChromaDB...")
database = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./Database_Akta"
)

print("\n🌟 MANTAP! Pangkalan data dengan otak LEGAL-BERT telah berjaya dibina.")