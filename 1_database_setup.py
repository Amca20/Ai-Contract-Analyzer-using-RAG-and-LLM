import os
import re # INI PENTING UNTUK BACA METADATA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    print("Memulakan proses setup database...")

    # ---------------------------------------------------------
    # LANGKAH 1: BACA SEMUA FAIL PDF BUKU UNDANG-UNDANG
    # ---------------------------------------------------------
    folder_akta = "Database_Akta"
    
    # Semak sama ada folder wujud sebelum teruskan
    if not os.path.exists(folder_akta):
        print(f"Ralat: Folder '{folder_akta}' tidak dijumpai. Sila pastikan folder wujud dan letakkan fail PDF di dalamnya.")
        return

    print(f"Membaca dokumen PDF dari folder '{folder_akta}'...")
    loader = PyPDFDirectoryLoader(folder_akta)
    dokumen_mentah = loader.load()
    
    if not dokumen_mentah:
        print("Tiada fail PDF dijumpai dalam folder. Proses dihentikan.")
        return
        
    print(f"Berjaya membaca {len(dokumen_mentah)} mukasurat dokumen mentah.")
# ---------------------------------------------------------
    # LANGKAH 2A: ADVANCED SECTION-BASED CHUNKING (REGEX)
    # ---------------------------------------------------------
    print("Memulakan pemotongan teks menggunakan susunan hierarki Regex...")
    
    pemisah_akta = [
        # 1. Pastikan ia HANYA potong pada PART + Nombor Roman (I, V, X, L, C, D, M)
        r"\n(?=PART\s+[IVXLCDM]+\b)",     
        r"\n(?=BAHAGIAN\s+[IVXLCDM]+\b)", 
        # 2. Potong pada nombor Seksyen (Cth: "60E. ")
        r"\n(?=\d+[A-Z]?\.\s)",           
        # 3. Potong pada Sub-seksyen (Cth: "(1)", "(a)") supaya teks panjang tak terpotong tengah jalan
        r"\n(?=\([0-9a-zA-Z]+\)\s)",      
        r"\n\n",                       
        r"\n",                         
        r" "                           
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=pemisah_akta,
        is_separator_regex=True,
        chunk_size=1000,
        chunk_overlap=200 
    )
    
    chunks_akhir = text_splitter.split_documents(dokumen_mentah)

    # ---------------------------------------------------------
    # LANGKAH 2B: PENGEKSTRAKAN METADATA & SUNTIKAN KONTEKS
    # ---------------------------------------------------------
    print("Mengekstrak dan memasukkan metadata automatik (Sumber & Seksyen)...")
    
    current_part = "Umum"
    current_section = "Umum"
    current_source = ""

    for chunk in chunks_akhir:
        sumber_fail_penuh = chunk.metadata.get('source', '')
        nama_fail = os.path.basename(sumber_fail_penuh)
        nama_akta_bersih = nama_fail.replace(".pdf", "").strip()
        
        if nama_fail != current_source:
            current_source = nama_fail
            current_part = "Umum" 
            current_section = "Umum"

        # TANGKAP METADATA: Hanya cari perkataan PART diikuti Nombor Roman yang sah
        part_match = re.search(r'\b(PART|BAHAGIAN)\s+([IVXLCDM]+)\b', chunk.page_content)
        if part_match:
            current_part = f"{part_match.group(1)} {part_match.group(2)}"

        sec_match = re.search(r'(?:^|\n)(\d+[A-Z]?)\.\s', chunk.page_content)
        if sec_match:
            current_section = sec_match.group(1)

        chunk.metadata['nama_akta'] = nama_akta_bersih
        chunk.metadata['part_name'] = current_part
        chunk.metadata['section_num'] = current_section
        
        # PEMBERSIHAN TEKS (DATA CLEANING AUTOMATIK)
        # Buang teks "64 Laws of Malaysia ACT 265" atau apa-apa variasi header PDF
        teks_bersih = re.sub(r'\d+\s+Laws of Malaysia\s+ACT\s+\d+', '', chunk.page_content, flags=re.IGNORECASE)
        # Buang 'Employment 15', 'Contracts 13' dsb (biasanya di header)
        teks_bersih = re.sub(r'(Employment|Contracts)\s+\d+', '', teks_bersih, flags=re.IGNORECASE)
        # Sapu bersih semua line break dan jadikan perenggan padat
        teks_bersih = re.sub(r'\s+', ' ', teks_bersih).strip()
        
        # SUNTIKAN KONTEKS
        teks_diperkaya = f"[Sumber: {nama_akta_bersih}, Bahagian: {current_part}, Seksyen: {current_section}]\n{teks_bersih}"
        chunk.page_content = teks_diperkaya
        
    # ---------------------------------------------------------
    # LANGKAH 3: GENERATE EMBEDDINGS (LEGAL-BERT)
    # ---------------------------------------------------------
    print("Memuat turun dan menyiapkan model embeddings (Legal-BERT)...")
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name="./custom-malaysian-legal-bert/checkpoint-5090",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # ---------------------------------------------------------
    # LANGKAH 4: SIMPAN KE CHROMADB (LOCAL VECTOR DATABASE)
    # ---------------------------------------------------------
    folder_database = "./chroma_db"
    print(f"Menyimpan vektor ke dalam pangkalan data '{folder_database}'...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks_akhir,
        embedding=embeddings_model,
        persist_directory=folder_database
    )
    
    print("\n=================================================")
    print("✅ DATABASE BERJAYA DIBINA DAN DISIMPAN!")
    print("=================================================")

if __name__ == "__main__":
    main()