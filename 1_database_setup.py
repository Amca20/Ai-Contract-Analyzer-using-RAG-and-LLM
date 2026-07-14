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
        r"\n(?=PART\s+[A-ZIVX]+)",     # Pisah pada permulaan 'PART' besar
        r"\n(?=BAHAGIAN\s+[A-ZIVX]+)", # Pisah pada permulaan 'BAHAGIAN'
        r"\n(?=\d+[A-Za-z]*\.\s)",     # Pisah pada corak '18. ' atau '60A. '
        r"\n\n",                       # Fallback: Perenggan baharu
        r"\n",                         # Fallback: Baris baharu
        r" "                           # Fallback: Ruang kosong
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=pemisah_akta,
        is_separator_regex=True,
        chunk_size=1500,
        chunk_overlap=150
    )
    
    chunks_akhir = text_splitter.split_documents(dokumen_mentah)
    print(f"Selesai! Dokumen dipotong kepada {len(chunks_akhir)} chunks yang mengikut konteks perundangan.")

    # ---------------------------------------------------------
    # LANGKAH 2B: PENGEKSTRAKAN METADATA AUTOMATIK (YANG KAU TERTINGGAL TADI)
    # ---------------------------------------------------------
    print("Mengekstrak dan memasukkan metadata automatik (Sumber & Seksyen)...")
    
    current_part = "Umum"
    current_section = "Umum"
    current_source = ""

    for chunk in chunks_akhir:
        # Dapatkan nama fail yang bersih dari sumber asal
        sumber_fail_penuh = chunk.metadata.get('source', '')
        nama_fail = os.path.basename(sumber_fail_penuh)
        
        # Reset memori jika skrip mula baca fail akta baharu
        if nama_fail != current_source:
            current_source = nama_fail
            current_part = "Umum" 
            current_section = "Umum"

        # Cari perkataan 'PART' atau 'BAHAGIAN'
        part_match = re.search(r'(PART\s+[A-ZIVX]+|BAHAGIAN\s+[A-ZIVX]+)', chunk.page_content, re.IGNORECASE)
        if part_match:
            current_part = part_match.group(1).upper()

        # Cari nombor seksyen (Cth: "60A. " atau "19. " di awal perenggan atau baris)
        sec_match = re.search(r'(?:^|\n)(\d+[A-Z]?)\.\s', chunk.page_content)
        if sec_match:
            current_section = sec_match.group(1)

        # Kemas kini metadata ke dalam chunk (Simpan ke ChromaDB nanti)
        chunk.metadata['nama_akta'] = nama_fail
        chunk.metadata['part_name'] = current_part
        chunk.metadata['section_num'] = current_section

        if chunks_akhir.index(chunk) < 5:
            print(f"Buktinya -> Fail: {nama_fail} | Part: {current_part} | Seksyen: {current_section}")

    # ---------------------------------------------------------
    # LANGKAH 3: GENERATE EMBEDDINGS (LEGAL-BERT)
    # ---------------------------------------------------------
    print("Memuat turun dan menyiapkan model embeddings (Legal-BERT)...")
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name="nlpaueb/legal-bert-base-uncased",
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
