import os
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from langchain_ollama import OllamaEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import random

# Sample topics and content for marketing and operational data
TOPICS = {
    "marketing": {
        "branding": [
            "Branding merupakan proses menciptakan identitas yang kuat dan konsisten untuk bisnis. Ini melibatkan pembuatan logo, pilihan warna, pesan, dan positioning yang konsisten untuk membedakan perusahaan dari pesaingnya.",
            "Brand awareness adalah kemampuan konsumen untuk mengenali dan mengingat merek Anda. Ini merupakan langkah pertama dalam customer journey dan sangat penting untuk keberhasilan jangka panjang.",
            "Brand positioning adalah cara merek diposisikan di benak konsumen. Ini harus didasarkan pada nilai unik yang ditawarkan perusahaan dan membantu membedakannya dari pesaing.",
            "Brand equity adalah nilai tambah yang diberikan produk atau layanan dari kekuatan mereknya. Ini termasuk kesadaran merek, loyalitas pelanggan, persepsi kualitas, dan asosiasi merek.",
            "Brand consistency adalah memastikan semua aspek merek - visual, verbal, dan pengalaman - tetap konsisten di semua saluran dan titik kontak dengan pelanggan."
        ],
        "kampanye_pemasaran": [
            "Kampanye pemasaran digital melibatkan penggunaan saluran online seperti email, media sosial, SEO, dan iklan PPC untuk menjangkau target audiens. Strategi ini memungkinkan pemasaran yang lebih tepat sasaran dan terukur.",
            "Kampanye email marketing yang efektif dibangun dengan segmentasi yang cermat, personalisasi yang kuat, dan pengujian A/B untuk meningkatkan tingkat open rate dan conversion rate.",
            "Kampanye konten marketing berfokus pada penciptaan dan distribusi konten yang berharga, relevan, dan konsisten untuk menarik dan mempertahankan audiens yang ditentukan dengan jelas - dan, pada akhirnya, untuk mendorong tindakan pelanggan yang menguntungkan.",
            "Kampanye media sosial harus mempertimbangkan platform mana yang paling sesuai dengan target audiens dan tujuan bisnis, jenis konten yang paling efektif di platform tersebut, dan frekuensi posting optimal.",
            "Kampanye influencer marketing menggunakan individu dengan pengikut besar atau pengaruh dalam niche tertentu untuk mempromosikan produk atau layanan. Ini dapat sangat efektif untuk menjangkau audiens baru dan membangun kepercayaan."
        ],
        "customer_relationship": [
            "Customer Relationship Management (CRM) adalah pendekatan untuk mengelola interaksi perusahaan dengan pelanggan saat ini dan potensial, menggunakan analisis data tentang riwayat pelanggan untuk meningkatkan hubungan bisnis, khususnya berfokus pada retensi pelanggan.",
            "Customer lifetime value (CLV) adalah prediksi nilai bersih yang terkait dengan seluruh hubungan masa depan dengan pelanggan. Metrik ini membantu bisnis untuk mengambil keputusan tentang berapa banyak yang harus diinvestasikan dalam memperoleh pelanggan baru.",
            "Customer segmentation adalah praktik membagi pelanggan ke dalam kelompok berdasarkan karakteristik umum seperti demografi, perilaku, atau nilai. Ini memungkinkan perusahaan untuk menargetkan pemasaran mereka secara lebih efektif.",
            "Customer loyalty programs dirancang untuk mendorong pelanggan untuk terus melakukan pembelian dengan memberikan insentif seperti diskon, hadiah, atau akses eksklusif. Program yang efektif meningkatkan retensi dan nilai pelanggan.",
            "Customer feedback loops adalah sistem untuk mengumpulkan, menganalisis, dan bertindak berdasarkan umpan balik pelanggan secara berkelanjutan. Ini sangat penting untuk peningkatan produk, layanan, dan pengalaman pelanggan secara keseluruhan."
        ],
        "analisis_pasar": [
            "Analisis pesaing melibatkan identifikasi dan evaluasi kekuatan dan kelemahan pesaing utama. Ini membantu perusahaan untuk memposisikan diri secara efektif di pasar dan mengidentifikasi peluang untuk diferensiasi.",
            "Riset pasar adalah proses pengumpulan, analisis, dan interpretasi informasi tentang pasar, produk, atau layanan yang akan ditawarkan untuk dijual di pasar itu, dan tentang pelanggan potensial dan yang sudah ada.",
            "Segmentasi pasar adalah proses membagi pasar besar menjadi kelompok konsumen yang lebih kecil dengan kebutuhan, karakteristik, atau perilaku yang berbeda yang mungkin memerlukan strategi pemasaran atau bauran produk terpisah.",
            "Target market adalah sekelompok pelanggan yang dipilih perusahaan untuk ditargetkan dengan upaya pemasarannya. Ini ditentukan melalui segmentasi pasar dan analisis untuk mengidentifikasi kelompok yang paling menguntungkan untuk dilayani.",
            "SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) adalah teknik perencanaan strategis yang membantu perusahaan mengidentifikasi kekuatan, kelemahan, peluang, dan ancaman yang terkait dengan persaingan bisnis atau perencanaan proyek."
        ]
    },
    "operasional": {
        "manajemen_gudang": [
            "Manajemen gudang adalah proses yang memastikan bahwa semua operasi di dalam gudang berjalan seefisien mungkin. Ini meliputi penerimaan, penyimpanan, pengambilan, dan pengiriman barang.",
            "Sistem manajemen gudang (WMS) adalah perangkat lunak yang membantu dalam pengelolaan dan kontrol operasi gudang harian, termasuk pelacakan inventaris, pengisian stok, dan optimalisasi ruang.",
            "Proses penerimaan gudang melibatkan verifikasi bahwa barang yang diterima sesuai dengan pesanan pembelian dalam hal kuantitas dan kualitas, serta memperbarui catatan inventaris secara akurat.",
            "Metode penyimpanan dan pengambilan barang seperti FIFO (First-In-First-Out) dan LIFO (Last-In-First-Out) memiliki implikasi signifikan untuk manajemen inventaris, terutama untuk barang yang memiliki tanggal kedaluwarsa.",
            "Optimalisasi tata letak gudang adalah kunci untuk meningkatkan efisiensi operasional. Barang yang sering dipesan harus disimpan di lokasi yang mudah diakses untuk mengurangi waktu pengambilan dan biaya tenaga kerja."
        ],
        "manajemen_inventaris": [
            "Manajemen inventaris adalah proses pemesanan, penyimpanan, dan penggunaan inventaris perusahaan: bahan mentah, komponen, dan produk jadi, serta pengemasan dan barang konsumsi.",
            "Economic Order Quantity (EOQ) adalah model inventaris yang menentukan jumlah optimal barang yang harus dipesan untuk meminimalkan total biaya inventaris, termasuk biaya penyimpanan dan biaya pemesanan.",
            "Safety stock adalah inventaris tambahan yang disimpan untuk mengurangi risiko stockout yang disebabkan oleh ketidakpastian dalam permintaan atau lead time pengiriman.",
            "ABC analysis adalah teknik untuk mengkategorikan item inventaris berdasarkan nilai dan volume penjualan tahunan, dengan item 'A' menyumbang nilai tertinggi dan memerlukan kontrol inventaris yang paling ketat.",
            "Inventory turnover adalah rasio yang mengukur berapa kali inventaris perusahaan telah terjual selama periode tertentu. Rasio yang tinggi menunjukkan manajemen inventaris yang baik dan permintaan yang kuat."
        ],
        "logistik_pengiriman": [
            "Manajemen rantai pasokan (supply chain management) adalah pengawasan barang, informasi, dan keuangan saat mereka bergerak dalam proses dari pemasok ke produsen ke pedagang grosir ke pengecer ke konsumen.",
            "Logistik last-mile adalah tahap terakhir dalam proses pengiriman, di mana barang bergerak dari pusat distribusi ke tujuan akhirnya. Ini sering kali adalah bagian yang paling mahal dan kompleks dari proses pengiriman.",
            "Cross-docking adalah praktik logistik di mana produk dari pemasok atau pabrik dikirim langsung ke pelanggan atau toko ritel dengan sedikit atau tanpa penanganan atau penyimpanan sementara.",
            "Reverse logistics berkaitan dengan penanganan dan pengelolaan barang yang dikembalikan ke pemasok, produsen, atau pengecer, serta reklamasi produk, recycling, dan pembuangan produk.",
            "Freight forwarding adalah pengaturan pengiriman barang dari satu titik ke titik lainnya melalui satu atau beberapa moda transportasi. Forwarder bertindak sebagai perantara antara pengiriman dan berbagai layanan transportasi."
        ],
        "kontrol_kualitas": [
            "Kontrol kualitas (QC) adalah proses yang memastikan produk dan layanan memenuhi harapan pelanggan. Ini melibatkan pemeriksaan dan pengujian untuk mengidentifikasi dan memperbaiki cacat.",
            "Six Sigma adalah metodologi untuk meningkatkan kualitas proses bisnis dengan mengurangi variabilitas dan menghilangkan cacat. Tujuannya adalah untuk mencapai tidak lebih dari 3,4 cacat per sejuta peluang.",
            "Kaizen adalah filosofi bisnis Jepang yang berfokus pada peningkatan berkelanjutan. Dalam kontrol kualitas, ini melibatkan peningkatan inkremental kecil yang berkelanjutan dalam proses produksi atau layanan.",
            "Total Quality Management (TQM) adalah pendekatan manajemen di mana semua anggota organisasi berpartisipasi dalam meningkatkan proses, produk, layanan, dan budaya di mana mereka bekerja, dengan tujuan akhir meningkatkan kualitas.",
            "Statistical Process Control (SPC) menggunakan metode statistik untuk memantau dan mengontrol proses untuk memastikan bahwa ia beroperasi secara efisien dan menghasilkan produk yang memenuhi spesifikasi."
        ],
        "manajemen_operasional": [
            "Manajemen operasional berkaitan dengan desain dan kontrol proses produksi dan bisnis dalam produksi barang atau jasa, dengan fokus pada penggunaan sumber daya secara efektif dan efisien.",
            "Key Performance Indicators (KPIs) adalah metrik yang digunakan untuk mengukur kinerja operasional. Contoh umum termasuk produktivitas, efisiensi, kualitas, dan waktu siklus.",
            "Lean management adalah pendekatan untuk menjalankan organisasi yang mendukung konsep peningkatan berkelanjutan, sebuah pendekatan jangka panjang untuk bekerja secara sistematis untuk mencapai perubahan kecil dan bertahap.",
            "Just-in-Time (JIT) adalah sistem produksi yang bertujuan untuk meningkatkan efisiensi dan mengurangi biaya dengan menerima bahan hanya sesuai kebutuhan dalam proses produksi, mengurangi inventaris dan biaya terkait.",
            "Business Process Reengineering (BPR) adalah analisis dan desain ulang alur kerja dan proses bisnis dalam suatu organisasi. BPR bertujuan untuk membantu organisasi secara fundamental memikirkan kembali bagaimana mereka melakukan pekerjaan."
        ]
    }
}

def setup_weaviate():
    """Initialize Weaviate client with cloud credentials"""
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url="epu3qawrpkjx7m0zafxq.c0.asia-southeast1.gcp.weaviate.cloud",
        auth_credentials=Auth.api_key("UKIqKrRSlhI633pl9GQbQgqWqqBGhEOQtIac"),
    )
    
    if not client.collections.exists("Document"):
        try:
            questions = client.collections.create(
                name="Document",
                vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),    # Set the vectorizer to "text2vec-openai" to use the OpenAI API for vector-related operations
                generative_config=wvc.config.Configure.Generative.cohere(),             # Set the generative module to "generative-cohere" to use the Cohere API for RAG
                properties=[
                    wvc.config.Property(
                        name="content",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="category",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="subcategory",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="source",
                        data_type=wvc.config.DataType.TEXT,
                    )
                ]
            )


            print(questions.config.get(simple=False))

        finally:
            client.close()
    
    return client

def generate_documents(output_dir="./data", category=None):
    """Generate dummy documents from predefined topics"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    created_files = []
    
    # If category is specified, only generate documents for that category
    categories_to_generate = [category] if category else TOPICS.keys()
    
    for category in categories_to_generate:
        for subcategory, paragraphs in TOPICS[category].items():
            # Generate a document that includes all paragraphs for this subcategory
            content = f"# {subcategory.replace('_', ' ').title()}\n\n"
            content += "\n\n".join(paragraphs)
            
            # Save to file
            filename = f"{category}_{subcategory}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(content)
            
            created_files.append({
                "path": filepath,
                "category": category,
                "subcategory": subcategory
            })
    
    return created_files

def load_documents_to_weaviate(client, files):
    """Load documents into domain-specific vector stores in Weaviate"""
    # Setup embeddings
    embeddings = OllamaEmbeddings(model="neural-chat")
    
    # Split files by category
    marketing_files = [f for f in files if f["category"] == "marketing"]
    operasional_files = [f for f in files if f["category"] == "operasional"]
    
    # Create category-specific vector stores
    vectorstore_marketing = WeaviateVectorStore(
        client=client,
        index_name="Document",
        text_key="content",
        embedding=embeddings,
        attributes=["category", "subcategory", "source"]
    )
    
    vectorstore_operasional = WeaviateVectorStore(
        client=client,
        index_name="Document",
        text_key="content",
        embedding=embeddings,
        attributes=["category", "subcategory", "source"]
    )
    
    # Process marketing documents
    marketing_docs = []
    for file_info in marketing_files:
        try:
            loader = TextLoader(file_info["path"])
            docs = loader.load()
            
            # Add metadata to documents
            for doc in docs:
                doc.metadata["category"] = file_info["category"]
                doc.metadata["subcategory"] = file_info["subcategory"]
                doc.metadata["source"] = os.path.basename(file_info["path"])
            
            marketing_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {file_info['path']}: {e}")
    
    # Process operational documents
    operasional_docs = []
    for file_info in operasional_files:
        try:
            loader = TextLoader(file_info["path"])
            docs = loader.load()
            
            # Add metadata to documents
            for doc in docs:
                doc.metadata["category"] = file_info["category"]
                doc.metadata["subcategory"] = file_info["subcategory"]
                doc.metadata["source"] = os.path.basename(file_info["path"])
            
            operasional_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {file_info['path']}: {e}")
    
    # Split and load documents by category
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    chunks_count = 0
    
    if marketing_docs:
        marketing_chunks = text_splitter.split_documents(marketing_docs)
        vectorstore_marketing.add_documents(marketing_chunks)
        chunks_count += len(marketing_chunks)
        print(f"Added {len(marketing_chunks)} marketing chunks to Weaviate")
    
    if operasional_docs:
        operasional_chunks = text_splitter.split_documents(operasional_docs)
        vectorstore_operasional.add_documents(operasional_chunks)
        chunks_count += len(operasional_chunks)
        print(f"Added {len(operasional_chunks)} operational chunks to Weaviate")
    
    return chunks_count, len(files)

def main():
    parser = argparse.ArgumentParser(description='Generate marketing and operational dummy data for Weaviate Cloud')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory to store/read documents')
    parser.add_argument('--category', type=str, choices=['marketing', 'operasional', 'all'],
                      default='all', help='Category of documents to generate')
    parser.add_argument('--regenerate', action='store_true',
                      help='Force regenerate all documents')
    
    args = parser.parse_args()
    
    # Setup Weaviate client
    client = setup_weaviate()
    
    # Check if we need to generate new data
    if args.regenerate and os.path.exists(args.data_dir):
        import shutil
        print(f"Removing existing data in {args.data_dir}...")
        for file in os.listdir(args.data_dir):
            if file.endswith('.txt'):
                os.remove(os.path.join(args.data_dir, file))
    
    # Generate documents if needed
    category_to_generate = None if args.category == 'all' else args.category
    
    if args.regenerate or not os.path.exists(args.data_dir) or not any(f.endswith('.txt') for f in os.listdir(args.data_dir)):
        print(f"Generating documents for {'all categories' if category_to_generate is None else category_to_generate}...")
        files = generate_documents(args.data_dir, category_to_generate)
        print(f"Generated {len(files)} documents")
    else:
        # Collect existing files with their categories
        files = []
        for filename in os.listdir(args.data_dir):
            if filename.endswith('.txt'):
                parts = filename.replace('.txt', '').split('_')
                if len(parts) >= 2:
                    category = parts[0]
                    subcategory = '_'.join(parts[1:])
                    
                    # Skip if we're only processing a specific category
                    if category_to_generate and category != category_to_generate:
                        continue
                    
                    files.append({
                        "path": os.path.join(args.data_dir, filename),
                        "category": category,
                        "subcategory": subcategory
                    })
    
    # Load documents into Weaviate
    if files:
        chunks_count, docs_count = load_documents_to_weaviate(client, files)
        print(f"Loaded {chunks_count} chunks from {docs_count} documents into Weaviate")
        
        # Print example queries
        print("\nYou can now run the Flask application to query this data.")
        print("Example marketing queries you might try:")
        print("- Apa strategi terbaik untuk kampanye pemasaran digital?")
        print("- Bagaimana cara meningkatkan brand awareness?")
        print("- Jelaskan tentang customer segmentation dan manfaatnya.")
        
        print("\nExample operational queries you might try:")
        print("- Bagaimana cara mengoptimalkan manajemen gudang?")
        print("- Apa itu sistem Just-in-Time dalam logistik?")
        print("- Jelaskan tentang metode kontrol kualitas dalam operasional.")
    else:
        print("No files to process.")
    client.close()
    
if __name__ == "__main__":
    main()