from langchain_ollama import ChatOllama
from flask import Flask, request, jsonify
from flask_cors import CORS
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate import WeaviateVectorStore
from weaviate.collections.classes.filters import Filter
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import os
from typing import TypedDict, Dict, List

# Perbaikan: Import AuthApiKey dari weaviate (Sudah diperbaiki, keeping comment for context)
from weaviate.auth import AuthApiKey

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Weaviate client with cloud credentials
# Perbaikan: Menghapus bagian yang duplikat dan menggunakan client yang benar (Sudah diperbaiki, keeping comment for context)
"""Initialize Weaviate client with cloud credentials"""
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="epu3qawrpkjx7m0zafxq.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key("UKIqKrRSlhI633pl9GQbQgqWqqBGhEOQtIac"),
)

# Setup embeddings
embeddings = OllamaEmbeddings(model="neural-chat")

# Setup vector stores for each domain
vectorstore_marketing = WeaviateVectorStore(
    client=client,
    index_name="Document",
    text_key="content",
    embedding=embeddings,
    attributes=["category", "subcategory", "source"],
)

vectorstore_operasional = WeaviateVectorStore(
    client=client,
    index_name="Document",
    text_key="content",
    embedding=embeddings,
    attributes=["category", "subcategory", "source"],
)

# Membuat retriever yang hanya mengambil dokumen dengan kategori "marketing"
marketing_filter = Filter.by_property("category").equal("marketing")

# Membuat retriever yang hanya mengambil dokumen dengan kategori "operasional"
operasional_filter = Filter.by_property("category").equal("operasional")

# Setup retrievers with a higher k value to ensure better document coverage
retriever_marketing = vectorstore_marketing.as_retriever(search_kwargs={"filters": marketing_filter})
retriever_operasional = vectorstore_operasional.as_retriever(search_kwargs={"filters": operasional_filter})

# LLM instance for all operations
llm = ChatOllama(model="neural-chat", temperature=0.1)

# Store conversation history
conversations = {}
conversation_memories = {}

# Define the state schema for LangGraph
class GraphState(TypedDict):
    question: str
    domain: str
    answer: str
    sources: List[Dict[str, str]]
    conversation_id: str

# Router node - Classify questions to appropriate domain
def route_domain(state: GraphState):
    question = state["question"]
    router_prompt = PromptTemplate.from_template("""
    Klasifikasikan pertanyaan berikut ke dalam salah satu domain:
    - "operasional" jika pertanyaan tentang gudang, logistik, pengiriman, inventaris, dsb.
    - "marketing" jika pertanyaan tentang promosi, kampanye, pelanggan, brand, dll.

    Pertanyaan: {question}
    Jawab hanya dengan satu kata: operasional atau marketing.
    """)
    chain = router_prompt | llm | (lambda x: x.content.strip().lower())
    domain = chain.invoke({"question": question})
    return {"domain": domain, "question": question}

# Function to check document relevance
def check_document_relevance(question, documents):
    if not documents:
        return False, []

    relevance_prompt = PromptTemplate.from_template("""
    Pertanyaan: {question}

    Dokumen yang tersedia:
    {documents}

    Apakah dokumen-dokumen tersebut mengandung informasi yang relevan untuk menjawab pertanyaan?
    Periksa baik-baik dan tentukan apakah dokumen tersebut BENAR-BENAR mengandung informasi yang diperlukan.

    Jawablah hanya dengan "ya" atau "tidak".
    """)

    # Prepare documents text for prompt
    docs_text = "\n\n".join([f"Dokumen {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])

    # Invoke LLM to check relevance
    chain = relevance_prompt | llm | (lambda x: x.content.strip().lower())
    result = chain.invoke({"question": question, "documents": docs_text})

    return result == "ya", documents

# Function to build RAG nodes for each domain
def build_rag_node(retriever, domain_name):
    def rag_node(state: GraphState):
        question = state["question"]
        conversation_id = state.get("conversation_id", "default")

        # Get conversation memory if it exists
        memory = conversation_memories.get(conversation_id,
                                        ConversationBufferMemory(
                                            memory_key="chat_history",
                                            return_messages=True,
                                            output_key='answer'
                                        ))

        # First, retrieve documents
        retrieved_docs = retriever.get_relevant_documents(question)

        # Check if documents are relevant
        is_relevant, documents = check_document_relevance(question, retrieved_docs)

        if not is_relevant:
            return {
                "answer": "Informasi tidak ditemukan di database.",
                "sources": [],
                "domain": domain_name
            }

        # Create a custom prompt that forces the model to use ONLY the retrieved documents
        qa_prompt = PromptTemplate.from_template("""
        Kamu adalah asisten AI yang hanya menjawab berdasarkan dokumen yang tersedia.

        Konteks:
        {context}

        Riwayat Percakapan:
        {chat_history}

        Pertanyaan: {question}

        Berikan jawaban HANYA berdasarkan informasi dari dokumen yang diberikan di atas.
        Jika informasi tidak ada dalam dokumen tersebut, jawab dengan "Informasi tidak ditemukan di database."
        Jangan menambahkan informasi atau pengetahuan dari sumber lain.
        """)

        # Create retrieval chain with custom prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )

        # Get response from the chain
        response = chain({"question": question})

        # Extract information
        answer = response["answer"]

        # If the answer contains anything indicating lack of information
        if ("tidak ditemukan" in answer.lower() or
            "tidak tersedia" in answer.lower() or
            "tidak ada informasi" in answer.lower() or
            "tidak ada dalam dokumen" in answer.lower()):
            answer = "Informasi tidak ditemukan di database."

        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                source_info = {
                    "content": doc.page_content[:150] + "...",
                    "category": doc.metadata.get("category", domain_name),
                    "subcategory": doc.metadata.get("subcategory", "general"),
                    "source": doc.metadata.get("source", "unknown")
                }
                sources.append(source_info)

        # Update conversation memory if needed
        if conversation_id not in conversation_memories:
            conversation_memories[conversation_id] = memory

        return {"answer": answer, "sources": sources, "domain": domain_name}

    return rag_node

# Build LangGraph for routing and executing domain-specific retrievals
def build_graph():
    # Create nodes
    operasional_node = build_rag_node(retriever_operasional, "operasional")
    marketing_node = build_rag_node(retriever_marketing, "marketing")

    # Build graph
    builder = StateGraph(GraphState)  # Pass the state schema to StateGraph
    builder.add_node("router", RunnableLambda(route_domain))
    builder.add_node("operasional", RunnableLambda(operasional_node))
    builder.add_node("marketing", RunnableLambda(marketing_node))

    builder.set_entry_point("router")
    builder.add_conditional_edges("router", lambda state: state["domain"], {
        "operasional": "operasional",
        "marketing": "marketing"
    })

    builder.add_edge("operasional", END)
    builder.add_edge("marketing", END)

    return builder.compile()

# Create graph instance
graph = build_graph()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    conversation_id = data.get('conversation_id', 'default')
    user_message = data.get('message', '')

    # Initialize conversation if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []

    # Add user message to conversation history
    conversations[conversation_id].append({"role": "user", "content": user_message})

    try:
        # Process with LangGraph
        output = graph.invoke({
            "question": user_message,
            "conversation_id": conversation_id
        })

        # Extract results
        assistant_message = output["answer"]
        sources = output.get("sources", [])
        domain = output.get("domain", "unknown")

        # Add assistant's message to conversation history
        conversations[conversation_id].append({"role": "assistant", "content": assistant_message})

        return jsonify({
            "message": assistant_message,
            "sources": sources,
            "domain": domain,
            "conversation_id": conversation_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    return jsonify(list(conversations.keys()))

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    if conversation_id in conversations:
        return jsonify(conversations[conversation_id])
    return jsonify({"error": "Conversation not found"}), 404

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    if conversation_id in conversations:
        del conversations[conversation_id]
        if conversation_id in conversation_memories:
            del conversation_memories[conversation_id]
        return jsonify({"success": True})
    return jsonify({"error": "Conversation not found"}), 404

@app.route('/api/categories', methods=['GET'])
def get_categories():
    return jsonify(["marketing", "operasional"])

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    category = request.form.get('category', 'general')
    subcategory = request.form.get('subcategory', 'general')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not os.path.exists("./data"):
        os.makedirs("./data")

    filepath = os.path.join("./data", file.filename)
    file.save(filepath)

    # Load and process the uploaded file
    try:
        loader = TextLoader(filepath)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_chunks = text_splitter.split_documents(documents)

        # Add metadata to each chunk
        for chunk in doc_chunks:
            chunk.metadata["category"] = category
            chunk.metadata["subcategory"] = subcategory
            chunk.metadata["source"] = file.filename

        # Choose the appropriate vectorstore based on category
        if category == "marketing":
            vectorstore = vectorstore_marketing
        elif category == "operasional":
            vectorstore = vectorstore_operasional
        else:
            # If category doesn't match our domains, default to a general vectorstore
            vectorstore = WeaviateVectorStore(  # Corrected class name
                client=client,
                index_name="Document",
                text_key="content",
                embedding=embeddings,
                attributes=["category", "subcategory", "source"]
            )

        # Add documents to vectorstore
        vectorstore.add_documents(doc_chunks)

        return jsonify({
            "success": True,
            "message": f"File uploaded and {len(doc_chunks)} chunks added to the {category} database"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    client.close()
