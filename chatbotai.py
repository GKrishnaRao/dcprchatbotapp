from flask import Flask, render_template, request, jsonify
import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    FieldSchema, CollectionSchema, DataType, 
    connections, Collection, AnnSearchRequest, WeightedRanker
)
from groq import Groq
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Initialize global variables
ef = None
collection = None
groq_client = None

def initialize_system():
    global ef, collection, groq_client
    
    # Initialize Groq client
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return False, "GROQ_API_KEY environment variable is not set"
        groq_client = Groq(api_key=groq_api_key)
    except Exception as e:
        return False, f"Failed to initialize Groq client: {str(e)}"

    # Initialize embedding function
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ef = BGEM3EmbeddingFunction(use_fp16=False, device=device)
    
    # Connect to Milvus
    try:
        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
    except Exception as e:
        return False, f"Failed to connect to Milvus: {str(e)}"

    # Initialize collection
    try:
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]
        schema = CollectionSchema(fields)
        
        collection = Collection(
            name="hybrid_demo",
            schema=schema,
            consistency_level="Eventually"
        )
        collection.load()
        return True, "System initialized successfully"
    except Exception as e:
        return False, f"Failed to initialize collection: {str(e)}"

# Initialize the system when the app starts
init_success, init_message = initialize_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if not init_success:
        return jsonify({'error': init_message}), 500
    
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Generate embeddings
        query_embeddings = ef([query])
        
        # Perform hybrid search
        hybrid_results = hybrid_search(
            collection,
            query_embeddings["dense"][0],
            query_embeddings["sparse"][[0]],
            sparse_weight=0.7,
            dense_weight=1.0
        )

        final_template = f"""
                You are an Legal expert Real Estate Agent assistant providing detailed and accurate information. The user has asked the following question:
                **Question:** {query}

                Below is relevant information retrieved from various sources:
                **Retrieved Information:**
                {hybrid_results}

                **Instruction:** Use only the query-relevant content from the retrieved information to answer the question. Focus on providing a detailed comprehensive, informative response based solely on the given data. If any conflicting details are present, prioritize the most reliable and consistent information.
                """
        
        # Get response from Groq
        chat_completion = groq_client.chat.completions.create(
               messages=[
                        {
                            "role": "user",
                            "content": final_template,
                        }
                    ],
                    model="llama-3.1-70b-versatile",
                    )

        # Extract the response
        ai_response = chat_completion.choices[0].message.content

        return jsonify({
            'results': hybrid_results,
            'answer': ai_response
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper functions (keep the same implementation)
def hybrid_search(col, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0, limit=5):
    try:
        dense_req = AnnSearchRequest(
            [query_dense_embedding], 
            "dense_vector", 
            {"metric_type": "IP", "params": {}}, 
            limit=limit
        )
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], 
            "sparse_vector", 
            {"metric_type": "IP", "params": {}}, 
            limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = col.hybrid_search(
            [sparse_req, dense_req], 
            rerank=rerank, 
            limit=limit, 
            output_fields=["text"]
        )[0]
        return [hit.get("text") for hit in res]
    except Exception as e:
        return None

if __name__ == '__main__':
    app.run(host="82.112.230.225", port=8000)
