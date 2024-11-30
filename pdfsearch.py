import streamlit as st

# Configure page settings
st.set_page_config(page_title="Document Q&A System", layout="wide")

import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    connections,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()


# Initialize Groq client with error handling
def initialize_groq_client():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY environment variable is not set")
            return None
            
        # Simple initialization without additional parameters
        client = Groq(
            api_key=groq_api_key
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return None

client = initialize_groq_client()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.collection = None
    st.session_state.ef = None

def initialize_system():# Use the initialized client
    client = initialize_groq_client()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ef = BGEM3EmbeddingFunction(use_fp16=False, device=device)
    dense_dim = ef.dim["dense"]
    print(dense_dim)
    try:
        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
        st.success("Successfully connected to Milvus")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None, None

    try:
        # Rest of your schema definition remains the same
        fields = [
                FieldSchema(
                name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
                ),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                ]
        schema = CollectionSchema(fields)
                
        collection_name = "hybrid_demo"
        col = Collection(
        name=collection_name,
        schema=schema,
        consistency_level="Eventually"  # Cloud recommended consistency level
        )
        print(col.schema)
        col.load()
        st.success("Successfully Load collection")
        return ef, col
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None, None    
    
    
def dense_search(col, query_dense_embedding, limit=5):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=5):
    try:
        search_params = {"metric_type": "IP", "params": {} }
        res = col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
            )[0]
        return [hit.get("text") for hit in res]
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None, None 

def hybrid_search(col,query_dense_embedding,query_sparse_embedding,sparse_weight=1.0,dense_weight=1.0,limit=5):
    try:
        dense_search_params = {"metric_type": "IP", "params": {}}
        
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = col.hybrid_search([sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"])[0]
        return [hit.get("text") for hit in res]
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None

def doc_text_formatting(ef, query, docs):
    tokenizer = ef.model.tokenizer
    query_tokens_ids = tokenizer.encode(query, return_offsets_mapping=True)
    query_tokens = tokenizer.convert_ids_to_tokens(query_tokens_ids)
    formatted_texts = []

    for doc in docs:
        ldx = 0
        landmarks = []
        encoding = tokenizer.encode_plus(doc, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])[1:-1]
        offsets = encoding["offset_mapping"][1:-1]
        for token, (start, end) in zip(tokens, offsets):
            if token in query_tokens:
                if len(landmarks) != 0 and start == landmarks[-1]:
                    landmarks[-1] = end
                else:
                    landmarks.append(start)
                    landmarks.append(end)
        close = False
        formatted_text = ""
        for i, c in enumerate(doc):
            if ldx == len(landmarks):
                pass
            elif i == landmarks[ldx]:
                if close:
                    formatted_text += "</span>"
                else:
                    formatted_text += "<span style='color:red'>"
                close = not close
                ldx = ldx + 1
            formatted_text += c
        if close is True:
            formatted_text += "</span>"
        formatted_texts.append(formatted_text)
    return formatted_texts

def format_result(text, max_length=200):
    """Format the result text for display"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def main():
    st.title("Chat With Real AI")
    
    # Initialize system if not already done
    if not st.session_state.initialized:
        st.session_state.ef, st.session_state.collection = initialize_system()
        if st.session_state.ef and st.session_state.collection:
            st.session_state.initialized = True
            st.success(f"Connected to collection with {st.session_state.collection.num_entities} documents")

    print("Collection schema details:")
    for field in st.session_state.collection.schema.fields:
        print(f"Field: {field.name}, Type: {field.dtype}, Params: {field.params}")
            
    # Query interface
    st.subheader("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Searching for answers..."):
            try:
                print("Started")
                # Generate embeddings for query
                query_embeddings = st.session_state.ef([query])
                
                # Perform searches with error handling
                try:
                    with st.spinner("Performing dense search..."):
                        dense_results = dense_search(st.session_state.collection, query_embeddings["dense"][0])
                    
                    with st.spinner("Performing sparse search..."):
                        sparse_results = sparse_search(st.session_state.collection, query_embeddings["sparse"][[0]])
                    
                    with st.spinner("Performing hybrid search..."):
                        hybrid_results = hybrid_search(
                            st.session_state.collection,
                            query_embeddings["dense"][0],
                            query_embeddings["sparse"][[0]],
                            sparse_weight=0.7,
                            dense_weight=1.0,
                        )
                except Exception as search_error:
                    st.error(f"Search operation failed: {str(search_error)}")
                
                
                # Display results
                st.subheader("Answer")
                
                final_template = f"""
                You are an Legal expert Real Estate Agent assistant providing detailed and accurate information. The user has asked the following question:
                **Question:** {query}

                Below is relevant information retrieved from various sources:
                **Retrieved Information:**
                {hybrid_results}

                **Instruction:** Use only the query-relevant content from the retrieved information to answer the question. Focus on providing a detailed comprehensive, informative response based solely on the given data. If any conflicting details are present, prioritize the most reliable and consistent information.
                """
                
                if client is None:
                    st.warning("Groq client not initialized. Please check your API key.")
                    return False
                try:
                    chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": final_template,
                        }
                    ],
                    model="llama-3.1-70b-versatile",
                    )
                    st.write(chat_completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Groq connection test failed: {str(e)}")
                    return False
            
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
                st.error("Please try again with a different query.")

if __name__ == "__main__":
    main()
