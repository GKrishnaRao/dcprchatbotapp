import streamlit as st
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

# Configure page settings
st.set_page_config(page_title="Document Q&A System", layout="wide")


# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.collection = None
    st.session_state.ef = None




def process_query(query):
    try:
        st.session_state.processing = True

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Generate embeddings for query
        status_text.text("Generating Answer...")
        progress_bar.progress(20)
        query_embeddings = st.session_state.ef([query])

        # Get context from previous conversations
        context = " ".join([f"Q: {q} A: {a}" for q, a in st.session_state.chat_history[-3:]])

        progress_bar.progress(40)

        # Perform hybrid search
        hybrid_results = hybrid_search(
            st.session_state.collection,
            query_embeddings["dense"][0],
            query_embeddings["sparse"][[0]],
            sparse_weight=0.7,
            dense_weight=1.0,
        )
        progress_bar.progress(60)
        # Create prompt with context
        final_template = f"""
        You are an Legal expert Real Estate Agent assistant providing detailed and accurate information.

        Previous conversation context:
        {context}

        Current question: {query}

        Retrieved Information:
        {hybrid_results}

        Instructions:
        1. Use the conversation context to maintain continuity
        2. Reference previous discussions if relevant
        3. Provide a detailed response based on the retrieved information
        4. If the question is a follow-up, connect it with previous context
        """
        print("CALL API")
        # Get response from Groq

        progress_bar.progress(80)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": final_template},
                {"role": "user", "content": query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1024
        )
        print(response)
        # Extract the response
        progress_bar.progress(100)
        ai_response = response.choices[0].message.content
        return ai_response
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        return None
    finally:
        st.session_state.processing = False

def display_chat():
    for question, answer in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.write(question)

        # AI response
        with st.chat_message("assistant"):
            st.write(answer if answer is not None else "Generating response...")


# Initialize Groq client with error handling
def initialize_groq_client():
    try:
        groq_api_key = "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"
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
            uri="https://in03-4e569f605c32eab.serverless.gcp-us-west1.cloud.zilliz.com",
            token="99cd003de770782d436a049c87fb669188dc4424443531a325043d7f42859ca8c3d058b952d2e92d33677cf72b4931d12150c29d"
        )
        st.success("Successfully connected")
    except Exception as e:
        st.error(f"Failed to connect: {e}")
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

    if "user_input" not in st.session_state:
        st.session_state.input_value =""

    # Chat interface
    # st.subheader("Chat Interface")

    # Create a placeholder for chat history
    chat_placeholder = st.empty()

    #Display chat history
    # Function to display chat messages
     # Chat interface
    if st.session_state.initialized:
        # Display chat history
        display_chat()

        # Chat input
        user_input = st.chat_input("Type your message here...")

        if user_input and not st.session_state.processing:
            # Add user message to chat history
            st.session_state.chat_history.append((user_input, None))

            # Process query and update chat history with response
            response = process_query(user_input)
            st.session_state.chat_history[-1] = (user_input, response)

            # Rerun to update the display
            st.rerun()



if __name__ == "__main__":
    main()

