import streamlit as st
import os
from google.cloud import storage

# Add this line here
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from dotenv import load_dotenv
import tempfile
import requests
import io
from langchain_community.document_loaders import PDFPlumberLoader
from milvus_model.hybrid import BGEM3EmbeddingFunction
import torch

from pymilvus import (
    FieldSchema,
    utility,
    CollectionSchema,
    DataType,
    connections,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
import time
from pymilvus import utility

# Load environment variables
load_dotenv()

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


def initialize_gcs_client():
    """Initialize Google Cloud Storage client"""
    try:
        # Create storage client
        # Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
        storage_client = storage.Client()
        return storage_client
    except Exception as e:
        st.error(f"Failed to initialize GCS client: {str(e)}")
        return None

def upload_to_gcs(storage_client, bucket_name, source_file, destination_blob_name):
    """Upload file to Google Cloud Storage"""
    if storage_client is None:
        return None

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Upload the file
        blob.upload_from_file(source_file)

        # Make the blob publicly accessible
        blob.make_public()

        return blob.public_url

    except Exception as e:
        st.error(f"Failed to upload file: {str(e)}")
        return None


def load_pdf_from_url(url):
    # Send GET request to download the PDF
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Download PDF content
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Use PDFPlumberLoader with the temporary file path
        loader = PDFPlumberLoader(temp_file_path)
        docs = loader.load()

        print("Number of pages in the PDF:", len(docs))

        # Extract text from each page for embedding
        doc_texts = [page.page_content for page in docs]

        # Clean up the temporary file
        import os
        os.unlink(temp_file_path)

        return docs, doc_texts

    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None, None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None, None


def insert_batch(collection, texts, sparse_vectors, dense_vectors, batch_size=50):
    total_inserted = 0

    try:
        # For efficiency, we insert batch_size records at a time
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))

            # Prepare the current batch
            batch_texts = texts[i:batch_end]
            batch_sparse = sparse_vectors[i:batch_end]
            batch_dense = dense_vectors[i:batch_end]

            # Create entities list for insertion
            entities = [
                batch_texts,      # Text content
                batch_sparse,     # Sparse vector embeddings
                batch_dense,      # Dense vector embeddings
            ]

            # Insert with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    insert_result = collection.insert(entities)
                    total_inserted += len(batch_texts)

                    # Ensure data is persisted
                    collection.flush()

                    print(f"Successfully inserted batch {i//batch_size + 1}, "
                          f"Records {i} to {batch_end}")
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to insert batch after {max_retries} attempts: {e}")
                        raise
                    print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff

    except Exception as e:
        print(f"Error during insertion: {e}")
        raise

    return total_inserted



# Add this function after your existing functions
def generate_questions_answers(client, doc_texts):
    """Generate 10 questions and answers from the document content"""

    # Combine all document texts
    combined_text = " ".join(doc_texts)

    prompt = f"""
    Based on the following document content, generate 10 relevant questions and their detailed answers.
    The questions should cover different aspects and important points from the document.

    Document Content:
    {combined_text}

    Please generate 10 questions and answers in this format:
    Q1: [Question]
    A1: [Detailed Answer]

    Q2: [Question]
    A2: [Detailed Answer]

    And so on...
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating questions: {str(e)}"




def generate_mcq_questions(client, doc_texts):
    """Generate multiple choice questions with 4 options and answers from the document content"""

    # Combine all document texts
    combined_text = " ".join(doc_texts)

    prompt = f"""
    Based on the following document content, generate 10 multiple choice questions (MCQs).
    Each MCQ should have 4 options (A, B, C, D) and clearly marked correct answer.
    The questions should test understanding of key concepts from the document.

    Document Content:
    {combined_text}

    Please generate the MCQs in this format:

    Q1: [Question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [A/B/C/D]
    Explanation: [Brief explanation why this is the correct answer]

    Q2: [Question]
    ...and so on for 10 questions.

    Guidelines:
    - Questions should be clear and unambiguous
    - All 4 options should be plausible
    - Only one option should be correct
    - Options should be of similar length
    - Include a mix of difficulty levels
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating MCQs: {str(e)}"

def display_mcq_quiz(mcq_content):
    """Display MCQs in an interactive quiz format"""
    # Initialize session state for answers and submitted state if not exists
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = {}
    if 'explanations' not in st.session_state:
        st.session_state.explanations = {}

    # Split content into individual questions
    questions = mcq_content.split("\n\n")

    # Process questions and store answers/explanations in session state
    for i, question_block in enumerate(questions, 1):
        if not question_block.strip():
            continue

        lines = question_block.split("\n")
        if not lines or not lines[0].startswith("Q"):
            continue

        question = lines[0].split(": ", 1)[1] if ": " in lines[0] else lines[0]
        options = [line for line in lines if line.startswith(("A)", "B)", "C)", "D)"))]

        # Store correct answers and explanations in session state
        for line in lines:
            if line.startswith("Correct Answer:"):
                st.session_state.correct_answers[f"Q{i}"] = line.split(": ")[1].strip()
            elif line.startswith("Explanation:"):
                st.session_state.explanations[f"Q{i}"] = line.split(": ")[1].strip()

        # Display question and options
        st.write(f"\n**Question {i}:** {question}")

        # Initialize answer for this question if not exists
        key = f"Q{i}"
        if key not in st.session_state.user_answers:
            st.session_state.user_answers[key] = None

        # Display radio buttons with current selection
        current_index = 0
        if st.session_state.user_answers[key] in ["A", "B", "C", "D"]:
            current_index = ["A", "B", "C", "D"].index(st.session_state.user_answers[key])

        answer = st.radio(
            "Select your answer:",
            ["A", "B", "C", "D"],
            key=f"q{i}",
            index=current_index,
            label_visibility="collapsed"
        )
        st.session_state.user_answers[key] = answer

        # Display options
        for option in options:
            st.write(option)

        st.write("---")

    # Submit button and results
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("Submit Quiz")

    if submit or st.session_state.quiz_submitted:
        st.session_state.quiz_submitted = True
        score = 0
        st.write("\n### Quiz Results")

        for q_num in st.session_state.user_answers:
            user_ans = st.session_state.user_answers[q_num]
            correct_ans = st.session_state.correct_answers.get(q_num)

            if user_ans == correct_ans:
                score += 1
                result = "✅ Correct"
            else:
                result = "❌ Incorrect"

            st.write(f"\n**{q_num}:** {result}")
            st.write(f"Your answer: {user_ans}")
            st.write(f"Correct answer: {correct_ans}")
            st.write(f"Explanation: {st.session_state.explanations.get(q_num, 'No explanation available')}")

        # Display final score
        st.write(f"\n## Final Score: {score}/{len(st.session_state.user_answers)}")
        percentage = (score / len(st.session_state.user_answers)) * 100
        st.write(f"Percentage: {percentage:.2f}%")




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

        collection_name = "bhagavad_gita"
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



    st.subheader("Uploaded a Documents & Ask a Questions")

    # Initialize session states if not already done
    if 'qa_content' not in st.session_state:
        st.session_state.qa_content = None
    if 'mcq_content' not in st.session_state:
        st.session_state.mcq_content = None
    if 'doc_texts' not in st.session_state:
        st.session_state.doc_texts = None


     # Get bucket name from environment variable
    BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

    if not BUCKET_NAME:
        st.error("GCS_BUCKET_NAME environment variable is not set")
        return

    # Initialize GCS client
    storage_client = initialize_gcs_client()
    if storage_client is None:
        st.error("Failed to initialize Google Cloud Storage client")
        return

      # File upload widget
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }

        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"{key}: {value}")

        # Upload button
        if st.button("Upload to Cloud Storage"):
            with st.spinner("Uploading file..."):
                try:
                    # Create a unique filename using timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    destination_blob_name = f"uploads/{timestamp}_{uploaded_file.name}"

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        # Write to temporary file
                        tmp_file.write(uploaded_file.getvalue())
                        # Reset file pointer
                        tmp_file.seek(0)

                        # Upload to GCS
                        public_url = upload_to_gcs(
                            storage_client,
                            BUCKET_NAME,
                            tmp_file,
                            destination_blob_name
                        )

                    if public_url:
                        st.success("File successfully uploaded!")
                        st.write("File path in storage:", destination_blob_name)

                        # Create an expander for the file URL
                        with st.expander("File Access Information"):
                            st.write("Your file has been uploaded and is now available in Google Cloud Storage.")
                            st.write("Storage path:", destination_blob_name)
                            st.write("Public URL:", public_url)

                        # Usage example:
                        pdf_url = public_url
                        docs, doc_texts = load_pdf_from_url(pdf_url)

                        if docs:
                            # Continue with your processing
                            # print("Document texts:", doc_texts)
                            device = "mps" if torch.backends.mps.is_available() else "cpu"
                            ef = BGEM3EmbeddingFunction(use_fp16=False, device=device)
                            dense_dim = ef.dim["dense"]

                            # Generate embeddings using BGE-M3 model
                            docs_embeddings = ef(doc_texts)
                            # Zilliz Cloud connection parameters
                            uri = "https://in03-4e569f605c32eab.serverless.gcp-us-west1.cloud.zilliz.com"  # e.g., "https://xxx.zillizcloud.com:443"
                            token ="99cd003de770782d436a049c87fb669188dc4424443531a325043d7f42859ca8c3d058b952d2e92d33677cf72b4931d12150c29d"
                            # Connect to Zilliz Cloud
                            connections.connect(
                                alias="default",  # Connection alias, default is 'default'
                                uri=uri,
                                token=token,
                                secure=True
                            )

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
                            # Create collection (drop the old one if exists)
                            col_name = "bhagavad_gita"
                            if utility.has_collection(col_name):
                                Collection(col_name).drop()

                            # Create collection with the same schema but with cloud-optimized settings
                            col = Collection(
                                name=col_name,
                                schema=schema,
                                consistency_level="Eventually"  # Cloud recommended consistency level
                            )

                            # Create indices for hybrid search - optimized for cloud deployment
                            sparse_index = {
                                "index_type": "SPARSE_INVERTED_INDEX",
                                "metric_type": "IP",
                                "params": {}  # Cloud default parameters
                            }
                            col.create_index(
                                field_name="sparse_vector",
                                index_params=sparse_index
                            )

                            dense_index = {
                                "index_type": "IVF_FLAT",    # Changed from AUTOINDEX for better cloud performance
                                "metric_type": "IP",
                                "params": {
                                    "nlist": 1024            # Number of clusters, adjust based on your data size
                                }
                            }
                            col.create_index(
                                field_name="dense_vector",
                                index_params=dense_index
                            )

                            # Load collection into memory
                            col.load()

                            # Extract the raw text from Document objects
                            doc_texts = [doc.page_content for doc in docs]

                            # Perform the insertion
                            try:
                                total_inserted = insert_batch(
                                    collection=col,
                                    texts=doc_texts,
                                    sparse_vectors=docs_embeddings["sparse"],
                                    dense_vectors=docs_embeddings["dense"],
                                    batch_size=50
                                )

                                # Verify insertion
                                print(f"Total entities inserted: {total_inserted}")
                                print(f"Collection entity count: {col.num_entities}")

                                # Optional: Create an alias for the collection
                                #utility.create_alias(
                                    #collection_name=col.name,
                                    #alias="latest_docs"
                                #)


                                # Modify your main code where the vector insertion is successful
                                # Add this after the successful insertion confirmation:

                                st.success(f"Successfully inserted {total_inserted} documents into the database")

                                # Store doc_texts in session state
                                st.session_state.doc_texts = doc_texts

                                # Generate Q&A content
                                with st.spinner("Generating questions and answers from the document..."):
                                    st.session_state.qa_content = generate_questions_answers(client, doc_texts)

                                # Generate MCQ content
                                with st.spinner("Generating multiple choice questions from the document..."):
                                    st.session_state.mcq_content = generate_mcq_questions(client, doc_texts)


                            except Exception as e:
                                print(f"Insertion process failed: {e}")

                            pass

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


    # Always display Q&A and MCQ sections if content exists (move this outside the file upload condition)
    if st.session_state.qa_content:
        with st.expander("📚 Generated Questions and Answers", expanded=True):
            st.markdown(st.session_state.qa_content)
            qa_bytes = st.session_state.qa_content.encode()
            st.download_button(
                label="Download Q&A",
                data=qa_bytes,
                file_name="document_questions_answers.txt",
                mime="text/plain"
            )

    if st.session_state.mcq_content:
        with st.expander("📝 Multiple Choice Quiz", expanded=True):
            display_mcq_quiz(st.session_state.mcq_content)
            mcq_bytes = st.session_state.mcq_content.encode()
            st.download_button(
                label="Download MCQs",
                data=mcq_bytes,
                file_name="document_mcqs.txt",
                mime="text/plain"
            )

    # Add this near the MCQ section
    if st.session_state.mcq_content:
        if st.button("Reset Quiz"):
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False
            st.experimental_rerun()

    # Initialize system if not already done
    if not st.session_state.initialized:
        st.session_state.ef, st.session_state.collection = initialize_system()
        if st.session_state.ef and st.session_state.collection:
            st.session_state.initialized = True
            #st.success(f"Connected to collection with {st.session_state.collection.num_entities} documents")

    print("Collection schema details:")
    for field in st.session_state.collection.schema.fields:
        print(f"Field: {field.name}, Type: {field.dtype}, Params: {field.params}")

    # Query interface
    st.subheader("Ask me anything.")
    query = st.text_input("Ask question:")

    if query:
        with st.spinner("Searching for answers..."):
            try:
                # Generate embeddings for query
                query_embeddings = st.session_state.ef([query])

                 # Display results
                st.subheader("Answer")

                # Perform searches with error handling
                try:
                    with st.spinner("Fetching answer..."):
                        dense_results = dense_search(st.session_state.collection, query_embeddings["dense"][0])

                    with st.spinner("Fetching answer..."):
                        sparse_results = sparse_search(st.session_state.collection, query_embeddings["sparse"][[0]])

                    with st.spinner("Fetching answer..."):
                        hybrid_results = hybrid_search(
                            st.session_state.collection,
                            query_embeddings["dense"][0],
                            query_embeddings["sparse"][[0]],
                            sparse_weight=0.7,
                            dense_weight=1.0,
                        )
                except Exception as search_error:
                    st.error(f"Search operation failed: {str(search_error)}")


                #final_template = f"""
                    #You are a knowledgeable spiritual guide specializing in the Bhagavad Gita, providing detailed and accurate information about its teachings, verses, and philosophical concepts. The user has asked the following question:
                    #**Question:** {query}

                    #Below is relevant information retrieved from various sources:
                    #**Retrieved Information:**
                    #{hybrid_results}

                    #**Instruction:** Use only the query-relevant content from the retrieved information to answer the question. Provide a comprehensive explanation based on the Bhagavad Gita's teachings. Include relevant verse references when applicable. Focus on delivering a clear, spiritually enriching response that stays true to the authentic teachings of the Gita. If multiple interpretations exist, present the most widely accepted understanding while acknowledging other perspectives if relevant.
                    #"""
                final_template = f"""
                    You are an expert in the content of the provided PDF document, equipped with the ability to answer questions based on the information contained within it. The user has asked the following question:
                    **Question:** {query}

                    Below is the relevant content extracted from the PDF:
                    **Retrieved Information:**
                    {hybrid_results}

                    **Instruction:** Use only the content from the PDF document to answer the question. Provide a detailed and accurate response that directly relates to the information retrieved. If no relevant information is found, respond with a polite message like: "Sorry, no relevant information found in the document to answer your question." Ensure your response stays focused on the content from the document, and avoid introducing information that is not derived from it. If multiple answers are possible based on the document, try to provide the most relevant or comprehensive explanation.
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
                    model="llama-3.3-70b-versatile",
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



