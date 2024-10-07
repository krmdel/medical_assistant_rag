import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm import tqdm
from typing import Dict, List
from langchain.embeddings import SentenceTransformerEmbeddings

def ingest_data(data_path='../data/data_metadata_small.csv', index_name='medical-questions'):
    # Load the data
    df = pd.read_csv(data_path)
    documents = df.to_dict(orient='records')

    # Initialize Elasticsearch client
    es_url = 'http://localhost:9200'  # Adjust the URL if your Elasticsearch instance is hosted elsewhere
    es_client = Elasticsearch(es_url)

    # Define the index settings with mappings
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "medical_department": {"type": "keyword"},
                "condition_type": {"type": "keyword"},
                "patient_demographics": {"type": "text"},
                "common_symptoms": {"type": "text"},
                "treatment_or_management": {"type": "text"},
                "severity": {"type": "keyword"},
                "question_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "answer_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "question_answer_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
            }
        }
    }

    # Delete the index if it exists and create a new one
    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)

    # Initialize the embedding model
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    # Index the documents into Elasticsearch
    for doc in tqdm(documents):
        question = doc.get('question', 'No question provided')
        answer = doc.get('answer', 'No answer provided')
        qa_combined = question + ' ' + answer

        doc['question'] = question
        doc['answer'] = answer
        doc['question_vector'] = model.encode(question).tolist()
        doc['answer_vector'] = model.encode(answer).tolist()
        doc['question_answer_vector'] = model.encode(qa_combined).tolist()

        # Use the document's 'id' field as the Elasticsearch document ID
        es_client.index(index=index_name, id=doc['id'], document=doc)

def hybrid_query_rrf(search_query: str, index_name='medical-questions') -> List[Dict]:
    # Initialize the embedding model and Elasticsearch client
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    es_url = 'http://localhost:9200'  # Adjust if your Elasticsearch is hosted elsewhere
    es_client = Elasticsearch(es_url)

    best_boost_params = {
        'question': 1.62,
        'answer': 1.70,
        'medical_department': 1.67,
        'condition_type': 0.97,
        'patient_demographics': 0.64,
        'common_symptoms': 1.75,
        'treatment_or_management': 0.27,
        'severity': 1.85
    }

    vector = embeddings.embed_query(search_query)
    k = 60  # RRF parameter

    # Define the KNN query
    knn_query = {
        "field": "question_answer_vector",
        "query_vector": vector,
        "k": 10,
        "num_candidates": 10000,
        "boost": 0.5,
    }

    # Build the fields with boosts
    fields_with_boosts = [f"{field}^{boost}" for field, boost in best_boost_params.items()]

    # Define the keyword query with boosts
    keyword_query = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": search_query,
                    "fields": fields_with_boosts,
                    "type": "best_fields",
                }
            },
        }
    }

    # Specify the fields to retrieve
    source_fields = [
        "id", "question", "answer", "medical_department", "condition_type",
        "patient_demographics", "common_symptoms", "treatment_or_management", "severity"
    ]

    # Execute KNN search
    knn_response = es_client.search(
        index=index_name,
        body={
            "knn": knn_query,
            "size": 10,
            "_source": source_fields
        }
    )
    knn_results = knn_response['hits']['hits']

    # Execute keyword search
    keyword_response = es_client.search(
        index=index_name,
        body={
            "query": keyword_query,
            "size": 10,
            "_source": source_fields
        }
    )
    keyword_results = keyword_response['hits']['hits']

    # Function to compute RRF score
    def compute_rrf(rank, k=60):
        return 1.0 / (k + rank)

    # Calculate RRF scores
    rrf_scores = {}
    # From KNN results
    for rank, hit in enumerate(knn_results):
        doc_id = hit['_id']
        score = compute_rrf(rank + 1, k)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score

    # From keyword results
    for rank, hit in enumerate(keyword_results):
        doc_id = hit['_id']
        score = compute_rrf(rank + 1, k)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score

    # Sort documents by RRF score in descending order
    reranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Retrieve the top documents
    final_results = []
    for doc_id, score in reranked_docs[:5]:
        # Search for the document in the combined results to avoid extra GET requests
        doc = next((hit for hit in (knn_results + keyword_results) if hit['_id'] == doc_id), None)
        if doc:
            source = doc['_source']
            final_results.append(source)

    return final_results
