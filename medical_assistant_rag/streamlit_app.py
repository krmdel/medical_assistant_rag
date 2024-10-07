import streamlit as st
import uuid
from rag import rag
import db
import csv
import boto3
from io import StringIO
from db import get_conversation_data
from dotenv import load_dotenv
import os
from datetime import datetime
import logging


load_dotenv('/home/ubuntu/medical_assistant_rag/.env')

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

st.title("Medical Assistant Q&A")

question = st.text_input("Ask a medical question:")

if st.button("Submit"):
    if question:
        conversation_id = str(uuid.uuid4())
        
        answer_data = rag(question)
        
        st.write("Answer:", answer_data['answer'])
        
        db.save_conversation(
            conversation_id=conversation_id,
            question=question,
            answer_data=answer_data
        )
        
        st.session_state['conversation_id'] = conversation_id
    else:
        st.warning("Please enter a question.")

if 'conversation_id' in st.session_state:
    st.write("Was this answer helpful?")
    col1, col2 = st.columns(2)
    
    if col1.button('👍'):
        db.save_feedback(
            conversation_id=st.session_state['conversation_id'],
            feedback=1
        )
        st.success("Thank you for your feedback!")
        del st.session_state['conversation_id']

    if col2.button('👎'):
        db.save_feedback(
            conversation_id=st.session_state['conversation_id'],
            feedback=-1
        )
        st.success("Thank you for your feedback!")
        del st.session_state['conversation_id']

# Set up logging
logging.basicConfig(level=logging.INFO)

def upload_csv_to_s3(data, bucket_name, file_name):
    try:
        # Create a CSV file in memory
        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)
        
        # Write CSV headers
        csv_writer.writerow(['ID', 'Question', 'Answer', 'Feedback'])
        
        # Write conversation data to CSV
        for row in data:
            csv_writer.writerow([
                str(row.get('id', '')), 
                str(row.get('question', '')), 
                str(row.get('answer', '')), 
                str(row.get('feedback', ''))
            ])

        # Debug: Print bucket name and file name
        logging.info(f"Uploading to bucket: {bucket_name}, file: {file_name}")

        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())
        logging.info(f"File uploaded to S3: {bucket_name}/{file_name}")
    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")

def generate_csv_file_name():
    return f'medical_assistant/conversations_feedback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

st.title('Medical Assistant with RAG')

if st.button('Export to CSV and Upload to S3'):
    # Get conversation data from the database
    conversation_data = get_conversation_data()

    CSV_FILE_NAME = generate_csv_file_name()
    
    # Upload CSV to S3
    upload_csv_to_s3(conversation_data, S3_BUCKET_NAME, CSV_FILE_NAME)
    
    st.success(f'CSV file uploaded to S3 bucket "{S3_BUCKET_NAME}" in "medical_assistant/" as "{CSV_FILE_NAME}"')