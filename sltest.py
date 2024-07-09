from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
import streamlit as st
import json
import tempfile
import shutil
import ollama
from langchain_community.document_loaders import UnstructuredEmailLoader
load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import boto3
from botocore.exceptions import ClientError

CHROMA_PATH = 'chroma_data'

PROMPT_TEMPLATE = '''
        Given the following text extracted from meeting minutes, generate a list of action items, their associated dates (if any) and the person/entity associated with the action item in JSON format. Each action item should be an object with the following properties:
        - "action": The specific task or action to be taken
        - "date": The due date or relevant date for the action (if mentioned), formatted as YYYY-MM-DD. If no date is specified, use null.
        - "entity": The person/group associated with the action item. If no person is associated, use null.

        Present the results as a JSON array of these objects. Ensure that each action item is clear, concise, and actionable. Ignore general discussion points or decisions that don't require specific actions.

        Meeting minutes text:

        {context}

        Please provide the JSON output of action items based on this text. If there are no action items in the meeting minutes, then return "None".
    '''

def main():
    st.title("MinutesAI")
    uploaded_file = st.file_uploader("Choose a meeting minutes file", type=["pdf", "eml"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            pdf_reader = PyPDFLoader(tmp_file_path)
            documents = pdf_reader.load()
        elif uploaded_file.type == "message/rfc822":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = UnstructuredEmailLoader(tmp_file_path)
            documents = loader.load()

        text = "\n\n".join(doc.page_content for doc in documents)

        if st.button("Extract Action Items"):
            with st.spinner("Extracting action items..."):
                action_items = extract_action_items(documents)

            st.subheader("Extracted Action Items:")
            try:
                action_items_json = json.loads(action_items)
                if action_items_json == "None":
                    st.write("No action items found.")
                else:
                    for item in action_items_json:
                        st.write(f"- Action: {item['action']}")
                        st.write(f"  Date: {item['date']}")
                        st.write("---")
                st.toast('Action Items Extracted!', icon='🎉')
                # send_email(action_items_json)
                st.toast('Email sent!', icon='🎉')
            except json.JSONDecodeError:
                st.write(action_items)
        
    
        st.subheader("Chat with the document")
        user_question = st.text_input("Ask a question about the meeting")
        
        if user_question:
            with st.spinner("Generating response..."):
                response = ollama.generate(
                    model="mistral",
                    prompt=f"Based on the following document, please answer the question: {user_question}\n\nDocument content: {text}"
                )
            
            st.write("Answer:", response['response'])
    

def extract_action_items(documents):
    text = "\n\n".join(doc.page_content for doc in documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=text)

    output = ollama.generate(
        model="mistral",
        prompt=prompt
    )

    return output['response']

def send_email(action_items):
    sns_client = boto3.client('sns')
    topic_arn = "arn:aws:sns:us-east-1:646474602593:MeetingMinutesAi"

    # Create email HTML
    text = "Action Items from Meeting\n\n"
    
    for item in action_items:
        if item['date'] is None:
            text += f"- {item['action']}\n"
        else:
            text += f"- {item['action']} on {item['date']}\n"
    
    # Prepare the email message
    subject = "Meeting Action Items"
    body_text = "Please view this email in an HTML-compatible email viewer."

    try:
        response = response = sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=text
        )
        
        print(response)
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")

if __name__ == "__main__":
    main()