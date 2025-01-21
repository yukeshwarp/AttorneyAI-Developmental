import streamlit as st
from utils.config import *
from parser import extractor, extract_trademark_details
from models import replace_disallowed_words
import requests
import json
from io import BytesIO
import logging
import fitz
import uuid
import asyncio
import json


if "documents" not in st.session_state:
    st.session_state.documents = {}
if "removed_documents" not in st.session_state:
    st.session_state.removed_documents = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("AttorneyAI")

uploaded_files = st.file_uploader(
    "Upload files less than 400 pages",
    type=["pdf", "docx", "xlsx", "pptx"],
    accept_multiple_files=True,
    help="If your question is not answered properly or there's an error, consider uploading smaller documents or splitting larger ones.",
    label_visibility="collapsed",
)

combined_responses = ""
comparison_results = {
    "High": [],
    "Moderate": [],
    "Name-Match": [],
    "Low": [],
    "No-conflict": [],
}

if uploaded_files:
    new_files = []
    for uploaded_file in uploaded_files:
        if (
            uploaded_file.name
            not in [
                st.session_state.documents[doc_id]["name"]
                for doc_id in st.session_state.documents
            ]
            and uploaded_file.name not in st.session_state.removed_documents
        ):
            new_files.append(uploaded_file)

    for new_file in new_files:
        st.success(f"File Selected: {new_file.name}")
        pdf_bytes = new_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        target_search = extract_search_target(doc)
        st.write(target_search)
        Index, document = extractor(doc)
        if Index.startswith("```json"):
            Index = Index[7:]  # Remove the "```json" part
        if Index.endswith("```"):
            Index = Index[:-3]  # Remove the trailing "```"
        
        # Convert the remaining text to a Python object
        try:
            Index = json.loads(Index)
        except:
            print("Error in json parsing:")

        st.write(Index)
            
        async def parallel_extraction():
            tasks = []
            for i in range(len(Index)):
                start_page = int(Index[i]["page-start"]) - 1
                if i == len(Index) - 1:
                    end_page = start_page + 4
                else:
                    end_page = int(Index[i + 1]["page-start"]) - 1
    
                document_chunk = "\n".join(extracted_pages[start_page:end_page])
                tasks.append(
                    extract_trademark_details(document_chunk, Index[i]["name"])
                )
    
            return await asyncio.gather(*tasks)

        async def process_trademarks():
            extracted_details = await parallel_extraction()
            for details in extracted_details:
                st.write(details)

        asyncio.run(process_trademarks())
