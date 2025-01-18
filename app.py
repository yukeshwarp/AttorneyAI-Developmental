import streamlit as st
#from utils.config import *
from utils.extractor import extractor
import requests
import json
from io import BytesIO
import logging
import fitz
import uuid

# Connect to Azure Redis
# redis_client = redis.StrictRedis(
#     host=redis_host, port=redis_port, password=redis_password, decode_responses=True
# )


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
        # st.write("hi2")
        Index, document = extractor(doc)
        st.write(Index)

        # # Create the document in memory
        # doc = Document()
        # for conflict_grade, results in comparison_results.items():
        #     count = len(results)
        #     doc.add_paragraph(f"{conflict_grade}: {count} entries")

        #     if results:
        #         doc.add_heading(conflict_grade, level=2)
        #         table = doc.add_table(rows=1, cols=5)
        #         header = table.rows[0].cells
        #         header[0].text = "Trademark Name and Class Number"
        #         header[1].text = "Trademark Status"
        #         header[2].text = "Serial/Registration Number"
        #         header[3].text = "Owner Name"
        #         header[4].text = "Design/Word"

        #         for result in results:
        #             row = table.add_row().cells
        #             row[0].text = (
        #                 f"{result['Trademark name']} (Class {result['Trademark class Number']})"
        #             )
        #             row[1].text = result["Trademark -_status"]
        #             row[2].text = (
        #                 f"{result['Trademark serial number']} / {result['Trademark registration number']}"
        #             )
        #             row[3].text = result["Trademark -_owner"]
        #             row[4].text = (
        #                 "Design" if result["Trademark design phrase"] else "Word"
        #             )

        # # Save the document to a BytesIO object
        # output = BytesIO()
        # doc.save(output)
        # output.seek(0)

        # # Streamlit app logic
        # st.title("Trademark Conflict Analysis")
        # st.write("Download the trademark conflict analysis document below.")

        # # Add a download button
        # st.download_button(
        #     label="Download Analysis Document",
        #     data=output,
        #     file_name="Trademark_Conflict_Analysis.docx",
        #     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # )
