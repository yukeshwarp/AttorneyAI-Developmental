import requests
import json
import fitz
from utils.config import *

llm_headers = {"Content-Type": "application/json", "api-key": llm_api_key}

def replace_disallowed_words(text):
    disallowed_words = {
        "sexual": "xxxxxx",
        "sex": "xxx",
        "hate": "xxxx",
    }
    for word, replacement in disallowed_words.items():
        text = text.replace(word, replacement)
    # Ensure single paragraph output
    text = " ".join(text.split())
    return text

def extractor(doc):
    
    page = doc[0]
    rect = page.rect
    height = 50
    clip = fitz.Rect(0, height, rect.width, rect.height - height)
    extracted_pages = []  # Array to store extracted text from each relevant page
    page_numbers = []  # Array to store corresponding page numbers
    extracted_pages2 = []  # Array to store text from all pages (optional)
    flag_uspto = False  # Flag to indicate USPTO Summary Page interval
    flag_state = False  # Flag to indicate State Summary Page interval

    for page_num, page in enumerate(doc, start=1):
        # Extract text with optional clipping
        text = page.get_text(clip=clip)
        text = replace_disallowed_words(text)
        extracted_pages2.append(text)

        # Check for interval boundaries
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if "USPTO Summary Page" in text:
            flag_uspto = True
        elif "ANALYST REVIEW −USPTO REPORT" in text:
            flag_uspto = False

        # if "State Summary Page" in text:
        #     flag_state = True
        # elif "ANALYST REVIEW −STATE REPORT" in text:
        #     flag_state = False

        # Store relevant text and page numbers for both intervals
        if flag_uspto or flag_state:
            extracted_pages.append(text)
            page_numbers.append(page_num)

    for extracted_text in extracted_pages:
        # st.write(extracted_text)
        prompt = f"""
            The task is to extract the name and associated page ranges in a structured JSON array format with each entry containing:
            - "name": The name of the entity (string).
            - "page-start": The first page number where the entity appears (integer)

            The data will be as below:
            ---
            # Example: 
            1. ARRID EXTRA DRY Registered 3 CHURCH & DWIGHT CO., INC. 73−716,876 15

            2. ARRID EXTRA EXTRA DRY Registered 3 CHURCH & DWIGHT CO., INC. 78−446,679 18

            3. EXTRA RICH FOR DRY, THIRSTY HAIR Cancelled 3 NAMASTE LABORATORIES, L.L.C. 77−847,568 21
            
            4. GOOD LEAF Published 32, 33 DIAGEO NORTH AMERICA, INC. 90−829,139 89
            
            5. SHEAR GENIUS Registered 35, 44 SHEAR GENIUS OF FORT MO HAVE LLC 537444 AZ 225
            
            6. SHEAR GENIUS Registered 44 FABIO PAWLOS 1454759 NJ 226
            
            7. SHEAR GENIUS Registered 44 SHEAR GENIUS LLC 44423600 ND 227
            
            It means that "ARRID EXTRA DRY" starts at page 15, "ARRID EXTRA EXTRA DRY" at page 16, "EXTRA RICH FOR DRY, THIRSTY HAIR" at page 21, "GOOD LEAF" at page 89, "SHEAR GENIUS" starts at page 225, "SHEAR GENIUS" at page 226, "SHEAR GENIUS" at page 227 and so on like that.
            ---
            
            Intelligently extract all entries of trademark name and their start page completely without leaving any entry from the given extracted text.
            The following text data is extracted from a document:
            ---
            
            {extracted_text} 
            
            ---
            Output format:
        """

        data = {
            "model": llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts details.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
        }

        url = f"{azure_llm_endpoint}/openai/deployments/{llm_model}/chat/completions?api-version={llm_api_version}"
        llm_response = requests.post(
            url, headers=llm_headers, json=data, timeout=120
        )

        # Extract the response content
        response = (
            llm_response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        record = response
        st_response = str(response)[7:-3]
        record = json.loads(st_response)
        return record, ""
#https://medium.com/@maximilian.vogel/i-scanned-1000-prompts-so-you-dont-have-to-10-need-to-know-techniques-a77bcd074d97
