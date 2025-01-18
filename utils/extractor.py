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
    index = ""
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
            index = f"""{index} \n  {text}"""
            page_numbers.append(page_num)

    prompt = f"""
        Task Objective: You are tasked with extracting the names of trademarks and their associated start page from a given text from index of a document. Each entry should be structured in a JSON format that contains:

        "name": The name of the trademark (string).
        "page-start": The page number where the trademark starts (integer).
        
        Example Input:
        1. ARRID EXTRA DRY Registered 3 CHURCH & DWIGHT CO., INC. 73−716,876 15
        2. ARRID EXTRA EXTRA DRY Registered 3 CHURCH & DWIGHT CO., INC. 78−446,679 18
        3. EXTRA RICH FOR DRY, THIRSTY HAIR Cancelled 3 NAMASTE LABORATORIES, L.L.C. 77−847,568 21
        4. GOOD LEAF Published 32, 33 DIAGEO NORTH AMERICA, INC. 90−829,139 89
        5. SHEAR GENIUS Registered 35, 44 SHEAR GENIUS OF FORT MO HAVE LLC 537444 AZ 225
        6. SHEAR GENIUS Registered 44 FABIO PAWLOS 1454759 NJ 226
        7. SHEAR GENIUS Registered 44 SHEAR GENIUS LLC 44423600 ND 227
        
        In this example:
        "ARRID EXTRA DRY" starts at page 15.
        "ARRID EXTRA EXTRA DRY" starts at page 18.
        "EXTRA RICH FOR DRY, THIRSTY HAIR" starts at page 21.
        "GOOD LEAF" starts at page 89.
        "SHEAR GENIUS" starts at pages 225, 226, and 227.
        The goal is to extract all trademark names along with the page number they appear on. If a trademark appears multiple times, each entry should have a separate object with its corresponding page.
        
        Guidelines for Extraction:
        1. The trademark name is usually located before the word "Registered," "Published," or "Cancelled."
        2. The start page number is always the last number in the entry and typically follows the trademark's registration details.
        3. Ignore any other numbers or content unrelated to the trademark name and its start page number.
        4. If there are multiple page numbers listed (e.g., 32, 33), extract the first page number for the start of the trademark entry.
        5. Ensure that the extraction covers all trademarks and their respective page numbers accurately.
        
        Output Format: The output should be in the following JSON format:
        [
            {
                "name": "ARRID EXTRA DRY",
                "page-start": 15
            },
            {
                "name": "ARRID EXTRA EXTRA DRY",
                "page-start": 18
            },
            {
                "name": "EXTRA RICH FOR DRY, THIRSTY HAIR",
                "page-start": 21
            },
            {
                "name": "GOOD LEAF",
                "page-start": 89
            },
            {
                "name": "SHEAR GENIUS",
                "page-start": 225
            },
            {
                "name": "SHEAR GENIUS",
                "page-start": 226
            },
            {
                "name": "SHEAR GENIUS",
                "page-start": 227
            }
        ]
        Important Notes:
        
        Make sure to extract each trademark and its start page from the given text completely without skipping any entries.
        Handle multiple occurrences of the same trademark name with different page numbers by creating a separate entry for each occurrence.
        The input text will follow a similar structure to the example provided, so adapt the extraction logic accordingly.
        Extracted text from index:
        {index}

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

    # record = response
    # st_response = str(response)[7:-3]
    # record = json.loads(st_response)
    return response, ""
#https://medium.com/@maximilian.vogel/i-scanned-1000-prompts-so-you-dont-have-to-10-need-to-know-techniques-a77bcd074d97
