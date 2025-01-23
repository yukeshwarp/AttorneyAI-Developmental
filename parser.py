import requests
import json
import fitz
from utils.config import *
import logging as log
import asyncio
from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Dict, Union, Optional
import random
from openai import AzureOpenAI
from utils.config import client
from models import replace_disallowed_words, TrademarkDetails

llm_headers = {"Content-Type": "application/json", "api-key": llm_api_key}

def extract_search_target(doc):
    first_page = doc[0]
    chunk = first_page.get_text()

    try:

        messages = [
            {
                "role": "system",
                "content": "You are a data extraction specialist proficient in parsing trademark documents.",
            },
            {
                "role": "user",
                "content": f"""
                Extract the following details from the content provided which has details about a trademark target:  

                "mark_searched", "classes_searched", "goods_&_services"
                
                Do not include any additional text or explanations.  

                Document chunk to extract from: 
                {chunk}
            """,
            },
        ]
        tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "extract_trademark_details",
                            "description": "Extracts trademark details from a provided document chunk.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "mark_searched": {"type": "string"},
                                    "classes_searched": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                    "goods_services": {"type": "string"},
                                },
                                "required": [
                                    "mark_searched",
                                    "classes_searched",
                                    "goods_services",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]

            
        response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools = tools, temperature=0
            )

        details = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return details  # Successfully completed, return the result

    except Exception as e:
        log.error("Error in search extraction", e)


def extractor(doc):
    extracted_pages = []  # Array to store extracted text from each relevant page
    page_numbers = []  # Array to store corresponding page numbers
    extracted_pages2 = []  # Array to store text from all pages (optional)
    flag_uspto = False  # Flag to indicate USPTO Summary Page interval
    flag_state = False  # Flag to indicate State Summary Page interval
    index = ""
    iteration = 0
    page = doc[0]
    rect = page.rect
    height = 50
    clip = fitz.Rect(0, height, rect.width, rect.height-height)
    text = page.get_text(clip=clip)
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
            log.info(text)
            index = f"""{index} \n  {text}"""
            page_numbers.append(page_num)
            


    def query_count(count_prompt):
        count_data = {
            "model": llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that counts trademarks.",
                },
                {
                    "role": "user",
                    "content": count_prompt,
                },
            ],
            "temperature": 0.0,
        }

        count_url = f"{azure_llm_endpoint}/openai/deployments/{llm_model}/chat/completions?api-version={llm_api_version}"
        response = requests.post(count_url, headers=llm_headers, json=count_data, timeout=120)
        response_content = response.json()
        return int(response_content["choices"][0]["message"]["content"].strip())

    # Step 2: Iterate until count stabilizes
    previous_count = -1
    bprevious_count = previous_count
    count_prompt = f"""
        You are tasked with counting the total number of trademarks listed in the provided index text. 
        Simply return the total count as an integer without any additional text or explanations.

        Input Text:
        {index}
    """
    current_count = query_count(count_prompt)

    while current_count != previous_count and bprevious_count >= current_count:
        bprevious_count = previous_count
        previous_count = current_count
        count_prompt = f"""
        You are tasked with counting the total number of trademarks in the provided index text. 
        Ensure the count is accurate and consistent with the previously given count of {previous_count}. 
        If the current count differs, confirm and provide only the updated total as an integer.
        
        Input Text:
        {index}
        """
        current_count = query_count(count_prompt)
        iteration = iteration + 1
        if iteration == 5:
            log.error("Not able to extract exact number of entries from index!")
            break

    prompt = f"""  
        You are tasked with extracting trademark names and their associated starting page numbers from the provided index text. Each trademark should be represented as a JSON object with:  
        
        - "name": The exact name of the trademark (string).  
        - "page-start": The page number where the trademark starts (integer).  
        
        Example Input:
        1. ARRID EXTRA DRY Registered 3 CHURCH & DWIGHT CO., INC. 73−716,876 15  
        2. ARRID EXTRA EXTRA DRY Registered 3 CHURCH & DWIGHT CO., INC. 78−446,679 18  
        3. EXTRA RICH FOR DRY, THIRSTY HAIR Cancelled 3 NAMASTE LABORATORIES, L.L.C. 77−847,568 21  
        4. GOOD LEAF Published 32, 33 DIAGEO NORTH AMERICA, INC. 90−829,139 89  
        5. SHEAR GENIUS Registered 35, 44 SHEAR GENIUS OF FORT MOHAVE LLC 537444 AZ 225  
        6. SHEAR GENIUS Registered 44 FABIO PAWLOS 1454759 NJ 226  
        7. SHEAR GENIUS Registered 44 SHEAR GENIUS LLC 44423600 ND 227  
        
        Example Output: 
        [  
            {{ "name": "ARRID EXTRA DRY", "page-start": 15 }},  
            {{ "name": "ARRID EXTRA EXTRA DRY", "page-start": 18 }},  
            {{ "name": "EXTRA RICH FOR DRY, THIRSTY HAIR", "page-start": 21 }},  
            {{ "name": "GOOD LEAF", "page-start": 89 }},  
            {{ "name": "SHEAR GENIUS", "page-start": 225 }},  
            {{ "name": "SHEAR GENIUS", "page-start": 226 }},  
            {{ "name": "SHEAR GENIUS", "page-start": 227 }}  
        ]  
        
        Instructions:
        
        1. Trademark Name Extraction:
           - The trademark name is located immediately before the words "Registered," "Published," or "Cancelled."  
           - Extract the trademark name exactly as it appears without alterations.  
        
        2. Page Number Extraction:
           - The start page number is the last number in each entry.  
           - If multiple page numbers are listed (e.g., "32, 33"), use the first one as the start page.  
        
        3. Multiple Occurrences:
           - If a trademark appears multiple times with different page numbers, create a separate JSON object for each occurrence.  
        
        4. Output Format:
           - Return a JSON array containing the extracted trademarks and page numbers.  
           - Ensure the output matches the format shown in the example output.  
           - Do not include any additional text or explanations.  

        Number of entries to be extracted: {current_count}
        Input Text:
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
    response_content = llm_response.json()  # Parse the JSON response
    response = response_content["choices"][0]["message"]["content"].strip().lower()

    # record = response
    # st_response = str(response)[7:-3]
    # record = json.loads(st_response)
    return response, ""


async def extract_trademark_details(document_chunk: str, tm_name, target):
    max_retries = 5  # Maximum number of retries
    base_delay = 1  # Base delay in seconds
    jitter = 0.5  # Maximum jitter to add to the delay
    
    for attempt in range(1, max_retries + 1):
        try:
    
            messages = [
                {
                    "role": "system",
                    "content": "You are a data extraction specialist proficient in parsing trademark documents.",
                },
                {
                        "role": "user",
                        "content": f"""
                        Extract the following details from the provided trademark document and present them in the exact format specified:  
        
                        - Trademark Name  
                        - Status  
                        - Serial Number  (Return null if not present)
                        - International Class Number (as a list of integers)
                        - Goods & Services (Goods and services are given after every international class, extract them intelligently as they may span over more than one page.)
                        - Owner  
                        - Filed Date (format: MMM DD, YYYY, e.g., Jun 14, 2024, Return null if not present)  
                        - Registration Number  (Return null if not present)
                        - Design phrase (Return null if not present)
        
                        Instructions:  
                        - Return the results in the following format, replacing the example data with the extracted information:
                        - Ensure the output matches this format precisely.  
                        - Do not include any additional text or explanations. 
                        - Only extract information that is present in the document.  
                        - If a field is not found, set its value to `null`.  
                        - Do not make up or infer any information that is not explicitly stated. 
                        - For the **design_phrase**, extract it only if it is explicitly labeled as "Design Phrase" or "Design Mark."  
                        - Ignore headers, footers, or repeated phrases that are not explicitly associated with the fields.
        
                        Document chunk of to extract from: 
                        Trademark name: {tm_name} 
                        {document_chunk}  
                    """,
                    },
            ]
            tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "extract_trademark_details",
                                "description": "Extracts trademark details from a provided document chunk.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "trademark_name": {"type": "string"},
                                        "status": {"type": "string"},
                                        "serial_number": {"type": "string"},
                                        "international_class_number": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                        },
                                        "goods_services": {"type": "string"},
                                        "owner": {"type": "string"},
                                        "filed_date": {"type": "string"},
                                        "registration_number": {"type": "string"},
                                        "design_phrase": {"type": ["string", "null"]},
                                    },
                                    "required": [
                                        "trademark_name",
                                        "status",
                                        "international_class_number",
                                        "goods_services",
                                        "owner",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ]

                
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o", messages=messages, tools = tools, temperature=0
                ),
            )
            
            if hasattr(response.choices[0].message, 'function_call'):  
                details = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                if details['design_phrase'] == target:
                    details['design_phrase'] = "null"
                return details  # Successfully completed, return the result  
            else:  
                log.error("No function_call in response")  
                return None
            #details = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            # if details:
            #     # Attempt to create a TrademarkDetails instance
            #     try:
            #         details = TrademarkDetails(
            #             trademark_name=details.get("-_trademark_name"),
            #             status=details.get("-_status"),
            #             serial_number=details.get("serial_number"),
            #             international_class_number=details.get(
            #                 "international_class_number"
            #             ),
            #             owner=details.get("owner"),
            #             goods_services=details.get("goods_services"),
            #             page_number=details.get(
            #                 "page_number", -1
            #             ),
            #             registration_number=details.get(
            #                 "registration_number"
            #             ),
            #             design_phrase=details.get("design_phrase", ""),
            #         )
            #     except ValidationError as e:
            #         log.error(f"Validation error {e}")
            # details["-_trademark_name"] = tm_name
    
            #return details  # Successfully completed, return the result

        except Exception as e:
            if attempt == max_retries:
                raise  # Raise the exception if we've reached the maximum retries
            else:
                delay = base_delay * (
                    2 ** (attempt - 1)
                )  # Exponential backoff
                delay_with_jitter = delay + random.uniform(0, jitter)
                print(
                    f"Attempt {attempt} failed error: {e}. Retrying in {delay_with_jitter:.2f} seconds..."
                )
                await asyncio.sleep(delay_with_jitter)

#https://medium.com/@maximilian.vogel/i-scanned-1000-prompts-so-you-dont-have-to-10-need-to-know-techniques-a77bcd074d97
