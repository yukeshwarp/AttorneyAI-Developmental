import requests
import json
import fitz
from utils.config import *
import logging as log
import asyncio
from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Dict, Union, Optional
from typing import List, Dict, Union
import random
from openai import AzureOpenAI

llm_headers = {"Content-Type": "application/json", "api-key": llm_api_key}

class TrademarkDetails(BaseModel):
    trademark_name: str = Field(..., description="The name of the trademark.")
    status: str = Field(..., description="The status of the trademark (e.g., Registered, Cancelled).")
    serial_number: str = Field(..., description="The serial number of the trademark.")
    international_class_number: List[int] = Field(
        ..., description="A list of international class numbers associated with the trademark."
    )
    goods_services: str = Field(..., description="The goods and services associated with the trademark.")
    owner: str = Field(..., description="The owner of the trademark.")
    filed_date: str = Field(
        ..., 
        description="The filed date of the trademark in MMM DD, YYYY format.",
        regex=r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{2}, \d{4}$",
    )
    registration_number: str = Field(..., description="The registration number of the trademark.")
    design_phrase: str = Field(..., description="The design phrase of the trademark.")

@validator("filed_date")
def validate_filed_date(cls, value):
    # Validate date format
    from datetime import datetime

    try:
        datetime.strptime(value, "%b %d, %Y")
    except ValueError:
        raise ValueError(
            f"Invalid filed date format: {value}. Expected format is MMM DD, YYYY."
        )
    return value


@validator("international_class_number", pre=True)
def validate_class_numbers(cls, value):
    if isinstance(value, str):
        # Convert string to list of integers
        try:
            return [int(num.strip()) for num in value.split(",")]
        except ValueError:
            raise ValueError(
                "Invalid international class numbers. Expected a list of integers."
            )
    return value


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
    extracted_pages = []  # Array to store extracted text from each relevant page
    page_numbers = []  # Array to store corresponding page numbers
    extracted_pages2 = []  # Array to store text from all pages (optional)
    flag_uspto = False  # Flag to indicate USPTO Summary Page interval
    flag_state = False  # Flag to indicate State Summary Page interval
    index = ""
    for page_num, page in enumerate(doc, start=1):
        # Extract text with optional clipping
        text = page.get_text()
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
    6. There migh be multiple trademarks with same name, don't swap them with anyother detail present in the index text, extract trademark name exactly.

    Output Format: The output should be in the following JSON format:
    [
        {{
            "name": "ARRID EXTRA DRY",
            "page-start": 15
        }},
        {{
            "name": "ARRID EXTRA EXTRA DRY",
            "page-start": 18
        }},
        {{
            "name": "EXTRA RICH FOR DRY, THIRSTY HAIR",
            "page-start": 21
        }},
        {{
            "name": "GOOD LEAF",
            "page-start": 89
        }},
        {{
            "name": "SHEAR GENIUS",
            "page-start": 225
        }},
        {{
            "name": "SHEAR GENIUS",
            "page-start": 226
        }},
        {{
            "name": "SHEAR GENIUS",
            "page-start": 227
        }},
        {{
            "name": "NOTE",
            "page-start": 16
        }}
    ]
    Important Notes:
    Make sure to extract each trademark and its start page from the given text completely without skipping any entries.
    Handle multiple occurrences of the same trademark name with different page numbers by creating a separate entry for each occurrence.
    The input text will follow a similar structure to the example provided, so adapt the extraction logic accordingly.
    Extracted text from index:
    {index}
    ---
    Return only the structured output with no additional text.
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


async def extract_trademark_details(document_chunk: str, tm_name: str):
    max_retries = 5  # Maximum number of retries
    base_delay = 1  # Base delay in seconds
    jitter = 0.5  # Maximum jitter to add to the delay
    
    # Define the schema for function calling
    function_schema = {
        "name": "extract_trademark_details",
        "description": "Extracts trademark details from a provided document chunk.",
        "parameters": {
            "type": "object",
            "properties": {
                "trademark_name": {"type": "string", "description": "The name of the trademark."},
                "status": {"type": "string", "description": "The status of the trademark (e.g., Registered, Cancelled)."},
                "serial_number": {"type": "string", "description": "The serial number of the trademark."},
                "international_class_number": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "A list of international class numbers associated with the trademark.",
                },
                "goods_services": {"type": "string", "description": "The goods and services associated with the trademark."},
                "owner": {"type": "string", "description": "The owner of the trademark."},
                "filed_date": {"type": "string", "description": "The filed date of the trademark in MMM DD, YYYY format."},
                "registration_number": {"type": "string", "description": "The registration number of the trademark."},
                "design_phrase": {"type": "string", "description": "The design phrase of the trademark."},
            },
            "required": [
                "trademark_name", 
                "status", 
                "serial_number", 
                "international_class_number", 
                "goods_services", 
                "owner", 
                "filed_date", 
                "registration_number", 
                "design_phrase",
            ],
        },
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            client = OpenAIClient(
                api_key=llm_api_key,
                base_url=azure_llm_endpoint,
                api_version="2024-10-01-preview",
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a data extraction specialist proficient in parsing trademark documents.",
                },
                {
                    "role": "user",
                    "content": f"""
                    Extract the following details from the provided trademark document chunk:
    
                    Document chunk: 
                    {document_chunk}
                    Trademark name: {tm_name}
    
                    Ensure the response adheres to the pre-defined schema strictly.
                """,
                },
            ]
            
            # Make the OpenAI API call with function calling
            response = await client.chat_complete(
                model="gpt-4o",
                messages=messages,
                functions=[function_schema],  # Pass the schema
                function_call={"name": "extract_trademark_details"},  # Explicitly call the function
                temperature=0,
            )
            
            # Extract the structured response
            function_response = response["choices"][0]["message"]["function_call"]["arguments"]
            structured_data = json.loads(function_response)  # Convert JSON string to Python dict
            
            # Validate the response using Pydantic
            validated_data = TrademarkDetails(**structured_data)
            return validated_data.dict()  # Return validated data as a dictionary

        except ValidationError as ve:
            print(f"Validation Error: {ve.json()}")  # Log detailed validation errors
            raise ve  # Re-raise the validation error
        except Exception as e:
            if attempt == max_retries:
                raise  # Raise the exception if we've reached the maximum retries
            else:
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                delay_with_jitter = delay + random.uniform(0, jitter)
                print(f"Attempt {attempt} failed. Retrying in {delay_with_jitter:.2f} seconds...")
                await asyncio.sleep(delay_with_jitter)
