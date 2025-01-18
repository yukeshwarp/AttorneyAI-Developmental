from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Dict, Union, Optional

class TrademarkDetails(BaseModel):
    trademark_name: str = Field(
        ..., description="The name of the Trademark", example="DISCOVER"
    )
    status: str = Field(
        ..., description="The Status of the Trademark", example="Registered"
    )
    serial_number: str = Field(
        ...,
        description="The Serial Number of the trademark from Chronology section",
        example="87−693,628",
    )
    international_class_number: List[int] = Field(
        ...,
        description="The International class number or Nice Classes number of the trademark from Goods/Services section or Nice Classes section",
        example=[18],
    )
    owner: str = Field(
        ..., description="The owner of the trademark", example="WALMART STORES INC"
    )
    goods_services: str = Field(
        ...,
        description="The goods/services from the document",
        example="LUGGAGE AND CARRYING BAGS; SUITCASES, TRUNKS, TRAVELLING BAGS, SLING BAGS FOR CARRYING INFANTS, SCHOOL BAGS; PURSES; WALLETS; RETAIL AND ONLINE RETAIL SERVICES",
    )
    page_number: int = Field(
        ...,
        description="The page number where the trademark details are found in the document",
        example=3,
    )
    registration_number: Union[str, None] = Field(
        None,
        description="The Registration number of the trademark from Chronology section",
        example="5,809,957",
    )
    design_phrase: str = Field(
        "",
        description="The design phrase of the trademark",
        example="THE MARK CONSISTS OF THE STYLIZED WORD 'MINI' FOLLOWED BY 'BY MOTHERHOOD.'",
    )


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
