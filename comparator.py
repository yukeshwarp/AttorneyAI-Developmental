import ast
import requests
from utils.config import *
from rapidfuzz import fuzz
from typing import List, Dict, Union, Optional
from openai import AzureOpenAI
import re
import fitz
import random
import requests
import json
import asyncio
import phonetics


def compare_trademarks2(
    existing_trademark: List[Dict[str, Union[str, List[int]]]],
    proposed_name: str,
    proposed_class,
    proposed_goods_services: str,
) -> List[Dict[str, Union[str, int]]]:
    proposed_classes = proposed_class

    # Prepare the messages for the Azure OpenAI API
    messages = [
        {
            "role": "system",
            "content": """  
            You are a trademark attorney tasked with determining a conflict grade based on the given conditions.  
           
            Additional Instructions: 
           
            - Consider if the proposed trademark name appears anywhere within the existing trademark name, or if significant parts of the existing trademark name appear in the proposed name.  
            - Evaluate shared words between trademarks, regardless of their position.  
            - Assess phonetic similarities, including partial matches or subtle matches.  
            - Consider the overall impression created by the trademarks, including similarities in appearance, sound, pronounciation, and meaning.  
           
            Follow the conflict grading criteria as previously outlined, assigning "Name-Match" or "No-conflict" based on your analysis.  
            """,
        },
        {
            "role": "user",
            "content": f"""  
            Evaluate the potential conflict between the following existing trademarks and the proposed trademark.  
           
            Proposed Trademark:
            - Name: "{proposed_name}"  
           
            Existing Trademarks:
            - Name: "{existing_trademark['trademark_name']}"  
            - Status: "{existing_trademark['status']}"
           
            Instructions:
            1. Review the proposed and existing trademark data.  
            2. Determine if the trademarks are likely to cause confusion based on the Trademark name such as Phonetic match, Semantic similarity and String similarity.  
            3. Return the output with Conflict Grade only as 'Name-Match' or 'No-conflict', based on the reasoning.
            4. Provide reasoning for each Conflict Grade.
            5. Special Case: If the existing trademark status is "Cancelled" or "Abandoned," it will automatically be considered as No-conflict.  
           
            Output Format:
                Existing Name: Name of the existing trademark.
                Reasoning: Reasoning.
                Conflict Grade: Name-Match/No-conflict
        """,
        },
    ]



    # Call Azure OpenAI to get the response
    try:
        response_reasoning = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=500,
            top_p=1,
        )

        # Extract the content from the response
        reasoning_content = response_reasoning.choices[0].message.content
        conflict_grade = reasoning_content.split("Conflict Grade:", 1)[1].strip()
        reason = reasoning_content.split("Conflict Grade:", 1)[0].strip()
        st.write(reasoning_content)

        return conflict_grade, reason

    except Exception as e:
        print(f"Error while calling Azure OpenAI API: {e}")
        return []


def assess_conflict(
    existing_trademark: List[Dict[str, Union[str, List[int]]]],
    proposed_name: str,
    proposed_class,
    proposed_goods_services: str,
) -> List[Dict[str, int]]:

    def normalize_text_name(text):
        """Normalize text by converting to lowercase, removing special characters, and standardizing whitespace."""
        # Remove punctuation except hyphens and spaces
        # text = re.sub(r"[^\w\s-’]", "", text)
        # Convert to lowercase
        text = re.sub(r"’", " ", text)
        text = text.lower()
        # Standardize whitespace
        return " ".join(text.split())

    # Clean and standardize the trademark status
    status = existing_trademark["status"].strip().lower()
    # Check for 'Cancelled' or 'Abandoned' status
    if any(keyword in status for keyword in ["cancelled", "abandoned", "expired"]):
        conflict_grade = "No-conflict"
        reasoning = "The existing trademark status is 'Cancelled' or 'Abandoned.'"
    else:

        existing_trademark_name = normalize_text_name(
            existing_trademark["trademark_name"]
        )
        proposed_name = normalize_text_name(proposed_name)

        # Phonetic Comparison
        existing_phonetic = phonetics.metaphone(existing_trademark_name)
        proposed_phonetic = phonetics.metaphone(proposed_name)
        phonetic_match = existing_phonetic == proposed_phonetic

        # Semantic Similarity
        existing_embedding = semantic_model.encode(
            existing_trademark_name, convert_to_tensor=True
        )
        proposed_embedding = semantic_model.encode(
            proposed_name, convert_to_tensor=True
        )
        semantic_similarity = util.cos_sim(
            existing_embedding, proposed_embedding
        ).item()

        # String Similarity
        from rapidfuzz import fuzz

        string_similarity = fuzz.ratio(existing_trademark_name, proposed_name)

        def is_substring_match(name1, name2):
            return name1.lower() in name2.lower() or name2.lower() in name1.lower()

        substring_match = is_substring_match(existing_trademark_name, proposed_name)

        def has_shared_word(name1, name2):
            words1 = set(name1.lower().split())
            words2 = set(name2.lower().split())
            return not words1.isdisjoint(words2)

        shared_word = has_shared_word(existing_trademark_name, proposed_name)

        from fuzzywuzzy import fuzz

        def is_phonetic_partial_match(name1, name2, threshold=55):
            return fuzz.partial_ratio(name1.lower(), name2.lower()) >= threshold

        phonetic_partial_match = is_phonetic_partial_match(
            existing_trademark_name, proposed_name
        )

        if (
            phonetic_match
            or substring_match
            or shared_word
            or semantic_similarity >= 0.5
            or string_similarity >= 55
            or phonetic_partial_match >= 55
        ):
            conflict_grade = "Name-Match"
        else:
            conflict_grade = "No-conflict"

        semantic_similarity = semantic_similarity * 100

        # Reasoning
        reasoning = (
            f"Condition 1: {'Satisfied' if phonetic_match else 'Not Satisfied'} - Phonetic match found.\n"
            f"Condition 2: {'Satisfied' if substring_match else 'Not Satisfied'} - Substring match found.\n"
            f"Condition 3: {'Satisfied' if shared_word else 'Not Satisfied'} - Substring match found.\n"
            f"Condition 4: {'Satisfied' if phonetic_partial_match >= 55 else 'Not Satisfied'} - String similarity is ({round(phonetic_partial_match)}%).\n"
            f"Condition 5: {'Satisfied' if semantic_similarity >= 50 else 'Not Satisfied'} - Semantic similarity is ({round(semantic_similarity)}%).\n"
            f"Condition 6: {'Satisfied' if string_similarity >= 55 else 'Not Satisfied'} - String similarity is ({round(string_similarity)}%).\n"
        )

    return {
        "Trademark name": existing_trademark["trademark_name"],
        "Trademark status": existing_trademark["status"],
        "Trademark owner": existing_trademark["owner"],
        "Trademark class Number": existing_trademark["international_class_number"],
        "Trademark serial number": existing_trademark["serial_number"],
        "Trademark registration number": existing_trademark["registration_number"],
        "Trademark design phrase": existing_trademark["design_phrase"],
        "conflict_grade": conflict_grade,
        "reasoning": reasoning,
    }


def compare_trademarks(
    existing_trademark: Dict[str, Union[str, List[int]]],
    proposed_name: str,
    proposed_class,
    proposed_goods_services: str,
) -> Dict[str, Union[str, int]]:
    # Convert proposed classes to a list of integers
    international_class_numbers = existing_trademark["international_class_number"]
    proposed_classes = proposed_class
    if not (any(cls in international_class_numbers for cls in proposed_classes)):
        return assess_conflict(
            existing_trademark, proposed_name, proposed_class, proposed_goods_services
        )

    # Helper function for semantic equivalence
    def is_semantically_equivalent(name1, name2, threshold=0.80):
        embeddings1 = semantic_model.encode(name1, convert_to_tensor=True)
        embeddings2 = semantic_model.encode(name2, convert_to_tensor=True)
        similarity_score = util.cos_sim(embeddings1, embeddings2).item()
        return similarity_score >= threshold

    # Helper function for phonetic equivalence
    def is_phonetically_equivalent(name1, name2, threshold=80):
        return fuzz.ratio(name1.lower(), name2.lower()) >= threshold

    # Helper function for phonetically equivalent words
    def first_words_phonetically_equivalent(existing_name, proposed_name, threshold=80):
        existing_words = existing_name.lower().split()
        proposed_words = proposed_name.lower().split()
        if len(existing_words) < 2 or len(proposed_words) < 2:
            return False
        return (
            fuzz.ratio(" ".join(existing_words[:2]), " ".join(proposed_words[:2]))
            >= threshold
        )

    def is_exact_match(name1: str, name2: str) -> bool:
        # Initial exact match check
        if name1.strip().lower() == name2.strip().lower():
            return True
        else:
            # Check for near-exact matches using normalized forms
            normalized_name1 = normalize_text(name1)
            normalized_name2 = normalize_text(name2)
            if normalized_name1 == normalized_name2:
                return True
            elif fuzz.ratio(normalized_name1, normalized_name2) >= 95:
                # Near-exact match, supplement with LLM
                return is_exact_match_llm(name1, name2)
            else:
                return False

    def normalize_text(text: str) -> str:
        import unicodedata
        import re

        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)
        # Remove diacritics
        text = "".join(c for c in text if not unicodedata.combining(c))
        # Remove special characters and punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Convert to lowercase and strip whitespace
        return text.lower().strip()

    def is_exact_match_llm(name1: str, name2: str) -> bool:
        from openai import AzureOpenAI
        import os


        prompt = f"""  
            Are the following two trademark names considered exact matches, accounting for minor variations such as special characters, punctuation, or formatting? Respond with 'Yes' or 'No'.  
            
            Trademark Name 1: "{name1}"  
            Trademark Name 2: "{name2}"  
            """

        messages = [
            {
                "role": "system",
                "content": "You are a trademark expert specializing in name comparisons.",
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=5,
        )

        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer.lower()

    def is_semantically_equivalents(
        name1: str, name2: str, threshold: float = 0.80
    ) -> bool:
        embeddings1 = semantic_model.encode(name1, convert_to_tensor=True)
        embeddings2 = semantic_model.encode(name2, convert_to_tensor=True)
        similarity_score = util.cos_sim(embeddings1, embeddings2).item()
        if similarity_score >= threshold:
            return True
        elif similarity_score >= (threshold - 0.1):
            # Near-threshold case, supplement with LLM
            return is_semantically_equivalent_llm(name1, name2)
        else:
            return False

    def is_semantically_equivalent_llm(name1: str, name2: str) -> bool:
        prompt = f"""  
        Are the following two trademark names semantically equivalent? Respond with 'Yes' or 'No'.  
        
        Trademark Name 1: "{name1}"  
        Trademark Name 2: "{name2}"  
        """


        messages = [
            {
                "role": "system",
                "content": "You are an expert in trademark law and semantics.",
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=5,
        )

        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer.lower()

    def is_phonetically_equivalents(
        name1: str, name2: str, threshold: int = 80
    ) -> bool:
        from metaphone import doublemetaphone

        dm_name1 = doublemetaphone(name1)
        dm_name2 = doublemetaphone(name2)
        phonetic_similarity = fuzz.ratio(dm_name1[0], dm_name2[0])
        if phonetic_similarity >= threshold:
            return True
        elif phonetic_similarity >= (threshold - 10):
            # Near-threshold case, supplement with LLM
            return is_phonetically_equivalent_llm(name1, name2)
        else:
            return False

    def is_phonetically_equivalent_llm(name1: str, name2: str) -> bool:

        prompt = f"""  
        Do the following two trademark names sound the same or very similar when spoken aloud? Consider differences in spelling but similarities in pronunciation. Respond with 'Yes' or 'No'.  
        
        Trademark Name 1: "{name1}"  
        Trademark Name 2: "{name2}"  
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert in phonetics and trademark law.",
            },
            {"role": "user", "content": prompt},
        ]


        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=5,
        )

        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer.lower()

    condition_1A_satisfied = (
        existing_trademark["trademark_name"].strip().lower()
        == proposed_name.strip().lower()
    )
    condition_1B_satisfied = is_semantically_equivalent(
        existing_trademark["trademark_name"], proposed_name
    )
    condition_1C_satisfied = is_phonetically_equivalent(
        existing_trademark["trademark_name"], proposed_name
    )
    condition_1D_satisfied = first_words_phonetically_equivalent(
        existing_trademark["trademark_name"], proposed_name
    )
    condition_1E_satisfied = (
        existing_trademark["trademark_name"].lower().startswith(proposed_name.lower())
    )
    condition_1_satisfied = any(
        [
            condition_1A_satisfied,
            condition_1B_satisfied,
            condition_1C_satisfied,
            condition_1D_satisfied,
            condition_1E_satisfied,
        ]
    )

    def target_market_and_goods_overlaps(existing_gs, proposed_gs, threshold=0.65):
        embeddings1 = semantic_model.encode(existing_gs, convert_to_tensor=True)
        embeddings2 = semantic_model.encode(proposed_gs, convert_to_tensor=True)
        similarity_score = util.cos_sim(embeddings1, embeddings2).item()
        if similarity_score >= threshold:
            return True
        elif similarity_score >= (threshold - 0.1):
            # Supplement with LLM
            return target_market_and_goods_overlap_llm(existing_gs, proposed_gs)
        else:
            # Further check using keyword overlap
            # ... Additional code
            return False

    def target_market_and_goods_overlap_llm(existing_gs: str, proposed_gs: str) -> bool:
        prompt = f"""  
            Do the goods and services described in the existing trademark and the proposed trademark overlap or target the same market? Consider the descriptions carefully. Respond with 'Yes' or 'No'.  
            
            Existing Trademark Goods/Services:  
            "{existing_gs}"  
            
            Proposed Trademark Goods/Services:  
            "{proposed_gs}"  
            """

        messages = [
            {
                "role": "system",
                "content": "You are an expert in trademark law and market analysis.",
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=5,
        )

        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer.lower()


    # Check if any class in proposed_classes is in the international_class_numbers
    condition_2_satisfied = any(
        cls in international_class_numbers for cls in proposed_classes
    )

    import re
    from nltk.stem import WordNetLemmatizer
    def normalize_text(text):

        # Replace special hyphen-like characters with a standard hyphen
        text = re.sub(r"[−–—]", "-", text)
        # Remove punctuation except hyphens and spaces
        text = re.sub(r"[^\w\s-]", " ", text)
        # Convert to lowercase
        text = text.lower()
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"\bclass\b", "", text)
        text = re.sub(r"\bcare\b", "", text)
        text = re.sub(r"\bin\b", "", text)
        text = re.sub(r"\band\b", "", text)
        text = re.sub(r"\bthe\b", "", text)
        text = re.sub(r"\bfor\b", "", text)
        text = re.sub(r"\bwith\b", "", text)
        text = re.sub(r"\bfrom\b", "", text)
        text = re.sub(r"\bto\b", "", text)
        text = re.sub(r"\bunder\b", "", text)
        text = re.sub(r"\busing\b", "", text)
        text = re.sub(r"\bof\b", "", text)
        text = re.sub(r"\bno\b", "", text)
        text = re.sub(r"\binclude\b", "", text)
        text = re.sub(r"\bex\b", "", text)
        text = re.sub(r"\bexample\b", "", text)
        text = re.sub(r"\bclasses\b", "", text)
        text = re.sub(r"\bsearch\b", "", text)
        text = re.sub(r"\bscope\b", "", text)
        text = re.sub(r"\bshower\b", "", text)
        text = re.sub(r"\bproducts\b", "", text)
        text = re.sub(r"\bshampoos\b", "hair", text)

        # Standardize whitespace
        return " ".join(text.split())

    # Condition 3: Target market and goods/services overlap
    def target_market_and_goods_overlap(existing_gs, proposed_gs, threshold=0.65):
        # Normalize both strings
        existing_normalized = normalize_text(existing_gs)
        proposed_normalized = normalize_text(proposed_gs)

        embeddings1 = semantic_model.encode(existing_normalized, convert_to_tensor=True)
        embeddings2 = semantic_model.encode(proposed_normalized, convert_to_tensor=True)
        similarity_score = util.cos_sim(embeddings1, embeddings2).item()
        # st.write("Semantic Similarity Score:", similarity_score)
        if similarity_score >= threshold:
            return True

        # Split into words and lemmatize
        lemmatizer = WordNetLemmatizer()
        existing_words = {
            lemmatizer.lemmatize(word) for word in existing_normalized.split()
        }
        proposed_words = {
            lemmatizer.lemmatize(word) for word in proposed_normalized.split()
        }

        # Check for common words
        common_words = existing_words.intersection(proposed_words)
        # st.write("Common Words:", common_words)
        return bool(common_words)

    condition_3_satisfied = target_market_and_goods_overlap(
        existing_trademark["goods_services"], proposed_goods_services
    )

    # Clean and standardize the trademark status
    status = existing_trademark["status"].strip().lower()

    # Check for 'Cancelled' or 'Abandoned' status
    if any(keyword in status for keyword in ["cancelled", "abandoned", "expired"]):
        conflict_grade = "Low"
        reasoning = "The existing trademark status is 'Cancelled' or 'Abandoned.'"
    else:
        points = sum(
            [
                condition_1_satisfied,  # 1 point if any Condition 1 is satisfied
                condition_2_satisfied,  # 1 point if Condition 2 is satisfied
                condition_3_satisfied,  # 1 point if Condition 3 is satisfied
            ]
        )

        # Determine conflict grade based on points
        if points == 3:
            conflict_grade = "High"
        elif points == 2:
            conflict_grade = "Moderate"
        elif points == 1:
            conflict_grade = "Low"
        else:
            conflict_grade = compare_trademarks2(
                existing_trademark,
                proposed_name,
                proposed_class,
                proposed_goods_services,
            )

        if condition_1_satisfied:
            condition_1_details = []
            if condition_1A_satisfied:
                condition_1_details.append("Exact character-for-character match")
            if condition_1B_satisfied:
                condition_1_details.append("Semantically equivalent")
            if condition_1C_satisfied:
                condition_1_details.append("Phonetically equivalent")
            if condition_1D_satisfied:
                condition_1_details.append(
                    "First two or more words are phonetically equivalent"
                )
            if condition_1E_satisfied:
                condition_1_details.append(
                    "Proposed name is the first word of the existing trademark"
                )

        # Generate detailed reasoning for Condition 1
        if condition_1_satisfied:
            condition_1_reasoning = (
                f"Condition 1: Satisfied - {', '.join(condition_1_details)}."
            )
        else:
            condition_1_reasoning = "Condition 1: Not Satisfied."

        # Reasoning
        reasoning = (
            f"{condition_1_reasoning} \n"
            f"Condition 2: {'Satisfied' if condition_2_satisfied else 'Not Satisfied'} - Overlap in class numbers.\n"
            f"Condition 3: {'Satisfied' if condition_3_satisfied else 'Not Satisfied'} - Overlap in goods/services and target market."
        )

    # Return results
    return {
        "Trademark name": existing_trademark["trademark_name"],
        "Trademark status": existing_trademark["status"],
        "Trademark owner": existing_trademark["owner"],
        "Trademark class Number": existing_trademark["international_class_number"],
        "Trademark serial number": existing_trademark["serial_number"],
        "Trademark registration number": existing_trademark["registration_number"],
        "Trademark design phrase": existing_trademark["design_phrase"],
        "conflict_grade": conflict_grade,
        "reasoning": reasoning,
    }
