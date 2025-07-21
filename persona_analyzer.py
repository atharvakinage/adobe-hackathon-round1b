import os
import json
import datetime
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
import joblib


from round1a_module.outline_extractor import analyze_pdf_with_ml, get_text_properties

# --- Configuration ---
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
INPUT_JSON_FILENAME = "challenge1b_input.json"

# --- Helper Functions ---
def load_challenge_input(input_dir):
    """Loads the challenge1b_input.json file."""
    input_json_path = os.path.join(input_dir, INPUT_JSON_FILENAME)

    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file '{INPUT_JSON_FILENAME}' not found at {input_json_path}.")
        return None, None, None, None, None

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            challenge_input = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_json_path}: {e}")
        return None, None, None, None, None
    
    documents_info = challenge_input.get('documents', [])
    persona_data = challenge_input.get('persona', {})
    job_to_be_done = challenge_input.get('job_to_be_done', '') 

    pdf_paths = []
    input_document_names = []
    for doc_info in documents_info:
        filename = doc_info.get('filename')
        if filename:
            full_pdf_path = os.path.join(input_dir, filename)
            if os.path.exists(full_pdf_path):
                pdf_paths.append(full_pdf_path)
                input_document_names.append(filename)
            else:
                print(f"Warning: Document '{filename}' specified in input JSON not found at {full_pdf_path}. Skipping.")
    
    return pdf_paths, persona_data, job_to_be_done, input_document_names

def extract_full_text_by_page(pdf_path):
    """Extracts full text content for each page of a PDF."""
    page_texts = defaultdict(str)
    try:
        for page_num, page_layout in enumerate(extract_pages(pdf_path)):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_texts[page_num] += element.get_text() + " "
    except (ValueError, PDFSyntaxError, Exception) as e:
        print(f"Error extracting full text from {pdf_path}: {e}")
        return defaultdict(str)
    return page_texts

def analyze_documents_for_persona(pdf_paths, persona_data, job_description):
    """
    Analyzes documents to find and rank relevant sections based on persona and job.
    """
    extracted_sections = []
    
    query_text = f"{persona_data.get('role', '')} {persona_data.get('expertise', '')} {persona_data.get('focus_areas', '')} {job_description}"
    query_text = query_text.strip()

    if not query_text:
        print("Warning: Empty persona and job description. Cannot determine relevance.")
        return []

    doc_sections_for_vectorization = []
    
    MODEL_PATH = "/app/round1a_module/model/outline_classifier.joblib"
    SCALER_PATH = "/app/round1a_module/model/scaler.joblib"
    
    model = None
    scaler = None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Successfully loaded Round 1A ML model and scaler from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Round 1A ML model or scaler not found at {MODEL_PATH} or {SCALER_PATH}.")
        print("Please ensure your 'adobe-hackathon-round1a/model/' content is copied to 'adobe-hackathon-round1b/round1a_module/model/'.")
        return []

    for doc_path in pdf_paths:
        doc_filename = os.path.basename(doc_path)
        print(f"Extracting outline for {doc_filename} using Round 1A ML module...")
        
        outline_data = analyze_pdf_with_ml(doc_path, model, scaler)
        
        page_full_texts = extract_full_text_by_page(doc_path)

        for i, section in enumerate(outline_data.get('outline', [])):
            section_text = section['text']
            section_page = section['page']
            
            next_heading_page = section_page + 1
            if i + 1 < len(outline_data.get('outline', [])):
                next_heading_page = outline_data['outline'][i+1]['page']
            
            section_content = ""
            for p_num in range(section_page, min(next_heading_page, section_page + 5, len(page_full_texts))):
                if page_full_texts[p_num]:
                    section_content += page_full_texts[p_num]
            
            if not section_content and page_full_texts[section_page]:
                 section_content = page_full_texts[section_page]

            if section_content:
                doc_sections_for_vectorization.append({
                    "doc_path": doc_path,
                    "doc_filename": doc_filename,
                    "section_obj": section,
                    "full_content": section_content,
                    "page_start_for_content": section_page,
                    "page_end_for_content": next_heading_page
                })

    if not doc_sections_for_vectorization:
        print("No sections found in documents for relevance analysis.")
        return []

    corpus = [item["full_content"] for item in doc_sections_for_vectorization]
    corpus.append(query_text)

    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9) 
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vector = tfidf_matrix[-1]
    document_vectors = tfidf_matrix[:-1]

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()

    ranked_sections = []
    for i, item in enumerate(doc_sections_for_vectorization):
        score = similarity_scores[i]
        if score > 0:
            ranked_sections.append({
                "doc_path": item["doc_path"],
                "doc_filename": item["doc_filename"],
                "section_obj": item["section_obj"],
                "relevance_score": score,
                "full_content": item["full_content"],
                "page_start_for_content": item["page_start_for_content"],
                "page_end_for_content": item["page_end_for_content"]
            })
    
    ranked_sections.sort(key=lambda x: x["relevance_score"], reverse=True)

    final_extracted_sections = []
    for rank, section_data in enumerate(ranked_sections):
        section_title = section_data["section_obj"]["text"]
        section_page = section_data["section_obj"]["page"]
        doc_filename = section_data["doc_filename"]
        
        refined_text = section_data["full_content"] 

        final_extracted_sections.append({
            "document": doc_filename,
            "page_number": section_page,
            "section_title": section_title,
            "importance_rank": rank + 1,
            "sub_section_analysis": {
                "document": doc_filename,
                "refined_text": refined_text,
                "page_number": section_page
            }
        })
    
    return final_extracted_sections

def main():
    pdf_paths, persona_data, job_description, input_document_names = load_challenge_input(INPUT_DIR)

    if pdf_paths is None:
        return
    if not pdf_paths:
        print(f"Error: No valid PDF files found or specified in '{INPUT_JSON_FILENAME}'.")
        return

    print(f"Analyzing {len(pdf_paths)} PDFs with persona role: '{persona_data.get('role', 'N/A')}' and job: '{job_description}'")

    extracted_sections = analyze_documents_for_persona(pdf_paths, persona_data, job_description)

    output_data = {
        "metadata": {
            "input_documents": input_document_names,
            "persona": persona_data,
            "job_to_be_done": job_description,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections
    }

    output_json_path = os.path.join(OUTPUT_DIR, "challenge1b_output.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nRound 1B analysis complete. Output saved to: {output_json_path}")

if __name__ == "__main__":
    main()
