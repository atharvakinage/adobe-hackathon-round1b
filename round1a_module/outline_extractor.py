import os
import json
import joblib 
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTRect, LTFigure, LTTextBoxHorizontal

current_pdf_path = None

def get_font_weight_score(fontname):
    """Assigns a numerical score based on font name to indicate boldness."""
    if not fontname:
        return 0
    fontname_lower = fontname.lower()
    score = 0
    if 'bold' in fontname_lower or 'bd' in fontname_lower or 'heavy' in fontname_lower or 'black' in fontname_lower:
        score += 2
    if 'italic' in fontname_lower or 'it' in fontname_lower:
        score += 0.5
    return score

def get_text_properties(element, page_width, page_height, page_font_sizes=None, prev_element_bbox=None):
    """
    Extracts a rich set of features from a PDF text element (LTTextBoxHorizontal).
    Requires page_width and page_height for normalization.
    """
    if not isinstance(element, LTTextBoxHorizontal): # Now expecting LTTextBoxHorizontal
        return None
    
    text = element.get_text().strip()
    if not text:
        return None
        
    font_size = 0
    font_is_bold = False
    font_is_italic = False
    font_weight_score = 0
    
    for text_line in element:
        for character in text_line:
            if isinstance(character, LTChar):
                font_size = round(character.size, 2)
                fontname = character.fontname.lower()
                font_weight_score = get_font_weight_score(character.fontname)
                if 'bold' in fontname or 'bd' in fontname:
                    font_is_bold = True
                if 'italic' in fontname or 'it' in fontname:
                    font_is_italic = True
                break
        if font_size > 0:
            break

    relative_font_size = 0
    if page_font_sizes and font_size:
        median_font_size = np.median(list(page_font_sizes))
        if median_font_size > 0:
            relative_font_size = font_size / median_font_size

    vertical_space_above = 0
    if prev_element_bbox and element.bbox:
        vertical_space_above = element.bbox[1] - prev_element_bbox[3]
        vertical_space_above = max(0, min(vertical_space_above, 100))

    x_position_normalized = element.bbox[0] / page_width if page_width > 0 else 0
    
    line_width = element.bbox[2] - element.bbox[0]
    char_density = len(text) / line_width if line_width > 0 else 0

    has_prefix = 0
    if text and len(text) > 2:
        first_word = text.split(' ')[0]

        if (first_word.endswith('.') and (first_word[:-1].isdigit() or first_word[:-1].isalpha())) or \
           (first_word.isdigit() and len(first_word) < 4) or \
           (first_word.isupper() and len(first_word) < 8 and first_word.isalpha()) or \
           (text.lower().startswith("chapter ") and len(text.split()) < 5) or \
           (text.lower().startswith("section ") and len(text.split()) < 5):
            has_prefix = 1

    is_numeric_only = text.replace('.', '').replace(',', '').replace(' ', '').isdigit() and len(text) < 10

    return {
        "text": text,
        "font_size": font_size,
        "is_uppercase": text.isupper(),
        "is_bold": font_is_bold,
        "is_italic": font_is_italic,
        "font_weight_score": font_weight_score, 
        "line_length": len(text),
        "x_position": element.bbox[0], 
        "x_position_normalized": x_position_normalized, 
        "relative_font_size": relative_font_size,
        "vertical_space_above": vertical_space_above,
        "has_prefix": has_prefix,
        "char_density": char_density, 
        "is_numeric_only": is_numeric_only, 
        "bbox": element.bbox 
    }

def analyze_pdf_with_ml(pdf_path, model, scaler):
    """
    Analyzes a PDF using a pre-trained ML model for heading classification.
    """
    outline = []
    all_elements_raw = [] 
    title = "Untitled Document"

    for page_num, page_layout in enumerate(extract_pages(pdf_path)):
        page_width = page_layout.bbox[2] - page_layout.bbox[0]
        page_height = page_layout.bbox[3] - page_layout.bbox[1]

        page_elements_on_page = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                page_elements_on_page.append({
                    "element": element,
                    "page": page_num,
                    "bbox": element.bbox,
                    "page_width": page_width,
                    "page_height": page_height
                })
            elif isinstance(element, (LTLine, LTRect, LTFigure)):
                 page_elements_on_page.append({
                    "element": element,
                    "page": page_num,
                    "bbox": element.bbox,
                    "page_width": page_width,
                    "page_height": page_height
                })
        
        page_elements_on_page.sort(key=lambda x: x["bbox"][1], reverse=True)
        all_elements_raw.extend(page_elements_on_page)

    first_page_text_elements = [
        e["element"] for e in all_elements_raw if e["page"] == 1 and isinstance(e["element"], LTTextBoxHorizontal)
    ]
    if first_page_text_elements:
        title_candidates = []
        for elem in first_page_text_elements:
            page_data = next((e for e in all_elements_raw if e["element"] == elem), None)
            if page_data:
                props = get_text_properties(elem, page_data["page_width"], page_data["page_height"])
                if props:
                    title_candidates.append(props)
        
        if title_candidates:
            title_candidates.sort(key=lambda x: (x["font_size"], x["bbox"][3]), reverse=True)
            for cand in title_candidates:
                if cand["text"].strip() and len(cand["text"].strip()) > 10 and \
                   not cand["is_numeric_only"] and \
                   not cand["text"].lower().startswith("table of contents") and \
                   not cand["text"].lower().startswith("revision history") and \
                   not cand["text"].lower().startswith("acknowledgements") and \
                   not cand["text"].lower().startswith("page ") and \
                   not (cand["text"].lower().count("overview") > 0 and cand["text"].lower().count("foundation level") > 0):
                    title = cand["text"].strip().replace('\n', ' ')
                    break
            if title == "Untitled Document" and title_candidates:
                title = title_candidates[0]["text"].strip().replace('\n', ' ')


    features_list = []
    texts = []
    pages = []
    
    prev_element_bbox = None
    page_font_sizes_cache = {} 

    for i, elem_data in enumerate(all_elements_raw):
        element = elem_data["element"]
        page_num = elem_data["page"]
        page_width = elem_data["page_width"]
        page_height = elem_data["page_height"]

        if isinstance(element, LTTextBoxHorizontal):
            if page_num not in page_font_sizes_cache:
                current_page_text_elements = [
                    e["element"] for e in all_elements_raw if e["page"] == page_num and isinstance(e["element"], LTTextBoxHorizontal)
                ]
                page_font_sizes_cache[page_num] = {
                    get_text_properties(e, page_width, page_height)["font_size"]
                    for e in current_page_text_elements if get_text_properties(e, page_width, page_height) and get_text_properties(e, page_width, page_height)["font_size"] > 0
                }

            props = get_text_properties(element, page_width, page_height, page_font_sizes_cache.get(page_num), prev_element_bbox)
            
            if props:
                features_list.append([
                    props["font_size"],
                    int(props["is_uppercase"]), 
                    int(props["is_bold"]),      
                    int(props["is_italic"]),    
                    props["font_weight_score"], 
                    props["line_length"],
                    props["x_position"], 
                    props["x_position_normalized"], 
                    props["relative_font_size"],
                    props["vertical_space_above"],
                    int(props["has_prefix"]),   
                    props["char_density"],      
                    int(props["is_numeric_only"]) 
                ])
                texts.append(props["text"])
                pages.append(page_num) 
            prev_element_bbox = props["bbox"] if props else element.bbox
        else:
            prev_element_bbox = element.bbox


    if not features_list:
        return {"title": title, "outline": []}
        
    scaled_features = scaler.transform(features_list)
    
    predictions = model.predict(scaled_features)
    
    level_map = {0: "H1", 1: "H2", 2: "H3", 3: "Body"}
    
    raw_outline_candidates = []
    for i, pred in enumerate(predictions):
        level = level_map.get(pred)
        cleaned_text = texts[i].strip()
        page = pages[i]
        
        if level in ["H1", "H2", "H3"]:
            raw_outline_candidates.append({
                "level": level,
                "text": cleaned_text,
                "page": page,
                "is_numeric_only": (cleaned_text.replace('.', '').replace(',', '').isdigit() and len(cleaned_text) < 10)
            })

    final_outline = []
    seen_texts = set() 

    for i, entry in enumerate(raw_outline_candidates):
        text = entry["text"]
        page = entry["page"]
        level = entry["level"] 

        if len(text) < 5 and not text.isupper() and not entry["is_numeric_only"] and not text.replace('.', '').isdigit():
            continue
        
        if entry["is_numeric_only"] or \
           text.lower().startswith("page ") or \
           text.lower().startswith("version ") or \
           text.lower().startswith("appendix ") or \
           text.lower().startswith("table of contents") or \
           text.lower().startswith("revision history") or \
           text.lower().startswith("acknowledgements") or \
           (text.lower().count("overview") > 0 and text.lower().count("foundation level") > 0 and len(text) < 40):
            continue

        if '\n' in text:
            text = text.split('\n')[0].strip()
            if not text:
                continue
            if len(text) < 5 and not text.isupper() and not (text.replace('.', '').replace(',', '').replace(' ', '').isdigit() and len(text) < 10):
                continue
        
        
        if text.lower() in seen_texts:
            continue
        
        final_outline.append({
            "level": level,
            "text": text,
            "page": page
        })
        seen_texts.add(text.lower())

    return {"title": title, "outline": final_outline}


def process_pdfs_in_directory(input_dir, output_dir, model_path, scaler_path):
    """
    Loads the ML model and processes PDFs.
    """
    print("Loading ML model and scaler...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path} or {scaler_path}.")
        print("Please ensure you have run 'python train_model.py' first to generate these files.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Searching for PDFs in: {input_dir}")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)

            print(f"Processing {filename}...")
            try:
                global current_pdf_path
                current_pdf_path = pdf_path

                outline_data = analyze_pdf_with_ml(pdf_path, model, scaler)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(outline_data, f, indent=2, ensure_ascii=False)
                print(f"Successfully processed {filename}. Output saved to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                
if __name__ == "__main__":
    INPUT_DIR = "/app/input"
    OUTPUT_DIR = "/app/output"
    MODEL_PATH = "/app/model/outline_classifier.joblib"
    SCALER_PATH = "/app/model/scaler.joblib"
    
    process_pdfs_in_directory(INPUT_DIR, OUTPUT_DIR, MODEL_PATH, SCALER_PATH)
