import streamlit as st
import requests
import faiss
import numpy as np
import hashlib
import json
import os
import pdfplumber
from sentence_transformers import SentenceTransformer

# Function to generate a file hash
def generate_file_hash(file):
    file.seek(0)  # Ensure the file pointer is at the beginning
    return hashlib.sha256(file.read()).hexdigest()

'''
# Function to extract text from PDF (Improved using pdfplumber)
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle None case
    return text.strip()
'''
def extract_structured_text_from_pdf(pdf_path):

    

    # Error handling added while trying to open/parse PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return "", {}
    
    
    structured_text = {}
    current_section = "Preamble"  
    structured_text[current_section] = ""
    
    # Extracting metadata if available or encoded in pdf
    try:
        metadata = doc.metadata
        if metadata:
            # Extract title from metadata if available
            if metadata.get('title') and len(metadata.get('title').strip()) > 3:
                structured_text["Title"] = metadata.get('title').strip()
            
            # Extract author from metadata if available
            if metadata.get('author') and len(metadata.get('author').strip()) > 3:
                structured_text["Authors"] = metadata.get('author').strip()
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    
    # Using first page text to extract title information and Authors
    first_page = doc[0]
    first_page_text = first_page.get_text("text")
    first_few_lines = first_page_text.split('\n')[:20]  
    
    # Extracting title if not already found in metadata
    if "Title" not in structured_text:
        
        for line in first_few_lines:
            if line.strip() and len(line.strip()) > 3:
                
                if any(word in line.lower() for word in ['journal', 'conference', 'proceedings', 'volume']):
                    continue
                
                
                blocks = first_page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for block_line in block["lines"]:
                        line_text = " ".join([span["text"] for span in block_line["spans"]])
                        if line_text.strip() == line.strip():
                            # Check if spans have larger font or bold formatting
                            for span in block_line["spans"]:
                                if span["size"] > 12 or "bold" in span.get("font", "").lower():
                                    structured_text["Title"] = line.strip()
                                    break
                        if "Title" in structured_text:
                            break
                    if "Title" in structured_text:
                        break
                
                
                if "Title" not in structured_text:
                    structured_text["Title"] = line.strip()
                break
    
    # More author extraction code if not already found in metadata
    if "Authors" not in structured_text:
        author_candidates = []
        author_section_found = False
        
        
        author_patterns = [
            r'(?:^|\s)(?:by|authors?[:]\s+)([\w\s,.()-]+)',  # "by" or "authors:" prefix
            r'(?:^|\s)((?:[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|and|\s|&|$))+)',  # Names like "John Smith, Jane Doe"
            r'(?:^|\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)+)',  # Multiple names separated by commas
            r'(?:^|\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*and\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))'  # Names connected by "and"
        ]
        
        # Looking for lines potentially containing author names
        for i, line in enumerate(first_few_lines[1:10], 1): 
            line = line.strip()
            if not line:
                continue
                
            
            if any(word.lower() in line.lower() for word in ['abstract', 'introduction', 'copyright', 'published']):
                continue
                
            # Checking for email addresses which often accompany author names
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            has_email = bool(re.search(email_pattern, line))
            
            # Checking for any academic affiliations
            affiliation_terms = ['university', 'college', 'institute', 'department', 'laboratory', 'school', 'center']
            has_affiliation = any(term in line.lower() for term in affiliation_terms)
            
            # Check for any academic titles
            academic_titles = ['dr', 'prof', 'professor', 'phd']
            has_title = any(f" {title}" in f" {line.lower()} " for title in academic_titles)
            
            # Applying the author patterns to detect author names
            is_likely_author_line = False
            for pattern in author_patterns:
                if re.search(pattern, line):
                    is_likely_author_line = True
                    break
            
            # Scoring on the probability of text being an author name
            author_score = 0
            if ',' in line or ' and ' in line.lower():
                author_score += 3
            if has_email:
                author_score += 4
            if has_affiliation:
                author_score += 2
            if has_title:
                author_score += 3
            if is_likely_author_line:
                author_score += 5
            
            
            if re.search(r'[A-Za-z]\d+', line) or re.search(r'[A-Za-z]\*', line):
                author_score += 3
            
         
            if author_score >= 3 or is_likely_author_line:
                
                cleaned_line = re.sub(email_pattern, '', line)
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                author_candidates.append(cleaned_line)
                author_section_found = True
            elif author_section_found:
                
                break
        
        if author_candidates:
            structured_text["Authors"] = ', '.join(author_candidates)
    
 
    font_sizes = []
    font_styles = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if "size" in span:
                        font_sizes.append(span["size"])
                    if "font" in span:
                        font_styles.append(span["font"].lower())

    
    median_font_size = statistics.median(font_sizes) if font_sizes else 11
    most_common_font = Counter(font_styles).most_common(1)[0][0] if font_styles else ""
    
    # Section numbering patterns - comprehensive coverage for academic papers
    section_number_patterns = [
        
        r'^\d+\.\s+',                    # "1. "
        r'^\d+\.\d+\.\s+',               # "1.1. "
        r'^\d+\.\d+\.\d+\.\s+',          # "1.1.1. "
        r'^[A-Z]\.\s+',                  # "A. "
        r'^[a-z]\)\s+',                  # "a) "
        r'^\([a-z]\)\s+',                # "(a) "
        r'^[IVXLCDM]+\.\s+',             # "IV. " (Roman numerals)
        r'^\d+\s+',                      # "1 " (just a number and space)
        r'^\d+\)',                       # "1)"
        r'^\d+\-\d+',                    # "1-1"
        r'^Section\s+\d+',               # "Section 1"
    ]
    
    
    section_number_regex = '|'.join(section_number_patterns)
    
    # Common section names in research papers for search criteria
    common_sections = [
        "abstract", "introduction", "related work", "previous work", "background",
        "methodology", "methods", "experimental setup", "experiment", "experiments",
        "evaluation", "results", "discussion", "conclusion", "conclusions",
        "future work", "references", "appendix", "acknowledgments", "acknowledgements",
        "literature review", "materials and methods", "data", "implementation", 
        "system design", "procedure", "research design", "limitations", "contributions"
    ]
    
    
    header_footer_candidates = []
    
    
    for page_idx in range(min(5, len(doc))):  
        page = doc[page_idx]
        page_dict = page.get_text("dict")
        
        
        top_text = ""
        bottom_text = ""
        
        if "blocks" in page_dict:
            
            blocks = sorted(page_dict["blocks"], key=lambda b: b["bbox"][1] if "bbox" in b else 0)
            
            
            if blocks:
                top_block = blocks[0]
                bottom_block = blocks[-1]
                
                if "lines" in top_block and top_block["lines"]:
                    line_spans = top_block["lines"][0]["spans"] if top_block["lines"][0]["spans"] else []
                    top_text = " ".join([span["text"] for span in line_spans])
                    
                if "lines" in bottom_block and bottom_block["lines"]:
                    line_spans = bottom_block["lines"][-1]["spans"] if bottom_block["lines"][-1]["spans"] else []
                    bottom_text = " ".join([span["text"] for span in line_spans])
        
        if top_text:
            header_footer_candidates.append(("top", top_text))
        if bottom_text:
            header_footer_candidates.append(("bottom", bottom_text))
    
    
    headers_footers = []
    candidate_counts = Counter([text for _, text in header_footer_candidates])
    for text, count in candidate_counts.items():
        if count >= 2 and len(text.strip()) > 0:  # If it appears on multiple pages, likely a header/footer
            headers_footers.append(text)
    
    
    def is_header_footer(text):
        return any(hf in text for hf in headers_footers)
    
   
    def get_previous_line(block, line):
        if "lines" in block:
            try:
                line_idx = block["lines"].index(line)
                if line_idx > 0:
                    return block["lines"][line_idx - 1]
            except ValueError:
                pass
        return None
    
    def get_next_line(block, line):
        if "lines" in block:
            try:
                line_idx = block["lines"].index(line)
                if line_idx < len(block["lines"]) - 1:
                    return block["lines"][line_idx + 1]
            except ValueError:
                pass
        return None
    
    def line_is_empty(line):
        if not line or "spans" not in line:
            return True
        return all(not span.get("text", "").strip() for span in line["spans"])
    
    # Processing each page for section extraction
    previous_spans = []  
    figures_tables = []  
    previous_line_empty = True 
    
    for page_num, page in enumerate(doc):
        
        blocks = page.get_text("dict")["blocks"]
        
        
        blocks = sorted(blocks, key=lambda b: b["bbox"][1] if "bbox" in b else 0)
        
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
            
            for line_idx, line in enumerate(block["lines"]):
                line_spans = []
                line_text = ""
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    
                    line_spans.append(span)
                    line_text += text + " "
                
                line_text = line_text.strip()
                if not line_text:
                    previous_line_empty = True
                    continue
                
                
                if is_header_footer(line_text):
                    continue
                
                # Checking for figure or table captions
                figure_match = re.search(r'(fig(?:ure)?\.?\s+\d+|table\.?\s+\d+)[\.\:](.*)', line_text, re.IGNORECASE)
                if figure_match:
                    figure_type = "Table" if "table" in figure_match.group(1).lower() else "Figure"
                    caption_text = figure_match.group(2).strip()
                    figures_tables.append((figure_type, caption_text))
                    continue
                
                # Checking if this is a section header based on multiple criteria
                is_header = False
                section_name = line_text
                
                # Checking for section numbering pattern
                has_section_number = bool(re.match(section_number_regex, line_text))
                
                
                clean_text = re.sub(section_number_regex, '', line_text).lower().strip()
                
                # Checking if the text contains a common section name
                contains_common_section = any(section in clean_text for section in common_sections)
                
                # Checking for formatting propert that suggest that the text is a header
                is_bold = any("bold" in span.get("font", "").lower() for span in line_spans)
                is_larger_font = any(span["size"] > (median_font_size * 1.2) for span in line_spans if "size" in span)
                is_all_caps = line_text.isupper() and len(line_text) > 3
                
                # Getting contextual information
                is_paragraph_start = False
                if line_idx == 0 or previous_line_empty:
                    is_paragraph_start = True
                
                previous_line = get_previous_line(block, line)
                next_line = get_next_line(block, line)
                previous_line_empty = line_is_empty(previous_line)
                next_line_empty = line_is_empty(next_line)
                
                                
                last_line_ends_with_sentence = False
                if previous_spans:
                    last_text = previous_spans[-1]["text"].strip()
                    
                    if last_text and (last_text.endswith('.') or last_text.endswith('?') or last_text.endswith('!')):
                        last_line_ends_with_sentence = True
                
                header_score = 0
                
                if has_section_number:
                    header_score += 6  
                if contains_common_section:
                    header_score += 5 
                
                # Require BOTH bold AND larger font for a strong signal
                if is_bold and is_larger_font:
                    header_score += 5  
                elif is_bold:  
                    header_score += 2  
                elif is_larger_font:  
                    header_score += 2
                
                # Only short all-caps text gets a boost
                if is_all_caps and len(line_text) < 30:
                    header_score += 1  
                
                # Penalize long text that's unlikely to be a header
                if len(line_text) > 30:
                    header_score -= 3 
                
                # Context analysis - boost score if appears to be at paragraph start
                if is_paragraph_start:
                    header_score += 1
                
                # Penalize if text appears to be within a paragraph
                if (previous_line and not previous_line_empty and 
                    next_line and not next_line_empty and 
                    not has_section_number and not is_all_caps):
                    header_score -= 3  

                # Then in the header scoring:
                if last_line_ends_with_sentence or previous_line_empty:
                    header_score += 3  
                else:
                    header_score -= 4  
                
                # Threshold score for considering a line as a header 
                is_header = header_score >= 10 
                
                if is_header:
                    # Cleaning section name (remove excess whitespace, numbering)
                    section_name = re.sub(section_number_regex, '', line_text)
                    section_name = section_name.strip()
                    
                    # If section name is too short, try to combine with next line
                    if len(section_name) <= 3:
                        # Look ahead to next line
                        next_line_text = ""
                        if next_line and "spans" in next_line:
                            next_line_text = " ".join([s["text"] for s in next_line["spans"] if "text" in s])
                        
                        if next_line_text and not re.match(section_number_regex, next_line_text):
                            section_name = f"{section_name} {next_line_text}".strip()
                    
                    # Checking if section name is valid
                    if len(section_name) > 1:
                        if section_name.lower() in [s.lower() for s in structured_text.keys()]:
                            
                            section_name = f"{section_name} (continued)"
                        
                        current_section = section_name
                        structured_text[current_section] = ""
                else:
                    # Appendding text to current section 
                    if not re.match(r'^\d+$', line_text) and len(line_text) > 1:
                        structured_text[current_section] += line_text + " "
                
                previous_spans = line_spans
                previous_line_empty = False
    
    # Special handling for the Abstract section
    if not any(s.lower() == "abstract" for s in structured_text.keys()):
        
        abstract_patterns = [
            r'Abstract[:\s—-]+([^\.]+(?:\.[^\.]+){0,10})',  
            r'ABSTRACT[:\s—-]+([^\.]+(?:\.[^\.]+){0,10})',  
            r'Abstract\s*\n+\s*([^\.]+(?:\.[^\.]+){0,10})' 
        ]
        
        for pattern in abstract_patterns:
            abstract_match = re.search(pattern, first_page_text, re.IGNORECASE | re.DOTALL)
            if abstract_match:
                structured_text["Abstract"] = abstract_match.group(1).strip()
                break
    
    # Special handling for references
    for key in list(structured_text.keys()):
        if "reference" in key.lower():
            ref_text = structured_text[key]
            
            ref_patterns = [
                r'(?:\[\d+\]|\[\w+\d+\]|\d+\.|\(\d+\))\s+[^[\d\(]+',  # Standard numbered references
                r'(?:[A-Z][a-z]+(?:,|\s+and|\s+et al\.)\s+\d{4})[^,\d]+'  # Author year format
            ]
            
            for pattern in ref_patterns:
                ref_matches = re.findall(pattern, ref_text)
                if ref_matches:
                    structured_text[key] = '\n'.join([r.strip() for r in ref_matches if r.strip()])
                    break
    
    # Adding figures and tables to the structured text
    if figures_tables:
        structured_text["Figures and Tables"] = '\n'.join([f"{ft_type}: {caption}" for ft_type, caption in figures_tables])
    
    # Formatting the structured text as a well-formatted plain text
    formatted_text = ""
    for section, content in structured_text.items():
        if section == "Preamble" and not content.strip():
            continue  # Skip empty preamble
        formatted_text += f"== {section} ==\n{content.strip()}\n\n"
        
    structured_text = extract_reference_section(structured_text, doc)
    
    # Validating the output to check if we atleast detected the key sections we'd expect in academic papers
    expected_sections = ["Abstract", "Introduction", "Conclusion"]
    missing_sections = [s for s in expected_sections if not any(es.lower() in s.lower() for es in structured_text.keys() for s in expected_sections)]
    
    if missing_sections:
        print(f"Warning: Could not find these expected sections: {', '.join(missing_sections)}")
    
    return formatted_text, structured_text


def extract_reference_section(structured_text: dict, doc: fitz.Document) -> dict:
    
    # Expanded section for reference
    REF_HEADERS = {
        'reference', 'references', 'bibliography', 'works cited',
        'literature cited', 'reference list', 'publications','R EFERENCES'
        'REFERENCE', 'REFERENCES', 'BIBLIOGRAPHY', 'WORKS CITED'
    }
    
    # Bibliographic entry patterns for regex
    BIBLIO_PATTERNS = [
        # Author-date styles
        r'^[A-Z][a-z]+,\s[A-Z]\.(?:\s[A-Z]\.)?\s\(\d{4}\)\.',  # Author, A. A. (2023).
        r'^[A-Z][a-z]+\s[A-Z][a-z]+,\s[A-Z]\.\s\(\d{4}\)\.',   # Author Name, A. (2023).
        
        # Numbered styles
        r'^\[\d+\]\s+[A-Z]',                                   # [1] Author...
        r'^\d+\.\s+[A-Z]',                                     # 1. Author...
        
        # Journal article patterns
        r'\bvol\.\s\d+,\spp?\.\s\d+-\d+',                      # vol. 12, p. 123-125
        r'\b\d+\(\d+\):\s\d+-\d+',                             # 12(3): 123-125
        
        # Book patterns
        r'\b[A-Z][a-z]+:\s[A-Z][a-z]+\s[Pp]ress',             # City: Publisher Press
        r'\b[A-Z][a-z]+\s[Pp]ress',                           # Publisher Press
        
        # Common bibliography elements
        r'\bdoi:\s?10\.\d+',                                   # doi: 10.xxxx
        r'\bhttps?://[^\s]+',                                  # URLs
        r'\bISBN\s[\d\-X]+',                                   # ISBN numbers
        r'\bRetrieved\sfrom'                                   # Retrieved from...
    ]
    
    
    ref_sections = {}
    for section, content in structured_text.items():
        section_lower = section.lower()
        
       
        if any(ref_header in section_lower for ref_header in REF_HEADERS):
            ref_sections[section] = content
            continue
            
        
        pattern_count = sum(len(re.findall(pattern, content)) for pattern in BIBLIO_PATTERNS)
        if pattern_count >= 3:
            ref_sections[section] = content
    

    for section, content in ref_sections.items():
        
        cleaned_content = re.sub(r'\s+', ' ', content)  
        
       
        split_success = False
        split_methods = [
            r'(?<=\])\s+(?=\[)',    # Between numbered refs [1] [2]
            r'\n\s*\d+\.\s+',       # Numbered list 1. 2.
            r'\n\s*[A-Z][a-z]+,',   # New line starting with Author Last,
            r'\n\s*[A-Z][a-z]+\s',  # New line starting with Author
            r'\n\s*•\s+',           # Bullet points
            r'\n\s*\*'              # Asterisk separators
        ]
        
        references = []
        for method in split_methods:
            if not references:  
                references = [r.strip() for r in re.split(method, cleaned_content) if r.strip()]
                split_success = len(references) > 3  
        
        # Format references consistently
        if split_success:
            formatted_refs = []
            for i, ref in enumerate(references, 1):
               
                ref = re.sub(r'^[\[\(]?\d+[\]\)]?\s*', '', ref)
                
                if ref and not ref[0].isupper():
                    ref = ref[0].upper() + ref[1:]
                formatted_refs.append(f"[{i}] {ref}")
            
            structured_text[section] = '\n'.join(formatted_refs)
    
    return structured_text


# Cached function to load embedding model (avoiding multiple loads)
@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Function to generate embeddings
def generate_embeddings(text):
    model = load_embedding_model()
    return model.encode(text, convert_to_numpy=True).astype('float32')

# Function to create or load FAISS index
def get_faiss_index():
    if os.path.exists("faiss_index.index"):
        return faiss.read_index("faiss_index.index")
    return None

# Function to save FAISS index
def save_faiss_index(index):
    faiss.write_index(index, "faiss_index.index")

# Function to load knowledge base
def load_knowledge_base():
    if os.path.exists("knowledge_base.json"):
        with open("knowledge_base.json", "r") as f:
            return json.load(f)
    return []

# Function to save knowledge base
def save_knowledge_base(knowledge_base):
    with open("knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f)

# Function to clear the session state and delete saved files
def restart_session():
    st.session_state.clear()  # Clear the session state
    if os.path.exists("faiss_index.index"):
        os.remove("faiss_index.index")  # Delete the FAISS index file
    if os.path.exists("knowledge_base.json"):
        os.remove("knowledge_base.json")  # Delete the knowledge base file
    st.rerun()  # Rerun the app to reset the UI

# Function to retrieve relevant information
def retrieve_relevant_info(query_embedding, index, knowledge_base, top_k=3):
    if index is None or index.ntotal == 0:
        return ["No relevant documents found."]

    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = []
    for i in indices[0]:
        if 0 <= i < len(knowledge_base):
            retrieved_docs.append(knowledge_base[i])
    
    return retrieved_docs if retrieved_docs else ["No relevant documents found."]

def augment_prompt_with_retrieved_info(text, relevant_docs, query=None):
    """
    Modifies prompt structure based on whether it is an initial summarization 
    or a follow-up question.
    """
    retrieved_info = "\n\n".join(relevant_docs)

    if query:
        # If it's a user question, don't force a summary format
        augmented_prompt = f"Context Information:\n{retrieved_info}\n\nUser Question: {query}\n\nAnswer:"
    else:
        # Default summarization format
        augmented_prompt = f"Original Text:\n{text}\n\nRetrieved Information:\n{retrieved_info}\n\nSummary:"
    
    return augmented_prompt


# Function to call Ollama API for summarization or follow-up questions
def call_ollama_api(prompt, text=None):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": f"{prompt}: {text}" if text else prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=3000)  # Increase timeout to 30s
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except requests.exceptions.RequestException as e:
        return f"Ollama API Error: {str(e)}"


# Streamlit UI
st.title("Research Paper Chatbot with RAG")
st.write("Upload a research paper (PDF) and chat with it!")

# Restart session button
if st.button("Restart Session"):
    restart_session()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = get_faiss_index()
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = load_knowledge_base()

# Ensure FAISS index is initialized
if st.session_state.faiss_index is None:
    st.session_state.faiss_index = faiss.IndexFlatL2(384)  # Default MiniLM embedding dimension

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="file_uploader")

if uploaded_file is not None:
    file_hash = generate_file_hash(uploaded_file)
    
    if file_hash not in st.session_state.processed_files:
        st.session_state.processed_files.add(file_hash)

        with st.spinner("Extracting text from PDF..."):
            text = extract_structured_text_from_pdf(uploaded_file)
            st.session_state.extracted_text = text
            st.session_state.knowledge_base.append(text)
            save_knowledge_base(st.session_state.knowledge_base)

        with st.spinner("Generating embeddings and setting up FAISS index..."):
            embeddings = generate_embeddings(text)
            embeddings = np.array([embeddings]).astype('float32')  # Ensure 2D shape for FAISS
            st.session_state.faiss_index.add(embeddings)
            save_faiss_index(st.session_state.faiss_index)

        with st.spinner("Generating summary with RAG..."):
            query_embedding = embeddings[0]
            relevant_docs = retrieve_relevant_info(query_embedding, st.session_state.faiss_index, st.session_state.knowledge_base, top_k=3)
            augmented_prompt = augment_prompt_with_retrieved_info(text, relevant_docs)
            summary = call_ollama_api(augmented_prompt)
            st.session_state.messages.append({"role": "assistant", "content": summary})
    
    else:
        st.warning("This file has already been processed.")

# Display extracted text preview
if "extracted_text" in st.session_state:
    with st.expander("Extracted Text Preview"):
        st.write(st.session_state.extracted_text[:1000] + "...")  # Show first 1000 characters

# Display FAISS status
if st.session_state.faiss_index.ntotal > 0:
    st.success(f"FAISS Index loaded with {st.session_state.faiss_index.ntotal} entries.")
else:
    st.warning("FAISS Index is empty. Upload a document to begin.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about the research paper..."):
    if "extracted_text" not in st.session_state:
        st.error("Please upload a PDF file first.")
    else:
        # Add user's question to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response to the question
        with st.spinner("Generating response..."):
            query_embedding = generate_embeddings(prompt)
            relevant_docs = retrieve_relevant_info(query_embedding, st.session_state.faiss_index, st.session_state.knowledge_base, top_k=3)
            
            # Pass the user question explicitly
            augmented_prompt = augment_prompt_with_retrieved_info(st.session_state.extracted_text, relevant_docs, query=prompt)
            response = call_ollama_api(prompt, augmented_prompt)

            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display AI's response
        with st.chat_message("assistant"):
            st.write(response)

