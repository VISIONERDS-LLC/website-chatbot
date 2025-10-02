import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_PATH = "faiss_index"


def process_project_document(content: str, source: str) -> List[Document]:
    documents = []
    
    # Find all project starts to correctly extract blocks
    # Using re.finditer to get start indices of all "PROJECT:" occurrences
    project_start_indices = [m.start() for m in re.finditer(r'(?i)PROJECT:', content)]
    
    project_blocks = []
    # If no 'PROJECT:' found but content exists, treat as one general project
    if not project_start_indices and content.strip():
        project_blocks.append(content)
    else:
        for i, start_idx in enumerate(project_start_indices):
            end_idx = project_start_indices[i+1] if i+1 < len(project_start_indices) else len(content)
            project_block_text = content[start_idx:end_idx].strip()
            if project_block_text:
                project_blocks.append(project_block_text)

    for project_block in project_blocks:
        lines = project_block.strip().split('\n')
        if not lines:
            continue
        
        # Extract project name. Assume first line or part of it is the project name.
        project_name_full = lines[0].strip()
        project_name_match = re.match(r'(?i)PROJECT:\s*([^\n]+)', project_name_full)
        if project_name_match:
            current_project = f"PROJECT: {project_name_match.group(1).strip()}"
            project_text = '\n'.join(lines[1:])
        else:
            # If "PROJECT:" isn't explicitly found in the first line of the block
            # (e.g., if the initial document text was treated as a single project block),
            # use the whole first line as the project name and the rest as text.
            current_project = f"PROJECT: {project_name_full}"
            project_text = '\n'.join(lines[1:])

        if not project_text.strip():
            continue
        
        # Expanded subsection patterns to cover all headings in the provided documents
        subsection_patterns = [
            r'Project Overview', r'Technical Architecture', r'Backend Architecture', r'Backend Architecture and Operations',
            r'Key Technical Implementations', r'Key Implementations',
            r'Technology Stack', r'Deployment', r'Integration', r'Data Integration System',
            r'Database Optimization and Matching Algorithm', r'Al-Powered Content Generation',
            r'Client Collaboration and Project Management', r'Patient and Doctor Management System',
            r'Appointment Scheduling System', r'Al-Powered Medication Recommendation Engine',
            r'Deployment and Integration Infrastructure', r'Scalable Al System Architecture',
            r'Containerized Al Model Management', r'Azure Cloud Infrastructure Implementation',
            r'Data Management and Storage Solutions', r'Performance Optimization with CDN',
            r'CI/CD and DevOps Implementation', r'Advanced Al Architecture with LangChain and LangGraph',
            r'Personalized Chatbot Capabilities', r'Modular Conversation Workflow Design',
            r'Persistent Memory and User Context Management', r'Intelligent Tools and Retrieval Systems',
            r'Proactive Communication and Intelligent Triggers', r'Multi-Platform Deployment Architecture',
            r'Security and Compliance Implementation', r'Real-Time Performance Optimization',
            r'Advanced Al Architecture with OpenAl and FastAPI', r'Intelligent Lesson Planning Capabilities',
            r'Modular Educational Workflow Design', r'Persistent Memory and Educational Context Management',
            r'Intelligent Tools and Retrieval Systems', r'Proactive Educational Communication and Intelligent Triggers',
            r'Multi-Platform Educational Architecture', r'Document Structure and RAG Integration Notes',
            r'Technology Stack Recommendation Module', r'Code Analysis and Explanation Module',
            r'Project Estimation and Resource Planning Module'
        ]
        
        # Corrected subsection_regex to avoid None values from re.split
        # This creates a single capturing group for the entire OR'd pattern
        combined_subsection_regex = '|'.join(subsection_patterns)
        parts = re.split(f'({combined_subsection_regex})', project_text, flags=re.IGNORECASE)
        
        current_subsection = "General"
        current_text = ""
        
        for part in parts:
            if part is None: # Explicitly skip None parts, though the regex fix should largely prevent this.
                continue
            part = part.strip()
            if not part: # Skip empty strings resulting from the split
                continue
            
            # Check if this part is a subsection header (case-insensitive full match)
            is_subsection_header = False
            for pattern in subsection_patterns:
                if re.fullmatch(pattern, part, re.IGNORECASE):
                    is_subsection_header = True
                    break
            
            if is_subsection_header:
                if current_text.strip(): # If there's content for the previous subsection, add it
                    chunks = split_text(current_text, max_length=800, overlap=200)
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "project": current_project,
                                "subsection": current_subsection,
                                "source": source
                            }
                        )
                        documents.append(doc)
                
                current_subsection = part # Update to the new subsection header
                current_text = "" # Reset text for the new subsection
            else:
                current_text += "\n" + part # Accumulate text for the current subsection
        
        # Add any remaining text from the last subsection
        if current_text.strip():
            chunks = split_text(current_text, max_length=800, overlap=200)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "project": current_project,
                        "subsection": current_subsection,
                        "source": source
                    }
                )
                documents.append(doc)
    
    return documents


def split_text(text: str, max_length: int = 800, overlap: int = 200) -> List[str]:
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end < len(text):
            # Try to break at sentence or paragraph boundary
            break_point = -1
            
            # Prefer sentence end
            last_period = text[start:end].rfind('. ')
            if last_period != -1 and last_period > start + max_length // 2: # Ensure break point is not too early
                break_point = last_period + 1
            
            # Fallback to newline if no good sentence end
            if break_point == -1:
                last_newline = text[start:end].rfind('\n')
                if last_newline != -1 and last_newline > start + max_length // 2: # Ensure break point is not too early
                    break_point = last_newline + 1
            
            # If no good break point, just cut at max_length
            if break_point == -1:
                end = start + max_length
            else:
                end = break_point
        
        chunks.append(text[start:end].strip())
        # Ensure the next start position doesn't go backwards or stay the same
        start = end - overlap 
        if start < (end - max_length): # Prevent excessive overlap, ensure progress
            start = end - (max_length // 4) # Adjust overlap if it's too large for small chunks
        if start < 0: start = 0 # Avoid negative start index
    
    return chunks


def build_faiss_index(pdf_paths: List[str]):
    """Build FAISS index from PDFs and save to disk"""
    
    print("=" * 60)
    print("Building FAISS Index")
    print("=" * 60)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    all_documents = []
    
    # Process each PDF
    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"  ERROR: File not found!")
            continue
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        full_text = "\n\n".join([page.page_content for page in pages])
        
        if re.search(r'(?i)PROJECT:', full_text): # Check for "PROJECT:" case-insensitively
            docs = process_project_document(full_text, pdf_path)
        else:
            # For documents without "PROJECT:", treat as a single document under its filename
            chunks = split_text(full_text, max_length=800, overlap=200)
            docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "project": os.path.basename(pdf_path).replace(".pdf", ""), # Use filename without .pdf
                        "subsection": "General",
                        "source": pdf_path
                    }
                )
                for chunk in chunks
            ]
        
        all_documents.extend(docs)
        print(f"  Created {len(docs)} chunks")
    
    print(f"\nTotal documents: {len(all_documents)}")
    
    if not all_documents:
        print("ERROR: No documents to index!")
        return
    
    print("\nCreating FAISS index...")
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    
    print(f"Saving index to: {FAISS_INDEX_PATH}/")
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    print("\n✓ FAISS index built and saved successfully!")
    
    print(f"\nIndex Statistics:")
    print(f"  Total vectors: {len(all_documents)}")
    print(f"  Saved to: {FAISS_INDEX_PATH}/")
    
    return vectorstore


def verify_index():
    """Verify the saved index works"""
    print("\n=== Verifying Index ===")
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    test_query = "what projects do you have"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"Test query: '{test_query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. Project: {doc.metadata.get('project', 'N/A')}")
        print(f"   Subsection: {doc.metadata.get('subsection', 'N/A')}")
        print(f"   Content: {doc.page_content[:100]}...")
        print()


def main():
    PDF_PATHS = [
        "./Detailed Project Documentation (1).pdf",
        "./Company Info.pdf",
    ]
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in environment")
        return
    

    for path in PDF_PATHS:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            print("Please update PDF_PATHS in the script")
            return
    
    vectorstore = build_faiss_index(PDF_PATHS)
    
    if vectorstore:
        verify_index()
        
        print("\n" + "=" * 60)
        print("✓ Build complete!")
        print("=" * 60)
        print(f"\nIndex saved to: {FAISS_INDEX_PATH}/")


if __name__ == "__main__":
    main()
