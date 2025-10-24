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


def create_company_overview_chunks(content: str, source: str) -> List[Document]:
    """Create chunks specifically for company overview information"""
    
    # Company overview chunk
    company_section = re.search(r'1\.\s*Company Overview.*?(?=2\.|$)', content, re.DOTALL | re.IGNORECASE)
    if company_section:
        overview_text = company_section.group(0)
        # Add searchable keywords
        enhanced_overview = f"Company Overview - Visionerds:\n{overview_text}\n\nKeywords: company information, about us, what we do, services, visionerds"
        
        yield Document(
            page_content=enhanced_overview,
            metadata={
                "source": source,
                "content_type": "company_overview",
                "section": "Company Overview",
                "keywords": "company, about, visionerds, overview, services"
            }
        )
    
    # Services section
    services_section = re.search(r'2\.\s*Core Services.*?(?=3\.|$)', content, re.DOTALL | re.IGNORECASE)
    if services_section:
        services_text = services_section.group(0)
        enhanced_services = f"Company Services - Visionerds:\n{services_text}\n\nKeywords: services, what we offer, capabilities, ai ml, development, custom software"
        
        yield Document(
            page_content=enhanced_services,
            metadata={
                "source": source,
                "content_type": "services",
                "section": "Core Services",
                "keywords": "services, ai, ml, development, custom software, mobile, web"
            }
        )
    
    # Technology stack
    tech_section = re.search(r'5\.\s*Typical Tech Stack.*?(?=6\.|$)', content, re.DOTALL | re.IGNORECASE)
    if tech_section:
        tech_text = tech_section.group(0)
        enhanced_tech = f"Company Technology Stack:\n{tech_text}\n\nKeywords: technology, tech stack, technologies we use, frameworks, programming languages"
        
        yield Document(
            page_content=enhanced_tech,
            metadata={
                "source": source,
                "content_type": "technology",
                "section": "Tech Stack",
                "keywords": "technology, tech stack, react, python, node, aws, azure"
            }
        )
    
    # Project highlights
    highlights_section = re.search(r'6\.\s*Project Highlights.*?(?=7\.|$)', content, re.DOTALL | re.IGNORECASE)
    if highlights_section:
        highlights_text = highlights_section.group(0)
        enhanced_highlights = f"Company Project Highlights:\n{highlights_text}\n\nKeywords: projects, project examples, what we built, case studies, achievements"
        
        yield Document(
            page_content=enhanced_highlights,
            metadata={
                "source": source,
                "content_type": "project_highlights",
                "section": "Project Highlights",
                "keywords": "projects, examples, case studies, achievements, clients"
            }
        )


def create_project_chunks(content: str, source: str) -> List[Document]:
    """Create chunks for individual projects"""
    
    # Find all projects
    project_matches = list(re.finditer(r'PROJECT:\s*([^\n]+)', content, re.IGNORECASE))
    
    for i, match in enumerate(project_matches):
        project_name = match.group(1).strip()
        start_pos = match.start()
        
        # Find end position (next project or end of document)
        if i + 1 < len(project_matches):
            end_pos = project_matches[i + 1].start()
        else:
            end_pos = len(content)
        
        project_content = content[start_pos:end_pos].strip()
        
        # Create project overview chunk
        overview_match = re.search(r'Project Overview(.*?)(?=\n[A-Z][a-z]|\n[A-Z][A-Z]|$)', project_content, re.DOTALL)
        if overview_match:
            overview_text = overview_match.group(1).strip()
            enhanced_overview = f"Project: {project_name}\n\nProject Overview:\n{overview_text}\n\nKeywords: {project_name.lower()}, project, what we built, project description"
            
            yield Document(
                page_content=enhanced_overview,
                metadata={
                    "source": source,
                    "content_type": "project_overview",
                    "project_name": project_name,
                    "section": "Project Overview",
                    "keywords": f"{project_name.lower()}, project, overview, description"
                }
            )
        
        # Create technical details chunk
        tech_sections = re.findall(
            r'(Technical Architecture|Technology Stack|Key Technical Implementations|Backend Architecture.*?|AI Architecture.*?)(.*?)(?=\n[A-Z][a-z]|\n[A-Z][A-Z]|$)', 
            project_content, 
            re.DOTALL | re.IGNORECASE
        )
        
        if tech_sections:
            tech_content = ""
            for section_name, section_content in tech_sections:
                tech_content += f"\n{section_name}:\n{section_content.strip()}\n"
            
            enhanced_tech = f"Project: {project_name}\n\nTechnical Details:{tech_content}\n\nKeywords: {project_name.lower()}, technical details, architecture, implementation, technology"
            
            yield Document(
                page_content=enhanced_tech,
                metadata={
                    "source": source,
                    "content_type": "project_technical",
                    "project_name": project_name,
                    "section": "Technical Details",
                    "keywords": f"{project_name.lower()}, technical, architecture, implementation"
                }
            )
        
        # Create full project chunk for comprehensive queries
        enhanced_full = f"Complete Project Details: {project_name}\n\n{project_content}\n\nKeywords: {project_name.lower()}, full project details, complete information"
        
        # Split if too long
        if len(enhanced_full) > 2000:
            chunks = split_long_content(enhanced_full, 1800, 400)
            for j, chunk in enumerate(chunks):
                yield Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "content_type": "project_full",
                        "project_name": project_name,
                        "section": f"Full Details Part {j+1}",
                        "chunk_index": j,
                        "keywords": f"{project_name.lower()}, complete, full details"
                    }
                )
        else:
            yield Document(
                page_content=enhanced_full,
                metadata={
                    "source": source,
                    "content_type": "project_full",
                    "project_name": project_name,
                    "section": "Full Details",
                    "keywords": f"{project_name.lower()}, complete, full details"
                }
            )


def create_project_list_chunk(content: str, source: str) -> Document:
    """Create a special chunk for project listing queries"""
    
    project_names = re.findall(r'PROJECT:\s*([^\n]+)', content, re.IGNORECASE)
    
    if project_names:
        project_list = "Our Projects - Complete List:\n\n"
        for i, name in enumerate(project_names, 1):
            project_list += f"{i}. {name.strip()}\n"
        
        project_list += f"\n\nWe have completed {len(project_names)} major projects including: "
        project_list += ", ".join([name.strip() for name in project_names])
        project_list += "\n\nKeywords: projects, project list, what projects, list of projects, our work, portfolio"
        
        return Document(
            page_content=project_list,
            metadata={
                "source": source,
                "content_type": "project_list",
                "section": "Project List",
                "total_projects": len(project_names),
                "keywords": "projects, list, portfolio, our work, what projects"
            }
        )


def split_long_content(content: str, max_length: int, overlap: int) -> List[str]:
    """Split long content while preserving structure"""
    if len(content) <= max_length:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + max_length
        
        if end < len(content):
            # Find good break point
            break_point = find_break_point(content, start, end)
            if break_point > start:
                end = break_point
        
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(end - overlap, start + max_length // 2)
        if start >= len(content):
            break
    
    return chunks


def find_break_point(content: str, start: int, max_end: int) -> int:
    """Find optimal break point"""
    # Look for section breaks
    for break_char in ['\n\n', '\n•', '\n-', '. ', '?\n', '!\n']:
        pos = content[start:max_end].rfind(break_char)
        if pos != -1 and pos > (max_end - start) // 2:
            return start + pos + len(break_char)
    return max_end


def build_faiss_index(pdf_paths: List[str]):
    """Build optimized FAISS index for company and project information"""
    
    print("=" * 60)
    print("Building Optimized FAISS Index for Company & Projects")
    print("=" * 60)
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    all_documents = []
    
    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"  ERROR: File not found!")
            continue
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n\n".join([page.page_content for page in pages])
        
        filename = os.path.basename(pdf_path)
        
        if "company" in filename.lower() or "info" in filename.lower():
            # Process company information
            print("  Processing as company information document")
            company_docs = list(create_company_overview_chunks(full_text, pdf_path))
            all_documents.extend(company_docs)
            print(f"  Created {len(company_docs)} company information chunks")
            
        elif "project" in filename.lower() or "detailed" in filename.lower():
            # Process project information
            print("  Processing as project documentation")
            
            # Create project list chunk
            project_list_doc = create_project_list_chunk(full_text, pdf_path)
            if project_list_doc:
                all_documents.append(project_list_doc)
                print("  Created project list chunk")
            
            # Create individual project chunks
            project_docs = list(create_project_chunks(full_text, pdf_path))
            all_documents.extend(project_docs)
            print(f"  Created {len(project_docs)} project-specific chunks")
        
        else:
            # Generic processing
            print("  Processing as generic document")
            chunks = split_long_content(full_text, 1000, 200)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path,
                        "filename": filename,
                        "chunk_index": i,
                        "content_type": "general"
                    }
                )
                all_documents.append(doc)
            print(f"  Created {len(chunks)} generic chunks")
    
    print(f"\nTotal documents: {len(all_documents)}")
    
    if not all_documents:
        print("ERROR: No documents to index!")
        return
    
    print("\nCreating FAISS index...")
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    
    print(f"Saving index to: {FAISS_INDEX_PATH}/")
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    print("\n✓ Optimized FAISS index built successfully!")
    
    # Show content type distribution
    content_types = {}
    for doc in all_documents:
        ct = doc.metadata.get('content_type', 'unknown')
        content_types[ct] = content_types.get(ct, 0) + 1
    
    print(f"\nContent type distribution:")
    for content_type, count in content_types.items():
        print(f"  {content_type}: {count} chunks")
    
    return vectorstore


def verify_index():
    """Test the index with typical queries"""
    print("\n=== Verifying Index with Test Queries ===")
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    test_queries = [
        "list your projects",
        "what projects do you have",
        "tell me about the company",
        "what does visionerds do",
        "technical details of ramped project",
        "what technologies do you use",
        "ai projects you built"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            content_type = doc.metadata.get('content_type', 'unknown')
            section = doc.metadata.get('section', 'N/A')
            project = doc.metadata.get('project_name', 'N/A')
            
            print(f"  {i}. Type: {content_type} | Section: {section} | Project: {project}")
            print(f"     Content: {doc.page_content[:100]}...")


def main():
    PDF_PATHS = [
        "./Company Info.pdf",
        "./Detailed Project Documentation.pdf",
    ]
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in environment")
        return
    
    # Check if files exist
    for path in PDF_PATHS:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            print("Available files:")
            for file in os.listdir("."):
                if file.endswith(".pdf"):
                    print(f"  - {file}")
            return
    
    vectorstore = build_faiss_index(PDF_PATHS)
    
    if vectorstore:
        verify_index()
        
        print("\n" + "=" * 60)
        print("✓ Optimized index complete!")
        print("=" * 60)
        print("\nRecommended queries to test:")
        print("- 'list your projects' or 'what projects do you have'")
        print("- 'tell me about the company' or 'what does visionerds do'")
        print("- 'technical details of [project name]'")
        print("- 'what technologies do you use'")


if __name__ == "__main__":
    main()
