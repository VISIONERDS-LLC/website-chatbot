import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_class.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_PATH = "faiss_index"

client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info(f"✓ FAISS index loaded from {FAISS_INDEX_PATH}")
except Exception as e:
    logger.warning(f"⚠ Warning: Could not load FAISS index: {e}")
    logger.warning(f"  Run rebuild_faiss.py first to build the index")
    vectorstore = None


def rephrase_query(query: str, history: Optional[List[Dict]] = None) -> str:
    """
    Rephrase the user query into a standalone, clear search query.
    Handles follow-up questions and vague references.
    """
    logger.info(f"[REPHRASE] Original query: '{query}'")

    if not history or len(history) == 0:
        logger.info("[REPHRASE] No history found, using original query")
        return query
    
    # Get last 3 exchanges for context
    recent_history = history[-3:]
    history_context = ""
    for msg in recent_history:
        history_context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
    
    system_prompt = """You are a query rephrasing assistant. Your job is to convert follow-up questions into standalone, clear search queries.

Rules:
1. If the query is already clear and standalone, return it as-is
2. If it references "it", "that", "the first one", "more about this", etc., replace with specific entities from conversation history
3. Keep the rephrased query concise (1-2 sentences max)
4. Maintain the original intent and question type
5. Output ONLY the rephrased query, nothing else

Examples:
- "Tell me more about it" → "Tell me more about [specific project/topic from history]"
- "What technologies does it use?" → "What technologies does [specific project] use?"
- "How does that work?" → "How does [specific feature/concept] work?"
"""

    user_prompt = f"""Conversation history:
{history_context}

Current query: {query}

Rephrase this query into a clear, standalone search query:"""

    try:
        logger.info("[REPHRASE] Calling OpenAI to rephrase query...")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_completion_tokens=100
        )

        rephrased = completion.choices[0].message.content.strip()
        logger.info(f"[REPHRASE] ✓ Rephrased query: '{rephrased}'")
        logger.info(f"[REPHRASE] Tokens used: {completion.usage.total_tokens} (prompt: {completion.usage.prompt_tokens}, completion: {completion.usage.completion_tokens})")
        return rephrased

    except Exception as e:
        logger.warning(f"[REPHRASE] ⚠ Failed: {e}. Using original query.")
        return query


def search_docs(query: str, top_k: int = 3, project_filter: Optional[str] = None) -> List[Dict]:
    if vectorstore is None:
        logger.error("[FAISS] ✗ FAISS index not loaded")
        raise Exception("FAISS index not loaded. Run rebuild_faiss.py first.")

    logger.info(f"[FAISS] Searching for query: '{query}' (top_k={top_k})")
    if project_filter:
        logger.info(f"[FAISS] Applying project filter: {project_filter}")

    if project_filter:
        results = vectorstore.similarity_search(query, k=top_k * 3)
        filtered = [doc for doc in results if doc.metadata.get('project') == project_filter]
        results = filtered[:top_k]
    else:
        results = vectorstore.similarity_search(query, k=top_k)

    logger.info(f"[FAISS] ✓ Retrieved {len(results)} documents from index")

    docs = []
    for idx, doc in enumerate(results, 1):
        doc_info = {
            "content": doc.page_content,
            "project": doc.metadata.get("project", "Unknown"),
            "subsection": doc.metadata.get("subsection", "General"),
            "source": doc.metadata.get("source", "Unknown")
        }
        docs.append(doc_info)
        logger.info(f"[FAISS] Document {idx}:")
        logger.info(f"  - Project: {doc_info['project']}")
        logger.info(f"  - Subsection: {doc_info['subsection']}")
        logger.info(f"  - Source: {doc_info['source']}")
        logger.info(f"  - Content preview: {doc_info['content'][:150]}...")

    return docs


def format_history(history: List[Dict]) -> str:
    if not history:
        return ""
    
    recent = history[-10:]
    formatted = "\n\nPrevious conversation:\n"
    for msg in recent:
        formatted += f"User: {msg['user']}\n"
        formatted += f"Assistant: {msg['assistant']}\n"
    
    return formatted


def query_rag(
    query: str,
    history: Optional[List[Dict]] = None,
    project_filter: Optional[str] = None,
    model: str = "gpt-4.1-nano",
    temperature: float = 0.1,
    enable_rephrasing: bool = True
) -> Dict:

    logger.info("=" * 80)
    logger.info("[QUERY_RAG] Starting RAG query pipeline")
    logger.info(f"[QUERY_RAG] Original query: '{query}'")
    logger.info(f"[QUERY_RAG] Model: {model}")
    logger.info(f"[QUERY_RAG] Temperature: {temperature}")

    # Rephrase query if history exists and rephrasing is enabled
    original_query = query
    if enable_rephrasing and history:
        query = rephrase_query(query, history)

    docs = search_docs(query, project_filter=project_filter)

    context = "\n\n".join([
        f"[Source: {doc['project']} - {doc['subsection']}]\n{doc['content']}"
        for doc in docs
    ])

    logger.info(f"[QUERY_RAG] Context length: {len(context)} characters")

    history_text = format_history(history) if history else ""
    
    system_prompt = """
You are a knowledgeable and friendly assistant representing our service-based IT company. 
Your primary role is to provide information about our company, projects, services, and technical capabilities using the provided documentation.

WHAT YOU CAN DO:
- Answer questions about our company's projects and services
- Explain our technology stack and technical capabilities
- Describe our development processes and methodologies
- Share information about industries we serve
- Discuss our past project experiences and achievements
- Provide details about our team's expertise

WHAT YOU CANNOT DO:
- Book calls, schedule meetings, or perform any booking/scheduling actions
- Access external systems or perform real-time actions
- Provide contact information or direct people to specific team members
- Make commitments on behalf of the company
- Discuss pricing, quotes, or contractual details
- Perform agentic tasks or external integrations

RESPONSE GUIDELINES:
- Speak from the company's perspective using "we" and "our"
- Be natural, conversational, and professional
- If asked about something outside your scope, politely explain what you can help with instead
- For booking/scheduling requests, suggest they contact the company through official channels
- Never make up information - only use what's in the provided context
- Avoid meta phrases like "based on the context" or "according to the documentation"

Your tone should be helpful and professional while staying focused on sharing company information.
"""

    
    user_prompt = f"""{history_text}

Context from documentation:
{context}

Current question: {original_query}

Please answer the question based on the context provided above. If the conversation history is relevant, use it to understand follow-up questions."""

    logger.info("[OPENAI] Sending request to OpenAI API...")
    logger.info(f"[OPENAI] System prompt length: {len(system_prompt)} characters")
    logger.info(f"[OPENAI] User prompt length: {len(user_prompt)} characters")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=4000
    )

    answer = completion.choices[0].message.content

    logger.info("=" * 80)
    logger.info("[OPENAI] ✓ COMPLETION RECEIVED FROM OPENAI")
    logger.info("=" * 80)
    logger.info(f"[OPENAI] Model used: {completion.model}")
    logger.info(f"[OPENAI] Finish reason: {completion.choices[0].finish_reason}")
    logger.info(f"[OPENAI] Token usage:")
    logger.info(f"  - Prompt tokens: {completion.usage.prompt_tokens}")
    logger.info(f"  - Completion tokens: {completion.usage.completion_tokens}")
    logger.info(f"  - Total tokens: {completion.usage.total_tokens}")
    logger.info("=" * 80)
    logger.info("[OPENAI] FULL COMPLETION CONTENT:")
    logger.info("=" * 80)
    logger.info(answer)
    logger.info("=" * 80)

    response = {
        "answer": answer,
        "sources": docs,
        "usage": {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        },
        "model": completion.model
    }

    # Include rephrased query info if it was used
    if enable_rephrasing and query != original_query:
        response["rephrased_query"] = query

    logger.info("[QUERY_RAG] ✓ RAG query pipeline completed successfully")
    logger.info("=" * 80)

    return response


def query_rag_streaming(
    query: str,
    history: Optional[List[Dict]] = None,
    project_filter: Optional[str] = None,
    model: str = "gpt-5-nano",
    enable_rephrasing: bool = True
):

    logger.info("=" * 80)
    logger.info("[QUERY_RAG_STREAMING] Starting streaming RAG query pipeline")
    logger.info(f"[QUERY_RAG_STREAMING] Query: '{query}'")
    logger.info(f"[QUERY_RAG_STREAMING] Model: {model}")

    # Rephrase query if history exists
    if enable_rephrasing and history:
        query = rephrase_query(query, history)

    docs = search_docs(query, project_filter=project_filter)
    logger.info("[OPENAI] Starting streaming response...")
    
    context = "\n\n".join([
        f"[Source: {doc['project']} - {doc['subsection']}]\n{doc['content']}"
        for doc in docs
    ])
    
    history_text = format_history(history) if history else ""
    
    system_prompt = """You are a knowledgeable and friendly assistant representing our service-based IT company. 
You answer questions about our company and its projects using the provided documentation.

Your primary role is to provide information about our company, projects, services, and technical capabilities.
You cannot book calls, schedule meetings, or perform any agentic actions - only provide company information."""
    
    user_prompt = f"""{history_text}

Context: {context}

Question: {query}

Answer based on the context provided."""
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def reload_index():
    """Reload FAISS index (useful after rebuilding)"""
    global vectorstore
    logger.info("[RELOAD] Attempting to reload FAISS index...")
    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"[RELOAD] ✓ FAISS index reloaded successfully from {FAISS_INDEX_PATH}")
        return True
    except Exception as e:
        logger.error(f"[RELOAD] ✗ Error reloading index: {e}")
        return False


if __name__ == "__main__":
    print("Testing FAISS RAG System with Query Rephrasing...\n")
    
    if vectorstore is None:
        print("ERROR: FAISS index not found!")
        print("Run: python rebuild_faiss.py")
        exit(1)
    
    # Test 1: Initial query (no rephrasing needed)
    print("=" * 60)
    print("Test 1: Initial Query")
    print("=" * 60)
    result = query_rag("Tell me about your projects?")
    print(f"Answer: {result['answer']}\n")
    print(f"Sources: {len(result['sources'])} documents")
    print(f"Tokens used: {result['usage']['total_tokens']}\n")
    
    # Test 2: Follow-up query (should trigger rephrasing)
    print("=" * 60)
    print("Test 2: Follow-up Query (Vague)")
    print("=" * 60)
    history = [
        {"user": "Tell me about Ramped?", "assistant": result['answer']}
    ]
    
    result2 = query_rag("Tell me more about it", history=history)
    if "rephrased_query" in result2:
        print(f"Rephrased query: {result2['rephrased_query']}")
    print(f"Answer: {result2['answer']}\n")
    
    # Test 3: Another follow-up
    print("=" * 60)
    print("Test 3: Second Follow-up")
    print("=" * 60)
    history.append({"user": "Tell me more about it", "assistant": result2['answer']})
    
    result3 = query_rag("What technologies does it use?", history=history)
    if "rephrased_query" in result3:
        print(f"Rephrased query: {result3['rephrased_query']}")
    print(f"Answer: {result3['answer']}\n")
    
    # Test 4: Streaming with rephrasing
    print("=" * 60)
    print("Test 4: Streaming Response")
    print("=" * 60)
    print("Streaming response for 'What else can you tell me?':")
    for chunk in query_rag_streaming("What else can you tell me?", history=history):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Test 5: Test irrelevant query handling
    print("=" * 60)
    print("Test 5: Irrelevant Query Test")
    print("=" * 60)
    result4 = query_rag("Book me a call with your team")
    print(f"Answer: {result4['answer']}\n")
