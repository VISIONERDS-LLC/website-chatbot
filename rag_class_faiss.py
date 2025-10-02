import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

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
    print(f"✓ FAISS index loaded from {FAISS_INDEX_PATH}")
except Exception as e:
    print(f"⚠ Warning: Could not load FAISS index: {e}")
    print(f"  Run rebuild_faiss.py first to build the index")
    vectorstore = None


def rephrase_query(query: str, history: Optional[List[Dict]] = None) -> str:
    """
    Rephrase the user query into a standalone, clear search query.
    Handles follow-up questions and vague references.
    """
    if not history or len(history) == 0:
        # No history, return original query
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
        print(f"🔄 Query rephrased: '{query}' → '{rephrased}'")
        return rephrased
        
    except Exception as e:
        print(f"⚠ Query rephrasing failed: {e}. Using original query.")
        return query


def search_docs(query: str, top_k: int = 3, project_filter: Optional[str] = None) -> List[Dict]:
    if vectorstore is None:
        raise Exception("FAISS index not loaded. Run rebuild_faiss.py first.")
    
    if project_filter:
        results = vectorstore.similarity_search(query, k=top_k * 3)
        filtered = [doc for doc in results if doc.metadata.get('project') == project_filter]
        results = filtered[:top_k]
    else:
        results = vectorstore.similarity_search(query, k=top_k)
    
    docs = []
    for doc in results:
        docs.append({
            "content": doc.page_content,
            "project": doc.metadata.get("project", "Unknown"),
            "subsection": doc.metadata.get("subsection", "General"),
            "source": doc.metadata.get("source", "Unknown")
        })
    
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
    
    # Rephrase query if history exists and rephrasing is enabled
    original_query = query
    if enable_rephrasing and history:
        query = rephrase_query(query, history)
    
    docs = search_docs(query, project_filter=project_filter)

    print(f"📄 Retrieved {len(docs)} documents")

    print(docs)
    
    context = "\n\n".join([
        f"[Source: {doc['project']} - {doc['subsection']}]\n{doc['content']}"
        for doc in docs
    ])
    
    history_text = format_history(history) if history else ""
    
    system_prompt = """
You are a knowledgeable and friendly assistant representing our service-based IT company. 
You answer questions about our company and its projects using the provided documentation.

Your tone should be:
- Natural, human-like, and conversational (as if you're part of the team)
- Clear, concise, and confident
- Helpful and professional

When responding:
- Speak from the company's perspective using "we" and "our"
- Avoid meta phrases like "based on the context", "described in the documentation", etc.
- Present information as if you're directly telling the user about the company
- If you don't know something, say so honestly that I dont know— never make up information
"""

    
    user_prompt = f"""{history_text}

Context from documentation:
{context}

Current question: {original_query}

Please answer the question based on the context provided above. If the conversation history is relevant, use it to understand follow-up questions."""
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=1000
    )
    
    answer = completion.choices[0].message.content

    print(answer,'here')
    
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
    
    return response


def query_rag_streaming(
    query: str,
    history: Optional[List[Dict]] = None,
    project_filter: Optional[str] = None,
    model: str = "gpt-5-nano",
    enable_rephrasing: bool = True
):
    
    # Rephrase query if history exists
    if enable_rephrasing and history:
        query = rephrase_query(query, history)
    
    docs = search_docs(query, project_filter=project_filter)
    
    context = "\n\n".join([
        f"[Source: {doc['project']} - {doc['subsection']}]\n{doc['content']}"
        for doc in docs
    ])
    
    history_text = format_history(history) if history else ""
    
    system_prompt = """You are a knowledgeable and friendly assistant representing our service-based IT company. 
You answer questions about our company and its projects using the provided documentation."""
    
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
    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ FAISS index reloaded")
        return True
    except Exception as e:
        print(f"Error reloading index: {e}")
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
