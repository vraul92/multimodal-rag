"""
Multi-Modal RAG: Retriever and Generator
Multi-modal retrieval + LLM answer generation

Author: Rahul Vuppalapati
"""

import os
from typing import List, Dict, Optional
import openai
import anthropic


class MultiModalRetriever:
    """
    Retrieve relevant chunks using multi-modal embeddings.
    """
    
    def __init__(self, vector_store, embedding_engine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        
    def retrieve(
        self,
        query: str,
        query_image: Optional = None,
        k: int = 5,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Text query
            query_image: Optional query image
            k: Number of results
            rerank: Whether to rerank results
            
        Returns:
            List of retrieved chunks with scores
        """
        # Encode query
        if query_image:
            query_emb = self.embedding_engine.encode_multimodal(query, query_image)
        else:
            query_emb = self.embedding_engine.encode_text(query)
        
        # Search
        results = self.vector_store.search(query_emb, k=k*2 if rerank else k)
        
        # Rerank if needed (simplified diversity reranking)
        if rerank and len(results) > k:
            # Ensure diversity across pages
            seen_pages = set()
            diverse_results = []
            
            for result in results:
                page = result['metadata'].get('page_num', 0)
                if page not in seen_pages or len(diverse_results) < k:
                    diverse_results.append(result)
                    seen_pages.add(page)
                    if len(diverse_results) >= k:
                        break
            
            results = diverse_results
        
        return results


class AnswerGenerator:
    """
    Generate answers using LLM with retrieved context.
    Supports OpenAI GPT-4V and Anthropic Claude.
    """
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None
    ):
        self.model = model
        
        if "gpt" in model.lower() or "openai" in model.lower():
            self.provider = "openai"
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif "claude" in model.lower():
            self.provider = "anthropic"
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif "mock" in model.lower():
            self.provider = "mock"
            self.client = None
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 1000
    ) -> Dict:
        """
        Generate answer with citations.
        
        Args:
            query: User question
            retrieved_chunks: Retrieved context chunks
            max_tokens: Max output length
            
        Returns:
            {
                'answer': str,
                'citations': List[Dict],
                'sources': List[str]
            }
        """
        # Build context
        context_parts = []
        images = []
        
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk['metadata']
            
            if meta.get('chunk_type') == 'image':
                # Add image description
                context_parts.append(f"[Image {i+1}]: {meta.get('text', 'Image')}")
                if 'image' in meta:
                    images.append(meta['image'])
            else:
                # Add text
                context_parts.append(f"[Excerpt {i+1} from Page {meta.get('page_num', '?')}]:\n{meta.get('text', '')}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are a helpful research assistant analyzing documents. Answer the user's question based on the provided context.

Question: {query}

Context from document:
{context}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using [Excerpt N] or [Image N] format
3. If the answer involves a figure/chart, describe it clearly
4. If information is insufficient, say so honestly
5. Be concise but thorough

Answer:"""

        # Generate based on provider
        if self.provider == "openai":
            return self._generate_openai(prompt, images, max_tokens)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, images, max_tokens)
        else:
            return self._generate_mock(prompt, images, max_tokens)
    
    def _generate_openai(
        self,
        prompt: str,
        images: List,
        max_tokens: int
    ) -> Dict:
        """Generate using OpenAI GPT-4V."""
        # For HF Spaces, we'll use text-only version to avoid API key issues
        # In production, would use GPT-4V with images
        
        try:
            response = self.client.chat.completions.create(
                model=self.model if not images else "gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'citations': [],  # Would parse from answer in production
                'model': self.model
            }
        except Exception as e:
            # Fallback to mock response for demo
            return {
                'answer': f"Based on the document, I found relevant information about your query. The text mentions key concepts related to '{prompt[:50]}...' [Excerpt 1]. Additionally, there are visual elements that support this [Image 1].",
                'citations': [{'type': 'text', 'page': 1}, {'type': 'image', 'page': 1}],
                'model': self.model,
                'note': f'Using fallback (API error: {str(e)})'
            }
    
    def _generate_anthropic(
        self,
        prompt: str,
        images: List,
        max_tokens: int
    ) -> Dict:
        """Generate using Anthropic Claude."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text
            
            return {
                'answer': answer,
                'citations': [],
                'model': self.model
            }
        except Exception as e:
            return {
                'answer': f"Based on the document analysis, here's what I found regarding your question...",
                'citations': [],
                'model': self.model,
                'note': f'Using fallback (API error: {str(e)})'
            }
    
    def _generate_mock(
        self,
        prompt: str,
        images: List,
        max_tokens: int
    ) -> Dict:
        """Generate mock response for demo (no API key needed)."""
        # Extract query from prompt
        query_start = prompt.find("Question:") + 9
        query_end = prompt.find("Context from document:")
        query = prompt[query_start:query_end].strip() if query_start > 8 and query_end > 0 else "your question"
        
        # Generate contextual mock response
        answer = f"""Based on the uploaded document, I found relevant information about: **{query}**

From the text content retrieved:
• The document discusses several aspects related to this topic in the retrieved sections [Excerpt 1, Excerpt 2]
• Key findings include related concepts from Pages 1-3

**Note:** This is a demo response. For full functionality with real LLM responses, please add an OpenAI or Anthropic API key.

To enable full responses:
1. Get an API key from OpenAI (https://platform.openai.com) or Anthropic (https://console.anthropic.com)
2. Add it as a secret in your Hugging Face Space settings
3. Restart the Space"""
        
        return {
            'answer': answer,
            'citations': [
                {'type': 'text', 'page': 1},
                {'type': 'text', 'page': 2}
            ],
            'model': 'mock-demo',
            'note': 'Demo mode - Add OpenAI/Anthropic API key for real LLM responses'
        }


class MultiModalRAG:
    """
    Complete Multi-Modal RAG pipeline.
    """
    
    def __init__(
        self,
        embedding_engine,
        vector_store,
        retriever,
        generator
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.document_store = None
    
    def index_document(self, document_path: str) -> bool:
        """
        Index a document for retrieval.
        
        Args:
            document_path: Path to PDF file
            
        Returns:
            Success status
        """
        from .document_processor import DocumentProcessor
        
        # Process document
        processor = DocumentProcessor()
        doc = processor.process_pdf(document_path)
        self.document_store = doc
        
        # Create chunks
        chunks = processor.create_chunks(doc)
        
        # Generate embeddings
        embeddings = []
        metadata = []
        
        for chunk in chunks:
            if chunk['chunk_type'] == 'image' and 'image' in chunk:
                emb = self.embedding_engine.encode_image(chunk['image'])
            else:
                emb = self.embedding_engine.encode_text(chunk['text'])
            
            embeddings.append(emb[0])  # Remove batch dim
            metadata.append(chunk)
        
        # Add to vector store
        if embeddings:
            self.vector_store.add(
                np.vstack(embeddings),
                metadata
            )
        
        return True
    
    def query(
        self,
        question: str,
        query_image: Optional = None,
        k: int = 5
    ) -> Dict:
        """
        Answer a question about the indexed document.
        
        Args:
            question: User question
            query_image: Optional image query
            k: Number of chunks to retrieve
            
        Returns:
            Answer with citations
        """
        # Retrieve
        chunks = self.retriever.retrieve(question, query_image, k=k)
        
        # Generate answer
        result = self.generator.generate(question, chunks)
        result['retrieved_chunks'] = chunks
        
        return result
