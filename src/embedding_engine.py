"""
Multi-Modal RAG: Embedding Engine
Generates embeddings for text and images using CLIP and BGE

Author: Rahul Vuppalapati
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image


class MultiModalEmbedding:
    """
    Generate embeddings for both text and images.
    
    Uses:
    - BGE-M3 for text embeddings (multilingual, high quality)
    - CLIP for image embeddings and cross-modal alignment
    """
    
    def __init__(
        self,
        text_model: str = "BAAI/bge-m3",
        clip_model: str = "ViT-B/32",
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load text embedding model
        print(f"🔄 Loading text model: {text_model}")
        self.text_encoder = SentenceTransformer(text_model, device=self.device)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        # Load CLIP for images and cross-modal
        print(f"🔄 Loading CLIP model: {clip_model}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_dim = 512  # CLIP ViT-B/32 dimension
        
        # Projection layer to align dimensions (optional)
        # Map text embeddings to CLIP space for joint retrieval
        self.projection = torch.nn.Linear(self.text_dim, self.clip_dim).to(self.device)
        
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embeddings.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embeddings array [N, D]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Get BGE embeddings
        embeddings = self.text_encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Project to CLIP space
        with torch.no_grad():
            embeddings = self.projection(embeddings)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Encode images to embeddings using CLIP.
        
        Args:
            images: Single PIL Image or list
            
        Returns:
            Embeddings array [N, 512]
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        embeddings = []
        
        for image in images:
            # Preprocess
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = F.normalize(image_features, p=2, dim=1)
                embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None
    ) -> np.ndarray:
        """
        Encode combined text + image query.
        
        Args:
            text: Optional text query
            image: Optional image query
            
        Returns:
            Combined embedding [1, D]
        """
        embeddings = []
        
        if text:
            text_emb = self.encode_text(text)
            embeddings.append(text_emb)
        
        if image:
            img_emb = self.encode_image(image)
            embeddings.append(img_emb)
        
        if not embeddings:
            raise ValueError("Must provide text or image")
        
        # Average embeddings
        combined = np.mean(np.vstack(embeddings), axis=0, keepdims=True)
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        
        return combined
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: [1, D] or [D]
            document_embeddings: [N, D]
            
        Returns:
            Similarities [N]
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity
        similarities = np.dot(document_embeddings, query_embedding.T).squeeze()
        
        return similarities


class VectorStore:
    """
    Simple FAISS-based vector store for multi-modal retrieval.
    """
    
    def __init__(self, dimension: int = 512):
        try:
            import faiss
            self.use_faiss = True
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
        except ImportError:
            print("⚠️ FAISS not available, using numpy fallback")
            self.use_faiss = False
            self.vectors = None
        
        self.dimension = dimension
        self.metadata = []
        
    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add vectors to store.
        
        Args:
            embeddings: [N, D] array
            metadata: List of metadata dicts for each vector
        """
        if self.use_faiss:
            self.index.add(embeddings.astype('float32'))
        else:
            if self.vectors is None:
                self.vectors = embeddings
            else:
                self.vectors = np.vstack([self.vectors, embeddings])
        
        self.metadata.extend(metadata)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: [1, D] or [D]
            k: Number of results
            
        Returns:
            List of {metadata, score} dicts
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if self.use_faiss:
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    results.append({
                        'metadata': self.metadata[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            return results
        else:
            # Numpy fallback
            similarities = np.dot(self.vectors, query_embedding.T).squeeze()
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_k_indices:
                results.append({
                    'metadata': self.metadata[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
            return results
    
    def save(self, path: str):
        """Save index and metadata."""
        if self.use_faiss:
            import faiss
            faiss.write_index(self.index, f"{path}.faiss")
        np.save(f"{path}.metadata.npy", self.metadata, allow_pickle=True)
    
    def load(self, path: str):
        """Load index and metadata."""
        if self.use_faiss:
            import faiss
            self.index = faiss.read_index(f"{path}.faiss")
        self.metadata = np.load(f"{path}.metadata.npy", allow_pickle=True).tolist()
