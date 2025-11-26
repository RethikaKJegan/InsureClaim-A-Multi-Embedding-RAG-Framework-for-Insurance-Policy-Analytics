# Week 2: Embedding Generation & FAISS Vector Database
# File: vector_store.py

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import json

# Required installations:
# pip install sentence-transformers faiss-cpu numpy

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class EmbeddedChunk:
    """Represents a chunk with its embedding vector"""
    chunk_id: int
    text: str
    embedding: np.ndarray
    source_file: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self, include_embedding=False):
        """Convert to dictionary for serialization"""
        data = {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'source_file': self.source_file,
            'page_number': self.page_number,
            'section_title': self.section_title,
            'metadata': self.metadata
        }
        if include_embedding:
            data['embedding'] = self.embedding.tolist()
        return data


class EmbeddingGenerator:
    """Generates semantic embeddings from text using Sentence Transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Available models:
        - 'all-MiniLM-L6-v2': Fast, 384 dimensions (recommended for starting)
        - 'all-mpnet-base-v2': Better quality, 768 dimensions
        - 'multi-qa-MiniLM-L6-cos-v1': Optimized for Q&A
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_chunks(self, chunks: List, show_progress: bool = True) -> List[EmbeddedChunk]:
        """
        Generate embeddings for DocumentChunk objects
        
        Args:
            chunks: List of DocumentChunk objects from Week 1
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddedChunk objects with embeddings
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.embed_batch(texts, show_progress=show_progress)
        
        # Create EmbeddedChunk objects
        embedded_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                metadata={
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'length': len(chunk.text)
                }
            )
            embedded_chunks.append(embedded_chunk)
        
        print(f"✅ Generated {len(embedded_chunks)} embeddings")
        return embedded_chunks


class FAISSVectorStore:
    """FAISS-based vector database for semantic search"""
    
    def __init__(self, embedding_dim: int = 384, index_type: str = 'flat'):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index
                - 'flat': Exact search (recommended for <100k vectors)
                - 'ivf': Faster approximate search (for larger datasets)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunks_metadata = []  # Store chunk metadata separately
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == 'flat':
            # L2 (Euclidean) distance - exact search
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            print(f"✅ Created FAISS Flat index (dim={self.embedding_dim})")
        
        elif self.index_type == 'ivf':
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            n_clusters = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            print(f"✅ Created FAISS IVF index (dim={self.embedding_dim}, clusters={n_clusters})")
            print("⚠️  Note: IVF index needs training before use")
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_embeddings(self, embedded_chunks: List[EmbeddedChunk]):
        """Add embedded chunks to the index"""
        if not embedded_chunks:
            print("⚠️  No chunks to add")
            return
        
        # Extract embeddings as numpy array
        embeddings = np.array([chunk.embedding for chunk in embedded_chunks])
        
        # Ensure correct shape
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        # Train IVF index if needed
        if self.index_type == 'ivf' and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)
            print("✅ Index trained")
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        for chunk in embedded_chunks:
            self.chunks_metadata.append(chunk.to_dict(include_embedding=False))
        
        print(f"✅ Added {len(embedded_chunks)} vectors to index")
        print(f"   Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of dictionaries with chunk data and similarity scores
        """
        if self.index.ntotal == 0:
            print("⚠️  Index is empty")
            return []
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks_metadata):  # Valid index
                result = {
                    'rank': i + 1,
                    'score': float(distance),
                    'similarity': self._distance_to_similarity(distance),
                    **self.chunks_metadata[idx]
                }
                results.append(result)
        
        return results
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert L2 distance to similarity score (0-1)"""
        # Lower distance = higher similarity
        # Using exponential decay: similarity = e^(-distance)
        return np.exp(-distance / 10)
    
    def save(self, save_path: str):
        """Save index and metadata to disk"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = save_dir / "chunks_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
        
        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'total_vectors': self.index.ntotal
        }
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Saved vector store to {save_path}")
    
    def load(self, load_path: str):
        """Load index and metadata from disk"""
        load_dir = Path(load_path)
        
        # Load config
        config_path = load_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load FAISS index
        index_path = load_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = load_dir / "chunks_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.chunks_metadata = pickle.load(f)
        
        self.embedding_dim = config['embedding_dim']
        self.index_type = config['index_type']
        
        print(f"✅ Loaded vector store from {load_path}")
        print(f"   Total vectors: {self.index.ntotal}")


class SemanticSearchEngine:
    """High-level interface for semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_type: str = 'flat'):
        """
        Initialize semantic search engine
        
        Args:
            model_name: Sentence transformer model name
            index_type: FAISS index type ('flat' or 'ivf')
        """
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_generator.embedding_dim,
            index_type=index_type
        )
    
    def index_documents(self, chunks: List):
        """
        Index document chunks
        
        Args:
            chunks: List of DocumentChunk objects from Week 1
        """
        print("\n" + "="*60)
        print("INDEXING DOCUMENTS")
        print("="*60)
        
        # Generate embeddings
        embedded_chunks = self.embedding_generator.embed_chunks(chunks)
        
        # Add to vector store
        self.vector_store.add_embeddings(embedded_chunks)
        
        print("="*60)
        print("✅ INDEXING COMPLETE")
        print("="*60)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks
        
        Args:
            query: Natural language query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Search
        results = self.vector_store.search(query_embedding, k)
        
        return results
    
    def save(self, save_path: str):
        """Save search engine state"""
        self.vector_store.save(save_path)
        
        # Save model info
        save_dir = Path(save_path)
        model_info = {
            'model_name': self.embedding_generator.model_name,
            'embedding_dim': self.embedding_generator.embedding_dim
        }
        with open(save_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def load(self, load_path: str):
        """Load search engine state"""
        # Load model info
        load_dir = Path(load_path)
        with open(load_dir / "model_info.json", 'r') as f:
            model_info = json.load(f)
        
        # Reinitialize with same model
        self.embedding_generator = EmbeddingGenerator(model_info['model_name'])
        
        # Load vector store
        self.vector_store.load(load_path)


# Example Usage
if __name__ == "__main__":
    print("\n" + "🚀"*30)
    print("WEEK 2: SEMANTIC SEARCH ENGINE DEMO")
    print("🚀"*30 + "\n")
    
    # Create sample chunks (simulating Week 1 output)
    from dataclasses import dataclass as dc
    
    @dc
    class MockChunk:
        chunk_id: int
        text: str
        source_file: str
        page_number: int
        section_title: str = None
        start_char: int = 0
        end_char: int = 0
    
    sample_chunks = [
        MockChunk(0, "Medical insurance covers hospitalization, surgery, and emergency care.", "policy.pdf", 1),
        MockChunk(1, "Orthopedic procedures including knee replacement and hip surgery are covered.", "policy.pdf", 2),
        MockChunk(2, "Dental services require separate dental insurance coverage.", "policy.pdf", 3),
        MockChunk(3, "Pre-existing conditions are covered after 12 months waiting period.", "policy.pdf", 4),
        MockChunk(4, "Mental health counseling is covered up to 20 sessions per year.", "policy.pdf", 5),
    ]
    
    # Initialize search engine
    search_engine = SemanticSearchEngine()
    
    # Index documents
    search_engine.index_documents(sample_chunks)
    
    # Perform searches
    test_queries = [
        "Does insurance cover knee surgery?",
        "What about dental care?",
        "Is therapy covered?"
    ]
    
    print("\n" + "="*60)
    print("SEMANTIC SEARCH RESULTS")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 60)
        
        results = search_engine.search(query, k=3)
        
        for result in results:
            print(f"\n  Rank {result['rank']} | Similarity: {result['similarity']:.2%}")
            print(f"  📄 {result['source_file']} (Page {result['page_number']})")
            print(f"  💬 {result['text'][:100]}...")
    
    print("\n" + "="*60)
    print("✅ WEEK 2 DEMO COMPLETE")
    print("="*60)