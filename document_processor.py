# Week 1: Document Ingestion & Preprocessing System
# File: document_processor.py

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Required installations:
# pip install PyMuPDF python-docx

import fitz  # PyMuPDF
from docx import Document


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    text: str
    chunk_id: int
    source_file: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0


class DocumentExtractor:
    """Handles extraction of text from PDF and DOCX files"""
    
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size = max_file_size_mb * 1024 * 1024
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file integrity and size"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large. Max size: {self.max_file_size / (1024*1024)}MB")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.pdf', '.docx']:
            raise ValueError(f"Unsupported file type: {file_ext}. Only .pdf and .docx allowed")
        
        return True
    
    def extract_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF with page-level granularity"""
        doc = fitz.open(file_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            pages_data.append({
                'page_number': page_num + 1,
                'text': text,
                'source_file': Path(file_path).name
            })
        
        doc.close()
        return pages_data
    
    def extract_from_docx(self, file_path: str) -> List[Dict]:
        """Extract text from DOCX with paragraph-level structure"""
        doc = Document(file_path)
        paragraphs_data = []
        
        for idx, para in enumerate(doc.paragraphs):
            if para.text.strip():  # Skip empty paragraphs
                paragraphs_data.append({
                    'paragraph_number': idx + 1,
                    'text': para.text,
                    'source_file': Path(file_path).name,
                    'style': para.style.name
                })
        
        return paragraphs_data
    
    def extract(self, file_path: str) -> List[Dict]:
        """Main extraction method that routes to appropriate extractor"""
        self.validate_file(file_path)
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_from_docx(file_path)


class TextCleaner:
    """Cleans and normalizes extracted text"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Apply comprehensive text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_headers_footers(text: str, patterns: List[str] = None) -> str:
        """Remove common header/footer patterns"""
        if patterns is None:
            patterns = [
                r'Page \d+ of \d+',
                r'Copyright ©.*?\d{4}',
                r'Confidential.*?$',
                r'^\d+\s*$'  # Page numbers on separate lines
            ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text
    
    def process(self, text: str) -> str:
        """Apply all cleaning operations"""
        text = self.remove_headers_footers(text)
        text = self.clean_text(text)
        return text


class DocumentChunker:
    """Splits documents into overlapping chunks for better context"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences (preserving context)"""
        # Simple sentence splitting (can be enhanced with spaCy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Create overlapping chunks from text"""
        if not text or len(text) < self.chunk_size:
            return [DocumentChunk(
                text=text,
                chunk_id=0,
                source_file=metadata.get('source_file', 'unknown'),
                page_number=metadata.get('page_number'),
                start_char=0,
                end_char=len(text)
            )]
        
        chunks = []
        sentences = self.chunk_by_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, create a chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_file=metadata.get('source_file', 'unknown'),
                    page_number=metadata.get('page_number'),
                    start_char=char_position,
                    end_char=char_position + len(chunk_text)
                ))
                
                # Keep overlap: retain last few sentences
                overlap_text = chunk_text[-self.overlap:] if len(chunk_text) > self.overlap else chunk_text
                overlap_sentences = self.chunk_by_sentences(overlap_text)
                
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
                char_position += len(chunk_text) - len(overlap_text)
                chunk_id += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_file=metadata.get('source_file', 'unknown'),
                page_number=metadata.get('page_number'),
                start_char=char_position,
                end_char=char_position + len(chunk_text)
            ))
        
        return chunks


class DocumentProcessor:
    """Main orchestrator for document processing pipeline"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.extractor = DocumentExtractor()
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Complete processing pipeline: extract -> clean -> chunk"""
        print(f"Processing: {file_path}")
        
        # Step 1: Extract
        print("  ├─ Extracting text...")
        raw_data = self.extractor.extract(file_path)
        
        # Step 2: Clean and chunk
        print("  ├─ Cleaning and chunking...")
        all_chunks = []
        
        for data in raw_data:
            # Clean text
            cleaned_text = self.cleaner.process(data['text'])
            
            if not cleaned_text:
                continue
            
            # Create metadata
            metadata = {
                'source_file': data['source_file'],
                'page_number': data.get('page_number'),
                'paragraph_number': data.get('paragraph_number')
            }
            
            # Chunk text
            chunks = self.chunker.create_chunks(cleaned_text, metadata)
            all_chunks.extend(chunks)
        
        print(f"  └─ Created {len(all_chunks)} chunks")
        return all_chunks
    
    def process_directory(self, directory_path: str) -> Dict[str, List[DocumentChunk]]:
        """Process all documents in a directory"""
        results = {}
        directory = Path(directory_path)
        
        for file_path in directory.glob('*'):
            if file_path.suffix.lower() in ['.pdf', '.docx']:
                try:
                    chunks = self.process_document(str(file_path))
                    results[file_path.name] = chunks
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
        
        return results


# Example Usage
if __name__ == "__main__":
    # Initialize processor
    processor = DocumentProcessor(chunk_size=800, overlap=150)
    
    # Process a single document
    try:
        chunks = processor.process_document("sample_policy.pdf")
        
        print("\n" + "="*60)
        print("SAMPLE CHUNKS:")
        print("="*60)
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Source: {chunk.source_file}")
            print(f"  Page: {chunk.page_number}")
            print(f"  Text Preview: {chunk.text[:200]}...")
            print(f"  Length: {len(chunk.text)} chars")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo test this code:")
        print("1. pip install PyMuPDF python-docx")
        print("2. Place a PDF or DOCX file in the same directory")
        print("3. Update the filename in the code")