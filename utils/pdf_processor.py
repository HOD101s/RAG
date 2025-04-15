"""Utility functions for processing PDF files"""

import io
import re
from typing import List

from PyPDF2 import PdfReader


def split_into_chunks(text: str, max_chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into smaller chunks with overlap.

    Args:
        text (str): Text to split into chunks
        max_chunk_size (int, optional): Maximum size of each chunk. Defaults to 500.
        overlap (int, optional): Number of characters to overlap between chunks. Defaults to 100.

    Returns:
        List[str]: List of text chunks
    """
    # Split text into sentences (handling common abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chunk_size, save current chunk
        if current_size + len(sentence) > max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep some sentences for overlap
            overlap_size = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                if overlap_size + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_size += len(s)
            
            current_chunk = overlap_chunk
            current_size = overlap_size
            
        current_chunk.append(sentence)
        current_size += len(sentence)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def extract_text_from_pdf(pdf_file: bytes) -> List[str]:
    """Extract text from a PDF file.

    Args:
        pdf_file (bytes): PDF file content as bytes

    Returns:
        List[str]: List of text chunks from the PDF
    """
    # Create a PDF reader object
    pdf_reader = PdfReader(io.BytesIO(pdf_file))
    
    # Extract text from each page
    text_chunks = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text.strip():  # Only process non-empty text
            # First split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Then split each paragraph into smaller chunks with overlap
            for paragraph in paragraphs:
                chunks = split_into_chunks(paragraph, max_chunk_size=500, overlap=100)
                text_chunks.extend(chunks)
    
    return text_chunks 