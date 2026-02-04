"""
Multi-Modal RAG: Document Processor
Extracts text, images, and layout from PDFs

Author: Rahul Vuppalapati
"""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional
import base64


class DocumentProcessor:
    """
    Process PDF documents to extract:
    - Text with layout information
    - Images/figures with captions
    - Tables (if detected)
    - Overall document structure
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF file and extract all content.
        
        Returns:
            {
                'pages': [
                    {
                        'page_num': int,
                        'text': str,
                        'images': [
                            {
                                'image': PIL.Image,
                                'bbox': (x0, y0, x1, y1),
                                'caption': str (optional)
                            }
                        ],
                        'layout': dict  # text blocks with positions
                    }
                ],
                'metadata': {
                    'title': str,
                    'author': str,
                    'total_pages': int
                }
            }
        """
        doc = fitz.open(pdf_path)
        
        result = {
            'pages': [],
            'metadata': {
                'title': doc.metadata.get('title', 'Untitled'),
                'author': doc.metadata.get('author', 'Unknown'),
                'total_pages': len(doc)
            }
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = self._process_page(page, page_num)
            result['pages'].append(page_data)
        
        doc.close()
        return result
    
    def _process_page(self, page: fitz.Page, page_num: int) -> Dict:
        """Process a single page."""
        # Extract text with layout
        text_blocks = page.get_text("blocks")
        text = "\n".join([block[4] for block in text_blocks if block[6] == 0])
        
        # Extract images
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get image bounding box (approximate from page layout)
            # In real implementation, would match with layout analysis
            bbox = self._find_image_bbox(page, xref)
            
            # Try to find caption
            caption = self._find_image_caption(text_blocks, bbox)
            
            images.append({
                'image': image,
                'bbox': bbox,
                'caption': caption,
                'page': page_num
            })
        
        # Also extract rendered page as image for visual understanding
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x for better quality
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return {
            'page_num': page_num,
            'text': text,
            'images': images,
            'page_image': page_image,
            'layout': self._extract_layout(text_blocks),
            'width': page.rect.width,
            'height': page.rect.height
        }
    
    def _find_image_bbox(self, page: fitz.Page, xref: int) -> Tuple[float, float, float, float]:
        """Find bounding box of an image on the page."""
        # Simplified - in production would use layout analysis
        # Return page bounds as fallback
        return (0, 0, page.rect.width, page.rect.height)
    
    def _find_image_caption(self, text_blocks: List, image_bbox: Tuple) -> Optional[str]:
        """Find caption for an image based on proximity."""
        # Look for text below or above the image
        # Simplified implementation
        for block in text_blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                # Look for caption indicators
                if text.startswith(('Figure', 'Fig.', 'Table', 'Chart')):
                    return text
        return None
    
    def _extract_layout(self, text_blocks: List) -> List[Dict]:
        """Extract layout structure from text blocks."""
        layout = []
        for block in text_blocks:
            if block[6] == 0:  # Text block
                layout.append({
                    'bbox': (block[0], block[1], block[2], block[3]),
                    'text': block[4],
                    'type': 'text'
                })
        return layout
    
    def create_chunks(self, document: Dict, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Create text chunks for embedding.
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for page in document['pages']:
            text = page['text']
            
            # Simple sliding window chunking
            # In production, would use semantic chunking
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                
                chunks.append({
                    'text': chunk_text,
                    'page_num': page['page_num'],
                    'chunk_type': 'text',
                    'bbox': None
                })
                
                start += chunk_size - overlap
            
            # Add images as chunks
            for img in page['images']:
                chunks.append({
                    'text': img['caption'] or f"Image on page {page['page_num']}",
                    'page_num': page['page_num'],
                    'chunk_type': 'image',
                    'image': img['image'],
                    'bbox': img['bbox']
                })
        
        return chunks
    
    def highlight_pdf(self, pdf_path: str, highlights: List[Dict], output_path: str):
        """
        Create a highlighted version of the PDF.
        
        Args:
            pdf_path: Original PDF
            highlights: List of {page, bbox, color} dicts
            output_path: Where to save highlighted PDF
        """
        doc = fitz.open(pdf_path)
        
        for h in highlights:
            page = doc[h['page']]
            bbox = fitz.Rect(h['bbox'])
            
            # Add highlight annotation
            highlight = page.add_highlight_annot(bbox)
            if 'color' in h:
                highlight.set_colors(stroke=h['color'])
            highlight.update()
        
        doc.save(output_path)
        doc.close()


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
