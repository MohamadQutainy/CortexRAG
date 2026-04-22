from typing import List, Dict

from rag.chunking.base import BaseChunker, ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("chunking.recursive")


SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
   

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
     
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

           
            best_split = end
            for sep in SEPARATORS:
                if not sep:
                    best_split = end
                    break
                
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx > start:
                    best_split = idx + len(sep)
                    break

            chunk_text = text[start:best_split].strip()
            if chunk_text:
                chunks.append(chunk_text)

            
            new_start = best_split - self.chunk_overlap
            
            if new_start <= start:
                
                start = best_split
            else:
                start = new_start

        return chunks

    @timed(label=" recursive chunking ")
    def chunk(self, document: Dict[str, str]) -> List[ChunkResult]:
        
        text = document["text"]
        metadata = {"source": document.get("source", ""), "type": document.get("type", "")}

        text_chunks = self._split_text(text)

        results = []
        for chunk_text in text_chunks:
            results.append(ChunkResult(page_content=chunk_text, metadata=metadata))

        logger.info(f"Created {len(results)} chunks from: {document.get('source', 'unknown')}")
        return results
