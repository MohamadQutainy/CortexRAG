from typing import List, Dict
from pydantic import BaseModel, Field
from litellm import completion
from multiprocessing import Pool
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.chunking.base import BaseChunker, ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("chunking.semantic")


class Chunk(BaseModel):
    
    headline: str = Field(
        description="A brief heading for this chunk that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk"
    )
    original_text: str = Field(
        description="The original text of this chunk, exactly as is"
    )


class Chunks(BaseModel):

    chunks: list[Chunk]


class SemanticChunker(BaseChunker):
  

    def __init__(self, model: str, average_chunk_size: int = 100, workers: int = 3):
        self.model = model
        self.average_chunk_size = average_chunk_size
        self.workers = workers
        self._wait = wait_exponential(multiplier=1, min=10, max=240)

    def _make_prompt(self, document: Dict[str, str]) -> str:
      
        how_many = (len(document["text"]) // self.average_chunk_size) + 1
        return f"""
You take a document and split it into overlapping chunks for a KnowledgeBase.

The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions.
Divide the document ensuring the entire content is covered — don't leave anything out.
This document should probably be split into at least {how_many} chunks.
There should be ~25% overlap between chunks (~50 words) for best retrieval results.

For each chunk, provide a headline, a summary, and the original text.

Here is the document:

{document["text"]}

Respond with the chunks.
"""

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    def _process_single(self, document: Dict[str, str]) -> List[ChunkResult]:
        
        messages = [{"role": "user", "content": self._make_prompt(document)}]
        response = completion(model=self.model, messages=messages, response_format=Chunks)
        reply = response.choices[0].message.content
        doc_chunks = Chunks.model_validate_json(reply).chunks

        results = []
        metadata = {"source": document["source"], "type": document["type"]}
        for chunk in doc_chunks:
            content = f"{chunk.headline}\n\n{chunk.summary}\n\n{chunk.original_text}"
            results.append(ChunkResult(page_content=content, metadata=metadata))

        return results

    def chunk(self, document: Dict[str, str]) -> List[ChunkResult]:
        
        return self._process_single(document)

    @timed(label="Semantic Chunking")
    def chunk_many(self, documents: List[Dict[str, str]]) -> List[ChunkResult]:
        chunks = []

        if self.workers <= 1:
            
            for doc in tqdm(documents, desc="Chunking"):
                chunks.extend(self._process_single(doc))
        else:
            
            with Pool(processes=self.workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(self._process_single, documents),
                    total=len(documents),
                    desc="Chunking",
                ):
                    chunks.extend(result)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
