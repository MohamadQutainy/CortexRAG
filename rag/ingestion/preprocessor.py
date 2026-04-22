import re
from typing import List, Dict

from rag.observability.logger import get_logger, timed

logger = get_logger("ingestion.preprocessor")


Document = Dict[str, str]


def clean_whitespace(document: Document) -> Document:
   
    text = document["text"]
   
    text = re.sub(r"\n{3,}", "\n\n", text)
   
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
 
    text = text.strip()

    return {**document, "text": text}


def normalize_unicode(document: Document) -> Document:
 
    text = document["text"]
    
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
  
    text = text.replace("\u2014", " — ").replace("\u2013", " – ")

    return {**document, "text": text}


def enrich_metadata(document: Document) -> Document:
   
    text = document["text"]
    word_count = len(text.split())
    line_count = len(text.splitlines())

    metadata = {
        "type": document.get("type", "unknown"),
        "source": document.get("source", "unknown"),
        "word_count": word_count,
        "line_count": line_count,
    }

    return {**document, "metadata": metadata}



PREPROCESSING_STEPS = [
    clean_whitespace,
    normalize_unicode,
    enrich_metadata,
]


@timed(label="Document Preprocessing")
def preprocess_documents(documents: List[Document]) -> List[Document]:

    processed = []
    for doc in documents:
        for step in PREPROCESSING_STEPS:
            doc = step(doc)
        processed.append(doc)

    logger.info(f"Processed {len(processed)} documents")
    return processed
