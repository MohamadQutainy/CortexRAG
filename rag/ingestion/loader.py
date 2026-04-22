from pathlib import Path
from typing import List, Dict

from rag.observability.logger import get_logger, timed

logger = get_logger("ingestion.loader")



Document = Dict[str, str]


def _read_markdown(file_path: Path) -> str:
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _read_text(file_path: Path) -> str:

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()



LOADERS = {
    ".md": _read_markdown,
    ".txt": _read_text,
    ".markdown": _read_markdown,
}


@timed(label="download docs")
def load_documents(knowledge_base_path: Path, extensions: list = None) -> List[Document]:

    if extensions is None:
        extensions = list(LOADERS.keys())

    documents = []

    if not knowledge_base_path.exists():
        logger.error(f"Knowledge base directory not found: {knowledge_base_path}")
        return documents


    for folder in knowledge_base_path.iterdir():
        if not folder.is_dir():
            continue

        for ext in extensions:
            for file in folder.rglob(f"*{ext}"):
                loader = LOADERS.get(ext)
                if loader is None:
                    continue

                try:
                    doc_type = folder.name
                    text = loader(file)
                    documents.append({
                        "type": doc_type,
                        "source": file.as_posix(),
                        "text": text,
                    })
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")

    logger.info(f"Loaded {len(documents)} documents from {knowledge_base_path}")
    return documents
