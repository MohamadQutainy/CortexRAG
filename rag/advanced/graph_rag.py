import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("advanced.graph_rag")


class EntityGraph:


    def __init__(self):
        
        self.nodes: Dict[str, dict] = {}
        
        self.edges: Dict[Tuple[str, str], dict] = {}
        
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_entity(self, name: str, entity_type: str = "", attributes: dict = None, source: str = ""):
       
        name_lower = name.lower().strip()
        if name_lower not in self.nodes:
            self.nodes[name_lower] = {
                "name": name,
                "type": entity_type,
                "attributes": attributes or {},
                "sources": {source},
            }
        else:
            self.nodes[name_lower]["sources"].add(source)
            if attributes:
                self.nodes[name_lower]["attributes"].update(attributes)

    def add_relation(self, entity1: str, entity2: str, relation: str, source: str = ""):
   
        e1 = entity1.lower().strip()
        e2 = entity2.lower().strip()
        self.edges[(e1, e2)] = {"relation": relation, "source": source}
        self.adjacency[e1].add(e2)
        self.adjacency[e2].add(e1)

    def get_related(self, entity: str, max_depth: int = 2) -> List[dict]:

        entity_lower = entity.lower().strip()
        visited = set()
        results = []

        def _traverse(current: str, depth: int):
            if depth > max_depth or current in visited:
                return
            visited.add(current)

            for neighbor in self.adjacency.get(current, set()):
                edge_key = (current, neighbor) if (current, neighbor) in self.edges else (neighbor, current)
                edge_data = self.edges.get(edge_key, {})

                if neighbor not in visited:
                    results.append({
                        "entity": self.nodes.get(neighbor, {}).get("name", neighbor),
                        "type": self.nodes.get(neighbor, {}).get("type", ""),
                        "relation": edge_data.get("relation", "related to"),
                        "depth": depth,
                    })
                    _traverse(neighbor, depth + 1)

        _traverse(entity_lower, 1)
        return results

    def get_context_for_entities(self, entities: List[str]) -> str:
   
        context_parts = []
        for entity in entities:
            related = self.get_related(entity, max_depth=1)
            if related:
                entity_name = self.nodes.get(entity.lower(), {}).get("name", entity)
                relations = [f"{r['entity']} ({r['relation']})" for r in related[:5]]
                context_parts.append(f"{entity_name} is related to: {', '.join(relations)}")

        return "\n".join(context_parts)

    def stats(self) -> dict:
        
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
        }


class GraphRAG:


    def __init__(self, llm_model: str = "openai/gpt-4.1-nano"):
        self.llm_model = llm_model
        self.graph = EntityGraph()

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    def _extract_entities_from_text(self, text: str, source: str) -> dict:

        prompt = f"""
Extract entities and relationships from this text.

Text:
{text[:2000]}

Return a JSON object with:
- "entities": list of {{"name": "...", "type": "person|product|company|contract|location"}}
- "relations": list of {{"entity1": "...", "entity2": "...", "relation": "..."}}

Return ONLY valid JSON, nothing else.
"""
        response = completion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response.choices[0].message.content.strip()

        # محاولة تحليل JSON
        try:
            # إزالة markdown code blocks إن وُجدت
            if reply.startswith("```"):
                reply = reply.split("```")[1]
                if reply.startswith("json"):
                    reply = reply[4:]
            return json.loads(reply)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from extraction results: {source}")
            return {"entities": [], "relations": []}

    @timed(label="Building the graph:")
    def extract_and_build(self, chunks: List[ChunkResult]):
    
        logger.info(f"Extracting entities from {len(chunks)} chunks...")

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            extracted = self._extract_entities_from_text(chunk.page_content, source)

         
            for entity in extracted.get("entities", []):
                self.graph.add_entity(
                    name=entity.get("name", ""),
                    entity_type=entity.get("type", ""),
                    source=source,
                )

         
            for rel in extracted.get("relations", []):
                self.graph.add_relation(
                    entity1=rel.get("entity1", ""),
                    entity2=rel.get("entity2", ""),
                    relation=rel.get("relation", "related to"),
                    source=source,
                )

        stats = self.graph.stats()
        logger.info(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    def enrich_context(self, query: str, chunks: List[ChunkResult]) -> List[ChunkResult]:

        if not self.graph.nodes:
            return chunks

      
        query_entities = []
        query_lower = query.lower()
        for entity_key, entity_data in self.graph.nodes.items():
            if entity_key in query_lower or entity_data["name"].lower() in query_lower:
                query_entities.append(entity_key)

        if not query_entities:
            return chunks

     
        graph_context = self.graph.get_context_for_entities(query_entities)

        if graph_context:
           
            graph_chunk = ChunkResult(
                page_content=f"[Graph Context]\n{graph_context}",
                metadata={"source": "knowledge_graph", "type": "graph_context"},
            )
            chunks = [graph_chunk] + chunks
            logger.info(f"Context enriched with relations for {len(query_entities)} entities")

        return chunks
