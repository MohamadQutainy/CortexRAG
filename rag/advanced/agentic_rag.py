from typing import List, Tuple
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.retrieval.base import BaseRetriever
from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("advanced.agentic")

AGENT_TOOLS = """
Available tools:
1. RETRIEVE: Search the knowledge base with a query. Use: RETRIEVE|<query>
2. ANSWER: Generate the final answer. Use: ANSWER|<answer>

Rules:
- You MUST use RETRIEVE at least once before ANSWER
- If the context is insufficient, try RETRIEVE with a different query
- When you have enough context, use ANSWER to respond
- Maximum {max_iterations} steps allowed
"""

AGENT_SYSTEM_PROMPT = """
You are an intelligent assistant that answers questions using a knowledge base.
You can search the knowledge base multiple times with different queries to find the best answer.

{tools}

For each step, respond with ONLY ONE tool call in the format: TOOL|argument
Think about what information you need and search strategically.
"""

class AgenticRAG:


    def __init__(
        self,
        retriever: BaseRetriever,
        model: str = "openai/gpt-4.1-nano",
        max_iterations: int = 5,
        company_name: str = "Insurellm",
    ):
        self.retriever = retriever
        self.model = model
        self.max_iterations = max_iterations
        self.company_name = company_name

    def _parse_action(self, response: str) -> Tuple[str, str]:
     
        response = response.strip()
        if "|" in response:
            parts = response.split("|", 1)
            tool = parts[0].strip().upper()
            argument = parts[1].strip() if len(parts) > 1 else ""
            return tool, argument
      
        return "ANSWER", response

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    @timed(label="RAG وكيل")
    def run(self, question: str) -> Tuple[str, List[ChunkResult]]:
       
        tools_desc = AGENT_TOOLS.format(max_iterations=self.max_iterations)
        system = AGENT_SYSTEM_PROMPT.format(tools=tools_desc)

        all_chunks: List[ChunkResult] = []
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\n\nDecide your first action."},
        ]

        for step in range(self.max_iterations):
            logger.info(f"Agent - Step {step + 1}/{self.max_iterations}")

            response = completion(model=self.model, messages=messages)
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            tool, argument = self._parse_action(reply)

            if tool == "RETRIEVE":
               
                logger.info(f"Agent searching for: {argument[:50]}...")
                chunks = self.retriever.retrieve(argument)
                all_chunks.extend(chunks)

                
                context = "\n\n".join(
                    f"[{i+1}] {c.page_content[:200]}..." for i, c in enumerate(chunks[:5])
                )
                tool_result = f"Retrieved {len(chunks)} results:\n{context}"
                messages.append({"role": "user", "content": f"Results:\n{tool_result}\n\nDecide next action."})

            elif tool == "ANSWER":
               
                logger.info("Agent generating final answer")

                
                if len(argument) < 50 and all_chunks:
                    full_context = "\n\n".join(
                        f"Extract from {c.metadata.get('source', 'N/A')}:\n{c.page_content}"
                        for c in all_chunks[:10]
                    )
                    final_prompt = f"""
Based on the following context, answer the question completely.
Context:
{full_context}

Question: {question}

Provide a complete, accurate answer.
"""
                    final_response = completion(
                        model=self.model,
                        messages=[{"role": "user", "content": final_prompt}],
                    )
                    return final_response.choices[0].message.content, all_chunks

                return argument, all_chunks

            else:
               
                messages.append({
                    "role": "user",
                    "content": f"Invalid tool '{tool}'. Use RETRIEVE|query or ANSWER|response.",
                })

        
        logger.warning("Agent reached maximum iterations - generating forced answer")
        if all_chunks:
            context = "\n\n".join(c.page_content for c in all_chunks[:10])
            final_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above."
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": final_msg}],
            )
            return response.choices[0].message.content, all_chunks

        return "I could not find enough information to answer this question.", all_chunks
