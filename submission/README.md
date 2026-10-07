# Hidden-set submission (50 questions)

* `hidden_agentic.jsonl`: **the submitted system** (Agentic GraphRAG). One JSON object per question: answer, citations, confidence, code-written notes, grounding verdict, tokens (context / LLM input / LLM output / total), LLM and tool calls, active time, stop reason, chunks retrieved, the reasoning chain and the full agentic trace (every step: agent, action, arguments, the orchestrator's stated plan, observation, time, tokens, and whether the TigerGraph MCP server served it).
* `agentic_trace_summary` in each line maps one to one to the guidebook's trace items: number of retrieval and reasoning steps, retrieval methods selected, specialised agents invoked, tools called, time per operation, tokens per operation, total tokens, number of chunks and citations, whether the system changed strategy, and when and why it stopped.
* `hidden_graphrag.jsonl`, `hidden_rag.jsonl`: the two baselines, same format.
* `hidden_answers.csv`: all three answers and token totals side by side.

Model: gemini-3.1-flash-lite. Graph: TigerGraph Savanna, graph `OlympicRAG`.
