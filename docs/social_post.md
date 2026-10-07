# Social post drafts

## LinkedIn

When does a question need an AI agent, and when is one graph lookup enough?

For the @TigerGraph Agentic GraphRAG Hackathon I built OlympicRAG: three question-answering pipelines (RAG, GraphRAG and Agentic GraphRAG) on one TigerGraph Savanna graph (2,951 Wikipedia articles), all using the same LLM and measured the same way.

RAG: 47% on the 100 public questions
GraphRAG: 65%
Agentic GraphRAG: 99%, and the fewest tokens per correct answer

What I learned:
- Counting is where plain LLM pipelines break. GraphRAG had every relevant event in its context and still miscounted; the agent counts in GSQL.
- "The Games held immediately before" needs a hop through time. A graph edge between editions makes it one step for the agent.
- Facts change. Medals get reallocated and articles disagree with infoboxes. Each version is stored as a Claim in the graph, and the agent reports both instead of picking one silently.
- Agents are not always worth it. For questions about a single article found by description, plain RAG was just as accurate for under 40% of the tokens.
- Every answer is checked in code before release: one cited source, every name and number in the evidence, no merging of events. If the check fails twice, the answer is "unknown", never a guess.

Graph queries run through the TigerGraph MCP server, and every answer comes with a full trace you can follow hop by hop.

#TigerGraph #GraphRAG #AgenticAI #KnowledgeGraphs #LLM

## X / Twitter (short)

RAG vs GraphRAG vs Agentic GraphRAG on one @TigerGraph graph for the Agentic GraphRAG Hackathon:
47%, 65%, 99%, and the agent spends the fewest tokens per correct answer.
Agents win on counting and hops through time. Every answer is checked in code before release. #GraphRAG
