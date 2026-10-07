# OlympicRAG: when does a question need an agent?

*Agentic GraphRAG Hackathon, Round 1 · Amal Francis V Ukken*

## What we built

Three question-answering pipelines on one TigerGraph Savanna graph, all using the same LLM (Gemini 3.1 Flash-Lite), run on the same questions and measured the same way:

1. **RAG**: vector search over 22,554 chunks, then one LLM call.
2. **GraphRAG**: entity linking, a fixed graph retrieval with one-hop expansion, vector search restricted to the graph-selected articles, then one LLM call.
3. **Agentic GraphRAG**: an agent harness with an orchestrator LLM that plans the investigation one step at a time. It picks one of thirteen specialised agents, reads what came back and decides the next hop. A code-level evidence evaluator decides whether it may stop.

The question we wanted to answer: **where does the agent pay for itself, and where is one graph lookup enough?**

## How it maps to the brief

| The brief asks for | Where it is |
|---|---|
| Agent harness managing state, tools, context, evidence and stopping | `agr/pipelines.py`: question state and linked entities; tool registry; step history passed to the orchestrator as context (long observations truncated, context tokens counted); evidence ledger with an id per fact or chunk; 8-step budget; stop reasons (`evidence_sufficient`, `withheld_after_review`, `step_budget_exhausted`) |
| Orchestrator deciding the next action from the question and evidence | one LLM call per step returns `{reasoning, action, args}` or a final answer with evidence ids |
| Entity linking agent | `entity_linker`: year, season, sport, venue, date phrase and exact event titles, from the graph catalogue |
| Graph traversal agents | `graph_search`, `venue_events`, `event_profile`, `edition_hop`, `athlete_medals`, `fact_history` |
| Aggregation agent | `aggregate`: exact counts and argmax with GSQL accumulators |
| Multi-hop reasoning agent | `multi_hop`: resolves one start event exactly, then walks a planned chain of hops along the edition edges with evidence at every hop; refuses to guess on an ambiguous start |
| Document retrieval agents | `find_documents` (title match) and `read_document` (ordered chunks, or only those containing a keyword) |
| Similarity search agents | `similarity_search` (HNSW over all chunks), `scoped_similarity` (vectors inside graph-selected articles) |
| Evidence evaluation agent | `agr/grounding.py`, enforced in code before any answer is released |
| RAG vs GraphRAG vs Agentic GraphRAG on accuracy, completeness, tokens | `agr/benchmark.py`, `results/metrics.json`, the dashboard |
| Savanna, GSQL, vector DB, TigerGraph MCP | one graph `OlympicRAG` with HNSW vectors; installed GSQL queries; every agent query sent through the official TigerGraph MCP server |
| Stretch: evolving, conflicting and uncertain facts | `Claim` vertices with `SUPERSEDES` edges, the `fact_history` agent and code-written conflict notes |

## How it works

**Ingestion with no LLM.** Wikipedia infoboxes are parsed into Event, Games, Sport, Venue, Athlete and Nation vertices, with NEXT_EDITION / PREV_EDITION edges between the same event at consecutive Games. Team names are split into athletes only when the split is certain, so no person is invented. Chunks and their 512-d embeddings sit in TigerGraph's HNSW index, so graph and vector search run in one engine.

**TigerGraph MCP.** The agents reach the graph through the official TigerGraph MCP server (`tigergraph-mcp`, `run_installed_query`). Each trace step records whether MCP or the RESTPP fallback served it. In the final agent runs, MCP served every graph step (371 of 371).

**Reasoning over time.** Some facts have several versions: a medallist "originally won" and was later disqualified, or the infobox and the article text give different competitor counts. Ingestion stores each version as a `Claim` vertex (1,021 in all) with a `SUPERSEDES` edge from a dated change to the original result. The source policy is written in code: a dated change supersedes the original, the infobox is the current official record, and a disagreeing statement is reported with its sentence, never dropped and never merged. `event_profile` flags an event whose facts have several versions, and the `fact_history` agent returns all of them with the resolution.

**No made-up answers.** In our previous TigerGraph hackathon, the judges' main criticism was answers that merged facts from different events. This system checks every answer in code before release:

- the whole answer must be supported by ONE cited evidence item;
- every name and number must appear in that evidence (a count must equal the GSQL aggregate);
- a silver medallist cannot be released as the gold winner (medal tables are read by position);
- a rejected answer goes back to the orchestrator with the reason. After two rejections the result is "unknown" with a note, never the rejected text.

Ties, conflicts and result changes are reported in notes written by code, which the LLM cannot edit. Each answer also carries a multi-hop reasoning chain built from the trace, so every hop leads back to the evidence it rests on.

## Key results

| Set | RAG | GraphRAG | Agentic GraphRAG |
|---|---|---|---|
| 100 public questions | 47% | 65% | **99%** |
| 36 harder multi-step questions (ours) | 33% | 33% | **100%** |
| 18 reasoning-over-time questions (ours) | 78% | 94% | **100%** |
| 12 film questions outside the graph (ours) | **100%** | 58% | **100%** |
| Tokens per public question | 3,116 | **2,364** | 3,095 |
| Tokens per correct public answer | 6,629 | 3,638 | **3,126** |
| LLM calls per public question | 1 | 1 | 2.04 |

**Where the agent pays off.** On counting questions GraphRAG had every relevant event in its context and still miscounted (29%); the agent counts in GSQL (100%). On "the Games held immediately before" questions GraphRAG fetches the year in the question (9%); the agent's multi-hop agent walks the edition chain (100%). On single-event lookups and venue + date questions GraphRAG is just as accurate and cheaper per question. Per correct answer, the agent is the cheapest of the three on the public set.

**Where the agent is overkill.** On our 12 film questions (films are in the corpus but not in the graph, and each question describes a film without naming it) plain RAG is as accurate as the agent (100%) at 38% of its token cost (2,684 against 7,002 tokens per question). The agent went straight to similarity search on 4 of them and tried title matching first on the other 8. GraphRAG drops to 58% with an exact pattern: it fails on every question set in an Olympic year (1992, 2002, 2004), because the graph matches the year to that year's Olympic events and GraphRAG then searches only those articles, and it gets every other question right, because there the graph finds nothing and GraphRAG falls back to searching the whole corpus. So the answer to the hackathon's question is per question type: counting, hops through time and multi-step chains need the agent; a single article found by description does not.

**Grounding.** Of 100 public answers, 100 from the agent are supported by one cited source, against 57 for GraphRAG and 69 for RAG. Across all sets the agent never released an answer that combined events, stated something absent from the evidence, or named the wrong medallist.

**Reasoning over time.** The agent reported both versions in all 5 count conflicts and noted the dated change in all 13 result-change questions. RAG did neither; GraphRAG managed 4 of 5 and 11 of 13.

**Multi-step behaviour.** On the harder set the agent chains steps: venue + date → event → previous edition (one `multi_hop` call); athlete → all medal events → count; two event lookups → compare; and, when the graph has no event for an article, find the article → read its medal line.

**Reproducibility.** A second complete, independent run of all 216 questions on all three pipelines gave the agent the same answer on every question (216 of 216, all grounded both times), against 191 for GraphRAG and 206 for RAG. Most of GraphRAG's changes were counts, because it asks the LLM to count; the agent counts in GSQL. The comparison is in `results_confirm/comparison.md`.

**Honest misses.** pub-028 is ambiguous in the corpus (three finals share the same venue and date text). The agent answers for one event with low confidence, and the code attaches a note naming all three. The expected answer was a different final, so the agent loses that point. We prefer that to blending three events into one answer.

## Limitations

- The provided questions are templated and Olympic-heavy, so the infobox graph covers most of them. Films and people in the corpus rely on vector search and document reading only.
- Our harder set and our reasoning-over-time set come from the same corpus and the same parsed data the graph uses. They show multi-step behaviour; they are not an unbiased estimate of difficulty.
- The agent costs more per question than GraphRAG on simple lookups, where one graph lookup is enough.
- Embeddings are static (fast and free); a transformer embedder would help plain RAG.

## What next

- Typed film and person subgraphs, so non-Olympic questions get the same graph treatment.
- Claims for more fact types (records, dates, venues), with a confidence score per source.
- Use TigerGraph's graph algorithm library (for example, similarity or centrality) to rank candidate events when the text is ambiguous.
