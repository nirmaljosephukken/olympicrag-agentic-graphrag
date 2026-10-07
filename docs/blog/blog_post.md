# When Does a Question Need an Agent? RAG vs GraphRAG vs Agentic GraphRAG on TigerGraph

*Built for the TigerGraph Agentic GraphRAG Hackathon. Code, results and traces: [github.com/nirmaljosephukken/olympicrag-agentic-graphrag](https://github.com/nirmaljosephukken/olympicrag-agentic-graphrag) · Live dashboard: [nirmaljosephukken.github.io/olympicrag-agentic-graphrag/dashboard](https://nirmaljosephukken.github.io/olympicrag-agentic-graphrag/dashboard/)*

Everyone is building agents. Fewer people are asking when an agent is actually worth it. An agent plans, calls tools, checks its work and tries again, and every one of those steps costs tokens and time. If a single retrieval answers the question just as well, the agent is overhead.

For the TigerGraph Agentic GraphRAG Hackathon I tried to answer that question with numbers. I built three question-answering pipelines on the same TigerGraph graph, gave them the same LLM and the same questions, and measured where each one succeeds and fails. I called the project OlympicRAG, because most of the corpus is about Olympic events.

## The setup

The hackathon provides 2,951 Wikipedia articles (Olympic events from 1988 to 2022, plus films and people), 100 public questions with answers and 50 hidden ones. I added three small sets of my own to stress specific behaviours: 36 harder multi-step questions, 18 questions about facts that changed over time, and 12 questions about films. Every gold answer in my sets is computed or quoted by code from the corpus, never written by an LLM.

All three pipelines use Gemini 3.1 Flash-Lite and one TigerGraph Savanna graph that holds both the structure and the vectors:

1. **RAG**: vector search over 22,554 text chunks, then one LLM call.
2. **GraphRAG**: link the entities in the question to the graph, fetch the matching events and their neighbours, search vectors only inside those articles, then one LLM call.
3. **Agentic GraphRAG**: an orchestrator LLM plans the investigation one step at a time. It chooses one of 13 specialised agents (entity linking, graph traversal, multi-hop reasoning, GSQL aggregation, similarity search, document retrieval, fact history and others), reads what came back, and decides the next step. An evidence evaluator written in code decides whether it may stop.

![Architecture](https://raw.githubusercontent.com/nirmaljosephukken/olympicrag-agentic-graphrag/main/docs/architecture.png)

## Building the graph without an LLM

Most Olympic articles start with an infobox: venue, date, number of competitors, nations, and the gold, silver and bronze medallists. I parse these directly into Event, Games, Sport, Venue, Athlete and Nation vertices, and link each event to the same event at the previous and next Games. No tokens are spent on extraction, and no relationship is invented.

One rule mattered more than I expected: team names are split into individual athletes only when the split is certain. If an infobox lists a relay team without separators, no Athlete vertices are created at all. A guessed split creates people who never existed.

The agents reach the graph through the official TigerGraph MCP server. Every step in a trace records whether MCP served it.

## The rule I cared about most: never release an unsupported answer

In a previous TigerGraph hackathon, the main feedback on our entry was that the LLM made up details and merged several items into one answer. So this time the safeguards live in code, not in the prompt. Before the agent may answer:

- the whole answer must be supported by **one** cited evidence item;
- every name and number in the answer must appear in that evidence (a count must equal the GSQL aggregate, not a number the LLM computed);
- a silver medallist can never be released as the gold winner;
- an answer that combines facts from two events is rejected.

A rejected answer goes back to the orchestrator with the reason. After two rejections the result is "unknown", never the rejected text. When several events match a question equally, the agent answers for one of them and the code attaches a note naming all the candidates.

## Results

On the 100 public questions:

| | RAG | GraphRAG | Agentic GraphRAG |
|---|---|---|---|
| Accuracy | 47% | 65% | **99%** |
| Tokens per question | 3,116 | **2,364** | 3,095 |
| Tokens per correct answer | 6,629 | 3,638 | **3,126** |
| Answers fully supported by one cited source | 69 | 57 | **100** |

GraphRAG is the cheapest per question. The agent is the cheapest per **correct** answer, and it is the only pipeline whose every answer is backed by its evidence.

The more interesting view is by question type, because it shows where the agent earns its cost and where it does not.

| Question type | RAG | GraphRAG | Agent | Verdict |
|---|---|---|---|---|
| Counting (21) | 0% | 29% | 100% | Agent needed |
| "Held immediately before" (22) | 64% | 9% | 100% | Agent needed |
| Single lookup (19) | 63% | 100% | 100% | GraphRAG is enough, and cheaper |
| Venue and date (28) | 64% | 100% | 96% | GraphRAG is enough, and cheaper |
| Superlatives (10) | 30% | 100% | 100% | GraphRAG is enough, and cheaper |

**Counting is where LLM pipelines break.** In 14 of GraphRAG's 15 misses on counting questions, every relevant event and its competitor count was already in its context. The LLM simply miscounted. The agent does not count in the LLM at all: it calls a GSQL aggregation and cites the result.

**Hops through time need structure.** For "who won the event at the Games held immediately before 2016", GraphRAG fetches the events of 2016, because that is the year named in the question, and the answer never reaches its context. The agent's multi-hop agent finds the 2016 event, follows the edge to the previous edition and answers from there, with evidence at each hop.

## Where the agent is overkill

To test the other side, I wrote 12 questions about films. The corpus contains 546 film articles, but the graph only models Olympic events, and each question describes a film without naming it ("the 1995 film directed by Bryan Singer and starring Stephen Baldwin").

| Films (12) | RAG | GraphRAG | Agent |
|---|---|---|---|
| Accuracy | 100% | 58% | 100% |
| Tokens per question | 2,684 | 3,218 | 7,002 |

Plain RAG is as accurate as the agent here, at under 40% of its tokens. When the answer sits in one article that a description can find, an agent is not worth it.

GraphRAG's 58% follows an exact pattern. It failed every question set in an Olympic year (1992, 2002, 2004) and answered every other one correctly. For "the 2002 film directed by Kunal Kohli", the only entity the graph recognises is the year, so it returns 40 events of the 2002 Winter Olympics and searches only those articles. The film can never be reached. In a non-Olympic year the graph finds nothing, GraphRAG falls back to searching everything, and it gets the answer. That is GraphRAG's blind spot: it trusts the graph's view of the question, which helps when the answer is in the graph and hurts when it is not.

## Facts that change

The hackathon's stretch goal is reasoning over evolving and conflicting facts. Wikipedia has plenty: a weightlifter "originally won" and was later disqualified, or the infobox says 48 competitors while the article text says 47.

I extract each version of a fact with code and store it as a Claim vertex in TigerGraph, with a SUPERSEDES edge from a dated decision to the original result. The source policy is explicit: a dated change supersedes the original, the infobox is the current official record, and a disagreeing statement is reported, never dropped or averaged. On 18 such questions RAG scored 78%, GraphRAG 94% and the agent 100%. The agent reported both versions in all 5 count conflicts and flagged the change in all 13 questions where a result had changed.

## Running everything twice

Before submitting, I ran every pipeline on every question a second time, independently. Out of 216 questions:

| | Same answer in both runs |
|---|---|
| RAG | 206 |
| GraphRAG | 191 |
| Agentic GraphRAG | **216** |

Most of GraphRAG's changes were counts: 8 in one run, 7 in the other, for the same question. The LLM counts differently each time. The agent's counts come from the graph, so they do not move. One question shows the conflict problem directly: for the 2016 men's triple jump, RAG answered 48 in one run and 47 in the other, GraphRAG did the opposite, and the agent answered 48, the official record, both times and reported the conflict.

## What I learned

1. **The answer to "do I need an agent?" is per question type.** Counting, hops through time and multi-step chains need one. A single lookup or a single article found by description does not.
2. **Put your guarantees in code.** A prompt that says "do not merge events" is a hope. A check that rejects any answer citing two events is a guarantee.
3. **Let the database do the arithmetic.** Most of the accuracy gap on counting questions and most of the run-to-run instability came from the LLM counting.
4. **Measure where your system is overkill, not just where it wins.** The film questions were the most useful 12 questions in the project.

It still has limitations. The provided questions are templated and Olympic-heavy, my extra sets come from the same corpus, and the agent costs more per question than GraphRAG on simple lookups. The one question the agent missed (out of 100) is ambiguous in the corpus itself: three finals share the same venue and date. The agent answered for one of them and the code flagged all three, which I prefer to blending them into one answer.

The code, every trace, the hidden-set outputs and a dashboard with every question are in the repository.

*Thanks to TigerGraph for the hackathon, the Savanna credits and the MCP server.*

#TigerGraph #GraphRAG #AgenticAI #RAG #KnowledgeGraph #LLM
