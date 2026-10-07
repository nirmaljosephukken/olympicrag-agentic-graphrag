# Demo video script (about 5 minutes; to stay under 5, drop the third, fifth and seventh examples)

Screen setup: terminal on the left, browser with `dashboard/index.html` on the right. Savanna console in a second tab.

## 0:00 to 0:30 · The question
"Some questions need one lookup. Others need a plan. This project measures where that line is.
I built the three pipelines the brief asks for on one TigerGraph graph with one LLM: plain RAG, GraphRAG, and Agentic GraphRAG, an agent that plans its own investigation.
All three answered the same 100 questions, and the dashboard shows what each one costs and how often it is right."

## 0:30 to 1:10 · The graph (Savanna tab)
Show graph `OlympicRAG` in Savanna (schema view).
"2,951 Wikipedia articles. I parse the infoboxes directly, with no LLM, into events, Games, sports, venues and medallists.
Each event links to the same event at the previous and next Games, which is what makes 'held immediately before' questions a single hop.
Every article is also chunked into 22,554 chunks, with vectors in TigerGraph's own HNSW index, so graph and vector search run in one engine.
The agents reach the graph through the official TigerGraph MCP server; each step in a trace says 'via mcp'."
Show the Claim vertices for one event (e.g. Q743905) in Savanna: "When a fact has several versions, each one is stored as a Claim, with a SUPERSEDES edge from a dated change to the original result."

## 1:10 to 2:30 · Watch the agent think (terminal)
```
python -m agr.ask "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?" --all
```
In the benchmark run (answers can vary slightly live): RAG said 0, GraphRAG said 4, the agent said 5 (correct).
* RAG pulls 8 similar chunks; a count needs all 11 biathlon events, and similarity search cannot guarantee that.
* GraphRAG had all 11 events with their competitor numbers in its context and still miscounted. Across the benchmark, 14 of its 15 count misses happened with every relevant event in context.
* The agent: entity linker finds sport, year, season → orchestrator picks `aggregate` → GSQL accumulators count exactly → the evidence evaluator confirms the number equals the aggregate's count → stop. Two LLM calls.

Second example (temporal):
```
python -m agr.ask "Who won the gold medal in the women's 200 metres athletics event at the Summer Olympics held immediately before 2016?" --all
```
In the benchmark run: RAG named the wrong athlete, GraphRAG answered "unknown" (its fixed retrieval fetches the 2016 Games named in the question), and the agent called its multi-hop agent: one call found the 2016 event, hopped back one edition along the graph and answered Allyson Felix (correct). The trace shows each hop with its evidence.

Third example (multi-hop, venue + date):
```
python -m agr.ask "Who won the gold medal in the event held at ExCeL Exhibition Centre on 30 July at the 2012 Summer Olympics?"
```
"Six events at ExCeL have a date that includes 30 July, but only one, women's épée, has the date text '30 July' exactly. The venue agent ranks candidates by exact date text, so the agent answers for that one event."

Fourth example (honest ambiguity, the grounding check):
```
python -m agr.ask "Who won the gold medal in the event held at Olympic Aquatic Centre on August 14, 2004 (heats & final)?"
```
"Here the corpus itself is ambiguous: three finals share that exact venue and date text. The agent is not allowed to merge them into one answer; the evidence evaluator rejects any answer that combines events or states anything the cited evidence does not. It answers for one event with low confidence, and the code (not the LLM) attaches a note naming all three events. Be upfront: in the benchmark the agent picked the 400 m freestyle (Ian Thorpe) and lost this point, while RAG and GraphRAG happened to name the expected answer, Michael Phelps. It is the agent's only miss, and the trade is deliberate: an answer it cannot pin to one event is flagged, never blended."

Fifth example (our harder set: the agent changes strategy):
```
python -m agr.ask "Who won the gold medal in the women's individual pursuit cycling event at the 1992 Summer Olympics?"
```
"This article has no infobox, so the graph has no event for it. The agent sees graph_search come back empty, switches to finding the article by its title, reads only the passage that contains 'Gold', and answers Petra Rossner. The grounding check reads the medal table by position: in an earlier run the agent picked Kathy Watt, who topped the qualifying table but won silver, and the check now rejects that."

Sixth example (reasoning over time, run with `TG_TRANSPORT=mcp`):
```
python -m agr.ask "How many competitors took part in Athletics at the 2008 Summer Olympics – Men's 110 metres hurdles?"
```
"The infobox says 43 competitors from 33 nations; the article text says 'Forty-two athletes from 32 nations competed.' The event profile flags the conflicting Claims in the graph. The answer is 43, the official record, and a note written by code names both versions with the sentence. Nothing is silently dropped, and nothing is averaged."

```
python -m agr.ask "Who originally won the gold medal in Weightlifting at the 1988 Summer Olympics – Men's 56 kg, before the result was changed?"
```
"Here the agent calls fact_history, reads the superseded result and answers Mitko Grablev, with a note that the result changed after the event. The reasoning chain at the bottom shows each hop and the evidence it rests on."

Seventh example (outside the graph, similarity search):
```
python -m agr.ask "Which film in the corpus, released in 1995, was directed by Bryan Singer and starring Stephen Baldwin?"
```
"Films are in the corpus but not in the graph, and the question never names the film. The agent goes to similarity search, finds The Usual Suspects by meaning and answers from the article. Plain RAG gets these right too, for far fewer tokens, so this is where the agent is not worth it."

## 3:00 to 4:20 · The dashboard
* Scorecard: accuracy, tokens per question, tokens per correct answer, LLM calls. Agent 99% against 65% for GraphRAG and 47% for RAG. GraphRAG is cheapest per question (2,364 tokens against 3,095 for the agent), but the agent is cheapest per correct answer (3,126 against 3,638 for GraphRAG and 6,629 for RAG).
* Grounding check: for every pipeline, how many answers are fully supported by one cited source, and how many combine events, state unsupported facts or name the wrong medallist (public, hidden, hard and time sets).
* By question type: the agent wins on counting (aggregation: RAG 0%, GraphRAG 29%, agent 100%) and "held immediately before" (temporal: 64%, 9%, 100%). On lookups, superlatives and venue + date questions GraphRAG is as accurate (venue + date: GraphRAG 100%, agent 96%), and cheaper.
* "When does the agent pay off?" table: that is the answer to the hackathon question, per question type.
* Harder multi-step set (36 of our own questions, gold computed by code): RAG 33%, GraphRAG 33%, agent 100%. Show the agent paths: venue → event → previous edition, two lookups then compare, and the text-only strategy change.
* Films outside the graph (12 questions): RAG 100%, GraphRAG 58%, agent 100%, but the agent spends 7,002 tokens per question against 2,684 for RAG. Say it plainly: this is where the agent is overkill, and finding where that line sits is the point of the benchmark.
* Reasoning over time (18 questions): RAG 78%, GraphRAG 94%, agent 100%, with both versions reported in all 5 count conflicts and the change noted in all 13 result-change questions.
* Agent behaviour: which specialised agents it called, the common paths, and why it stopped.
* Open one question in the explorer to show the full trace: each decision, tool, time and tokens per step.

## 4:20 to 5:00 · Close
"The agent is not always better; it is better where a question needs counting, a hop through time or more than one step. Where one graph lookup is enough, GraphRAG is just as accurate and cheaper. Every answer the agent gives is checked in code against the evidence it cites, so none of them merges events or states something the corpus does not.
Everything runs on TigerGraph Savanna, it is reproducible from the README, and the raw outputs and traces for the 50 hidden questions are in the repo."
