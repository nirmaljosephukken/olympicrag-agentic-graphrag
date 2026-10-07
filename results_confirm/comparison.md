# Reproducibility: `results` vs `results_confirm`

Two complete, independent runs of all three pipelines on every question set, same code, same graph, same LLM.

## public

| Pipeline | Accuracy run 1 | Accuracy run 2 | Same verdict | Tokens/q run 1 | Tokens/q run 2 | Grounded run 1 | Grounded run 2 |
|---|---|---|---|---|---|---|---|
| RAG | 47% | 47% | 98/100 | 3,116 | 3,114 | 69/100 | 68/100 |
| GraphRAG | 65% | 62% | 91/100 | 2,364 | 2,366 | 57/100 | 61/100 |
| Agentic GraphRAG | 99% | 99% | 100/100 | 3,095 | 3,142 | 100/100 | 100/100 |

## hard

| Pipeline | Accuracy run 1 | Accuracy run 2 | Same verdict | Tokens/q run 1 | Tokens/q run 2 | Grounded run 1 | Grounded run 2 |
|---|---|---|---|---|---|---|---|
| RAG | 33% | 33% | 36/36 | 3,136 | 3,138 | 17/36 | 17/36 |
| GraphRAG | 33% | 33% | 34/36 | 2,199 | 2,201 | 10/36 | 11/36 |
| Agentic GraphRAG | 100% | 100% | 36/36 | 4,584 | 4,508 | 36/36 | 36/36 |

## time

| Pipeline | Accuracy run 1 | Accuracy run 2 | Same verdict | Tokens/q run 1 | Tokens/q run 2 | Grounded run 1 | Grounded run 2 |
|---|---|---|---|---|---|---|---|
| RAG | 78% | 72% | 17/18 | 2,939 | 2,945 | 14/18 | 13/18 |
| GraphRAG | 94% | 100% | 17/18 | 1,938 | 1,935 | 15/18 | 15/18 |
| Agentic GraphRAG | 100% | 100% | 18/18 | 3,103 | 3,095 | 18/18 | 18/18 |

## open

| Pipeline | Accuracy run 1 | Accuracy run 2 | Same verdict | Tokens/q run 1 | Tokens/q run 2 | Grounded run 1 | Grounded run 2 |
|---|---|---|---|---|---|---|---|
| RAG | 100% | 100% | 12/12 | 2,684 | 2,684 | 12/12 | 12/12 |
| GraphRAG | 58% | 58% | 12/12 | 3,218 | 3,282 | 7/12 | 7/12 |
| Agentic GraphRAG | 100% | 100% | 12/12 | 7,002 | 6,731 | 12/12 | 12/12 |

## hidden (no gold answers: compared on the answer itself)

| Pipeline | Same answer | Unknown run 1 | Unknown run 2 | Grounded run 1 | Grounded run 2 |
|---|---|---|---|---|---|
| RAG | 46/50 | 13 | 13 | 31/50 | 30/50 |
| GraphRAG | 42/50 | 8 | 8 | 27/50 | 27/50 |
| Agentic GraphRAG | 50/50 | 0 | 0 | 50/50 | 50/50 |

## Questions whose verdict changed between the runs

| Set | Pipeline | Question | Run 1 | Run 2 |
|---|---|---|---|---|
| public | RAG | pub-016 | Arnd Peiffer (PASS) | Ole Einar Bjørndalen (FAIL) |
| public | RAG | pub-057 | Veronica Campbell (FAIL) | Allyson Felix (PASS) |
| public | GraphRAG | pub-003 | 8 (PASS) | 7 (FAIL) |
| public | GraphRAG | pub-012 | 6 (PASS) | 4 (FAIL) |
| public | GraphRAG | pub-027 | 4 (PASS) | 3 (FAIL) |
| public | GraphRAG | pub-033 | 4 (PASS) | 6 (FAIL) |
| public | GraphRAG | pub-039 | unknown (FAIL) | Sandra Perković (PASS) |
| public | GraphRAG | pub-049 | unknown (FAIL) | Martin Fourcade (PASS) |
| public | GraphRAG | pub-056 | Yevgeny Dementyev (PASS) | unknown (FAIL) |
| public | GraphRAG | pub-069 | 3 (PASS) | 0 (FAIL) |
| public | GraphRAG | pub-070 | 2 (FAIL) | 3 (PASS) |
| hard | GraphRAG | hard-001 | unknown (FAIL) | Georg Hackl (PASS) |
| hard | GraphRAG | hard-011 | Taufik Hidayat (PASS) | unknown (FAIL) |
| time | RAG | time-017 | 48 (PASS) | 47 (FAIL) |
| time | GraphRAG | time-017 | 47 (FAIL) | 48 (PASS) |
