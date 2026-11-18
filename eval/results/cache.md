## Cache threshold sweep (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

87 labelled pairs: 43 paraphrases that should hit, 44 near-misses that must not. A false hit serves the wrong answer, so the recommendation is the highest recall with a false-hit rate of at most 0%.

### Embedding similarity only

| threshold | precision | recall | F1 | false-hit rate | TP / FP / FN |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.549 | 0.907 | 0.684 | 72.7% | 39 / 32 / 4 |
| 0.82 | 0.551 | 0.884 | 0.678 | 70.5% | 38 / 31 / 5 |
| 0.84 | 0.544 | 0.861 | 0.667 | 70.5% | 37 / 31 / 6 |
| 0.86 | 0.552 | 0.861 | 0.673 | 68.2% | 37 / 30 / 6 |
| 0.88 | 0.533 | 0.744 | 0.621 | 63.6% | 32 / 28 / 11 |
| 0.90 | 0.536 | 0.698 | 0.606 | 59.1% | 30 / 26 / 13 |
| 0.92 | 0.510 | 0.581 | 0.543 | 54.5% | 25 / 24 / 18 |
| 0.94 | 0.550 | 0.512 | 0.530 | 40.9% | 22 / 18 / 21 |
| 0.96 | 0.458 | 0.256 | 0.328 | 29.5% | 11 / 13 / 32 |
| 0.98 **←** | 0.375 | 0.140 | 0.203 | 22.7% | 6 / 10 / 37 |

### Embedding similarity + lexical guard

| threshold | precision | recall | F1 | false-hit rate | TP / FP / FN |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.975 | 0.907 | 0.940 | 2.3% | 39 / 1 / 4 |
| 0.82 | 0.974 | 0.884 | 0.927 | 2.3% | 38 / 1 / 5 |
| 0.84 | 0.974 | 0.861 | 0.914 | 2.3% | 37 / 1 / 6 |
| 0.86 | 0.974 | 0.861 | 0.914 | 2.3% | 37 / 1 / 6 |
| 0.88 **←** | 1.000 | 0.744 | 0.853 | 0.0% | 32 / 0 / 11 |
| 0.90 | 1.000 | 0.698 | 0.822 | 0.0% | 30 / 0 / 13 |
| 0.92 | 1.000 | 0.581 | 0.735 | 0.0% | 25 / 0 / 18 |
| 0.94 | 1.000 | 0.512 | 0.677 | 0.0% | 22 / 0 / 21 |
| 0.96 | 1.000 | 0.256 | 0.407 | 0.0% | 11 / 0 / 32 |
| 0.98 | 1.000 | 0.140 | 0.245 | 0.0% | 6 / 0 / 37 |

- Recommended (embedding only): **0.98**, recall 0.140, precision 0.375, false-hit rate 22.7% (no threshold met the false-hit budget; fewest false hits shown)
- Recommended (with guard): **0.88**, recall 0.744, precision 1.000, false-hit rate 0.0%

### Most similar near-misses (embedding similarity)

| id | similarity | note | guard |
|---|---:|---|---|
| c049 | 0.999 | direction reversed between common nouns | direction_differs:checking->savings |
| c087 | 0.998 | long prompt, last clause flipped | long_prompt_not_exact |
| c047 | 0.997 | direction reversed | direction_differs:new->london |
| c086 | 0.997 | long prompt, ending reverses the meaning beyond the 128-token window | long_prompt_not_exact |
| c045 | 0.997 | translation direction reversed | direction_differs:ingilizce |
| c066 | 0.995 | unit direction reversed | direction_differs:kilometres->miles |
| c048 | 0.995 | translation direction reversed | direction_differs:türkçe |
| c071 | 0.995 | different version | numbers_differ |
| c044 | 0.990 | direction reversed | direction_differs:ankara |
| c065 | 0.985 | different year | numbers_differ |
| c067 | 0.979 | different date | numbers_differ |
| c083 | 0.977 | summarise vs translate | entity_differs:ingilizce |

### Paraphrases the guard wrongly vetoes

| id | similarity | note | veto |
|---|---:|---|---|
| c006 | 0.503 | synonyms (özgeçmiş/CV) | entity_differs:cv |
