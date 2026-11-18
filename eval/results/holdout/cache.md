## Cache threshold sweep (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

32 labelled pairs: 15 paraphrases that should hit, 17 near-misses that must not. A false hit serves the wrong answer, so the recommendation is the highest recall with a false-hit rate of at most 0%.

### Embedding similarity only

| threshold | precision | recall | F1 | false-hit rate | TP / FP / FN |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.444 | 0.800 | 0.571 | 88.2% | 12 / 15 / 3 |
| 0.82 | 0.444 | 0.800 | 0.571 | 88.2% | 12 / 15 / 3 |
| 0.84 | 0.440 | 0.733 | 0.550 | 82.3% | 11 / 14 / 4 |
| 0.86 | 0.435 | 0.667 | 0.526 | 76.5% | 10 / 13 / 5 |
| 0.88 | 0.455 | 0.667 | 0.540 | 70.6% | 10 / 12 / 5 |
| 0.90 | 0.455 | 0.667 | 0.540 | 70.6% | 10 / 12 / 5 |
| 0.92 | 0.357 | 0.333 | 0.345 | 52.9% | 5 / 9 / 10 |
| 0.94 | 0.385 | 0.333 | 0.357 | 47.1% | 5 / 8 / 10 |
| 0.96 | 0.125 | 0.067 | 0.087 | 41.2% | 1 / 7 / 14 |
| 0.98 **←** | 0.143 | 0.067 | 0.091 | 35.3% | 1 / 6 / 14 |

### Embedding similarity + lexical guard

| threshold | precision | recall | F1 | false-hit rate | TP / FP / FN |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.667 | 0.800 | 0.727 | 35.3% | 12 / 6 / 3 |
| 0.82 | 0.667 | 0.800 | 0.727 | 35.3% | 12 / 6 / 3 |
| 0.84 | 0.647 | 0.733 | 0.688 | 35.3% | 11 / 6 / 4 |
| 0.86 | 0.625 | 0.667 | 0.645 | 35.3% | 10 / 6 / 5 |
| 0.88 | 0.625 | 0.667 | 0.645 | 35.3% | 10 / 6 / 5 |
| 0.90 | 0.625 | 0.667 | 0.645 | 35.3% | 10 / 6 / 5 |
| 0.92 | 0.556 | 0.333 | 0.417 | 23.5% | 5 / 4 / 10 |
| 0.94 | 0.556 | 0.333 | 0.417 | 23.5% | 5 / 4 / 10 |
| 0.96 | 0.250 | 0.067 | 0.105 | 17.6% | 1 / 3 / 14 |
| 0.98 **←** | 0.333 | 0.067 | 0.111 | 11.8% | 1 / 2 / 14 |

- Recommended (embedding only): **0.98**, recall 0.067, precision 0.143, false-hit rate 35.3% (no threshold met the false-hit budget; fewest false hits shown)
- Recommended (with guard): **0.98**, recall 0.067, precision 0.333, false-hit rate 11.8% (no threshold met the false-hit budget; fewest false hits shown)

### Most similar near-misses (embedding similarity)

| id | similarity | note | guard |
|---|---:|---|---|
| h017 | 0.996 | translation direction reversed | direction_differs:türkçe |
| h023 | 0.995 | en fazla vs en az | polarity_flipped:en fazla/en az |
| h018 | 0.990 | direction reversed | direction_differs:dropbox->google |
| h031 | 0.990 | aç vs tok | **passes** |
| h029 | 0.988 | conversion direction (accusative vs dative) | **passes** |
| h021 | 0.988 | privative suffix | negation_differs |
| h024 | 0.963 | enable vs disable | **passes** |
| h019 | 0.945 | negation (-mamalı) | **passes** |
| h022 | 0.938 | fastest vs slowest | polarity_flipped:fast/slow |
| h028 | 0.916 | different unit | **passes** |
| h016 | 0.913 | direction reversed | direction_differs:bodrum |
| h032 | 0.904 | junior vs senior | **passes** |
