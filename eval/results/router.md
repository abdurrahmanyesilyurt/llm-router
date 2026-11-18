## Router label accuracy (test split: 36 prompts, 18 easy / 18 hard; kNN trained on 86)

| strategy | accuracy | TR | EN | precision (hard) | recall (hard) | routed to small | hard → small | est. cost / 1k req | cost vs always-large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 50.0% | 4 | $4.02 | 66% |
| knn | 0.750 | 0.667 | 0.833 | 0.680 | 0.944 | 30.6% | 1 | $5.07 | 83% |
| always-small | 0.500 | 0.500 | 0.500 | – | 0.000 | 100.0% | 18 | $0.31 | 5% |
| always-large | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.0% | 0 | $6.13 | 100% |
| oracle (labels) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50.0% | 0 | $4.65 | 76% |

Cost model: claude-haiku-5-5 vs claude-sonnet-5-5 list prices; input tokens ≈ len(prompt)/3.5 + 20, output tokens 300 (easy) / 900 (hard) including thinking, assumed equal on both tiers. Heuristic accuracy on the train split (used while writing the rules): 0.965.

kNN parameters used: k=11, margin=0.00. Leave-one-out on the train split (errors weighted 2:1, hard→small : easy→large) picks k=11, margin=0.00 (LOO accuracy 0.930):

| k | margin | LOO accuracy | routed to small | hard → small | easy → large |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.00 | 0.895 | 52.3% | 5 | 4 |
| 3 | 0.05 | 0.872 | 50.0% | 5 | 6 |
| 3 | 0.10 | 0.872 | 50.0% | 5 | 6 |
| 3 | 0.15 | 0.884 | 46.5% | 3 | 7 |
| 3 | 0.20 | 0.826 | 36.0% | 1 | 14 |
| 5 | 0.00 | 0.895 | 47.7% | 3 | 6 |
| 5 | 0.05 | 0.884 | 44.2% | 2 | 8 |
| 5 | 0.10 | 0.895 | 43.0% | 1 | 8 |
| 5 | 0.15 | 0.837 | 37.2% | 1 | 13 |
| 5 | 0.20 | 0.814 | 34.9% | 1 | 15 |
| 7 | 0.00 | 0.907 | 46.5% | 2 | 6 |
| 7 | 0.05 | 0.884 | 44.2% | 2 | 8 |
| 7 | 0.10 | 0.849 | 38.4% | 1 | 12 |
| 7 | 0.15 | 0.837 | 37.2% | 1 | 13 |
| 7 | 0.20 | 0.837 | 37.2% | 1 | 13 |
| 9 | 0.00 | 0.895 | 45.4% | 2 | 7 |
| 9 | 0.05 | 0.895 | 43.0% | 1 | 8 |
| 9 | 0.10 | 0.837 | 37.2% | 1 | 13 |
| 9 | 0.15 | 0.814 | 34.9% | 1 | 15 |
| 9 | 0.20 | 0.756 | 29.1% | 1 | 20 |
| 11 | 0.00 | 0.930 | 46.5% | 1 | 5 |
| 11 | 0.05 | 0.919 | 45.4% | 1 | 6 |
| 11 | 0.10 | 0.837 | 37.2% | 1 | 13 |
| 11 | 0.15 | 0.814 | 34.9% | 1 | 15 |
| 11 | 0.20 | 0.779 | 31.4% | 1 | 18 |
