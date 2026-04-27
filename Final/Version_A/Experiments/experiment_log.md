# Version A Experiment Log

Use this file to record concise experiment notes.

| Date | Model | Data split | Parameters | Key result | Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-04-26 | A0 Bayesian popularity | filtered temporal | `m=100` | Hit@10 `0.010031`, NDCG@10 `0.004618`, Coverage@10 `0.000621` | Keep as the main exact-match baseline, not the final metadata model |
| 2026-04-26 | A2 TF-IDF item content | filtered temporal | shared TF-IDF content index | Hit@10 `0.002185`, NDCG@10 `0.001134`, Coverage@10 `0.448520`, `0.28565` s/user | Keep as interpretable item-similarity baseline; do not emphasize due to slow runtime |
| 2026-04-26 | A3 user-profile content | filtered temporal | `max_features=30000`, `min_df=3`, `ngram_range=(1, 2)` | Hit@10 `0.002681`, NDCG@10 `0.001321`, Coverage@10 `0.457397` | Keep as the main personalized content baseline |
| 2026-04-26 | A4 content plus popularity rerank | filtered temporal | content `0.70`, popularity `0.25`, practical `0.05` | Hit@10 `0.002681`, NDCG@10 `0.001344`, Coverage@10 `0.452997` | Select as final Version A model |
| 2026-04-26 | A5 SVD semantic content | filtered temporal | `n_components=128` | Hit@10 `0.002086`, NDCG@10 `0.000956`, Coverage@10 `0.595067` | Keep as required semantic comparison; not final model |

## Notes

- Record only meaningful changes.
- Include runtime when it affects model choice.
- Keep failed runs if they explain why a simpler model was chosen.

## Final run notes

- Final run evaluated `10,069` users with `0` recommendation errors.
- Full notebook runtime was about `8,318` seconds, or `138.6` minutes.
- A2 was the main runtime bottleneck, taking about `2,876` seconds to evaluate.
- The final write-up should explain that A0 is the strongest popularity baseline, while A4 is the best metadata-first final model because it balances personalization, coverage, interpretability, and ranking quality.
