# Version C Model Notes

## C0_svd64 - SVD CF baseline 64
- precision@10: 0.002524
- recall@10: 0.025244
- ndcg@10: 0.013089
- hit@10: 0.025244
- coverage@10: 0.032638
- runtime: 8.3s

## C1_tfidf10k - TF-IDF content baseline 10000
- precision@10: 0.000105
- recall@10: 0.001047
- ndcg@10: 0.000530
- hit@10: 0.001047
- coverage@10: 0.455663
- runtime: 621.9s

## C2_a70 - Weighted hybrid alpha=0.7
- precision@10: 0.002200
- recall@10: 0.021996
- ndcg@10: 0.011938
- hit@10: 0.021996
- coverage@10: 0.173129
- runtime: 636.9s

## C3_t10 - Switching hybrid threshold=10
- precision@10: 0.001550
- recall@10: 0.015502
- ndcg@10: 0.007879
- hit@10: 0.015502
- coverage@10: 0.406253
- runtime: 270.8s

## C4_rrf60 - RRF hybrid k=60
- precision@10: 0.001655
- recall@10: 0.016550
- ndcg@10: 0.008522
- hit@10: 0.016550
- coverage@10: 0.291231
- runtime: 638.3s

