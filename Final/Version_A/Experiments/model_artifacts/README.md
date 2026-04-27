# Version A Model Artifacts

The full experiment notebook writes trained model artifacts here.

Expected generated files include:

- `a0_bayesian_score_vector.npy`
- `a0_bayesian_popularity_table.csv`
- `a0_bayesian_popularity_metadata.json`
- `tfidf_vectorizer.pkl`
- `tfidf_matrix.npz`
- `tfidf_candidate_indices.npy`
- `tfidf_candidate_recipe_ids.npy`
- `tfidf_content_metadata.json`
- `a4_rerank_weights.json`
- `A5_*_truncated_svd.pkl`
- `A5_*_svd_components.npy`
- `A5_*_explained_variance_ratio.npy`
- `A5_*_item_embeddings.npy`
- `a5_svd_metadata.json`
- `model_artifact_manifest.json`

Generated artifacts can be large, so this folder ignores them by default. Keep the README tracked so the folder structure is visible.
