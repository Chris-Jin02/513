"""Version C debug run - local execution script."""
import os
import json
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "../../Data/Pure_Data"
RESULTS_DIR = "../Results"
FIGURES_DIR = "../Results/Figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEBUG_MODE = False
MAX_DEBUG_USERS = 1000
POSITIVE_THRESHOLD = 4
K = 10
RANDOM_STATE = 42

C0_COMPONENT_VALUES = [32, 64, 128]
C1_MAX_FEATURES_VALUES = [5000, 10000, 20000]
C2_ALPHA_VALUES = [0.3, 0.5, 0.7, 0.9]
C3_THRESHOLD_VALUES = [5, 10, 20]
C4_RRF_K_VALUES = [10, 60]

np.random.seed(RANDOM_STATE)
print("DEBUG_MODE:", DEBUG_MODE)

# ---- Load data ----
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "interactions_train_filtered.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "interactions_test_filtered.csv"))
interactions_filtered = pd.read_csv(os.path.join(DATA_DIR, "interactions_filtered.csv"))
recipe_model_table = pd.read_csv(os.path.join(DATA_DIR, "recipe_model_table.csv"))
print("train shape:", train.shape)
print("test shape:", test.shape)
print("recipe_model_table shape:", recipe_model_table.shape)

# ---- Build user sets ----
test_positive = test[test["rating"] >= POSITIVE_THRESHOLD].copy()

def build_user_sets(df):
    out = defaultdict(set)
    for u, i in zip(df["user_id"].values, df["recipe_id"].values):
        out[u].add(i)
    return out

user_train_all_items = build_user_sets(train)
user_train_positive_items = build_user_sets(train[train["rating"] >= POSITIVE_THRESHOLD])
user_test_relevant_items = build_user_sets(test_positive)
user_train_count = train.groupby("user_id").size().to_dict()

eval_users = sorted(user_test_relevant_items.keys())
if DEBUG_MODE:
    eval_users = eval_users[:MAX_DEBUG_USERS]
    user_test_relevant_items = {u: user_test_relevant_items[u] for u in eval_users}

print("Eval users:", len(eval_users))

# ---- Evaluation functions ----
def precision_at_k(rec, rel, k):
    rec = rec[:k]
    if k == 0 or len(rec) == 0:
        return 0.0
    return sum(1 for x in rec if x in rel) / k

def recall_at_k(rec, rel, k):
    if len(rel) == 0:
        return 0.0
    rec = rec[:k]
    return sum(1 for x in rec if x in rel) / len(rel)

def ndcg_at_k(rec, rel, k):
    rec = rec[:k]
    if len(rec) == 0 or len(rel) == 0:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 1) for rank, item in enumerate(rec, start=1) if item in rel)
    ideal = min(len(rel), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(rec, rel, k):
    return 1.0 if any(x in rel for x in rec[:k]) else 0.0

def evaluate_model(model_id, model_name, phase, split_name, recommendations, runtime_seconds):
    per_user_rows = []
    unique_rec = set()
    total_recs = 0
    for u in eval_users:
        rel = user_test_relevant_items.get(u, set())
        rec_items = recommendations.get(u, [])
        rec_ids = [x[0] if isinstance(x, tuple) else x for x in rec_items][:K]
        unique_rec.update(rec_ids)
        total_recs += len(rec_ids)
        p = precision_at_k(rec_ids, rel, K)
        r = recall_at_k(rec_ids, rel, K)
        n = ndcg_at_k(rec_ids, rel, K)
        h = hit_at_k(rec_ids, rel, K)
        per_user_rows.append({"user_id": u, "model_id": model_id,
                              "precision_at_k": p, "recall_at_k": r,
                              "ndcg_at_k": n, "hit_at_k": h})
    per_user_df = pd.DataFrame(per_user_rows)
    eval_count = len(per_user_df)
    catalog_cov = len(unique_rec) / train["recipe_id"].nunique() if train["recipe_id"].nunique() > 0 else 0.0
    sec_per = runtime_seconds / eval_count if eval_count > 0 else 0
    agg = {"version": "C", "phase": phase, "model_id": model_id, "model_name": model_name,
           "split": split_name, "k": K, "evaluated_users": eval_count,
           "precision_at_k": float(per_user_df["precision_at_k"].mean()),
           "recall_at_k": float(per_user_df["recall_at_k"].mean()),
           "ndcg_at_k": float(per_user_df["ndcg_at_k"].mean()),
           "hit_at_k": float(per_user_df["hit_at_k"].mean()),
           "catalog_coverage_at_k": float(catalog_cov),
           "total_recommendations": int(total_recs),
           "unique_recommended_items": int(len(unique_rec)),
           "runtime_seconds": float(runtime_seconds),
           "seconds_per_user": float(sec_per)}
    return agg, per_user_df

# ---- Popularity fallback ----
pop_global_mean = train["rating"].mean()
pop = train.groupby("recipe_id").agg(R=("rating", "mean"), v=("rating", "count")).reset_index()
pop["score"] = (pop["v"] / (pop["v"] + 100)) * pop["R"] + (100 / (pop["v"] + 100)) * pop_global_mean
pop = pop.sort_values("score", ascending=False)
pop_ranked = list(zip(pop["recipe_id"].tolist(), pop["score"].tolist()))

def popularity_recs_for_users(pop_ranked, users):
    out = {}
    for u in users:
        seen = user_train_all_items.get(u, set())
        recs = []
        for rid, score in pop_ranked:
            if rid in seen:
                continue
            recs.append((rid, float(score)))
            if len(recs) >= K:
                break
        out[u] = recs
    return out

print("Popularity fallback ready.")

# ==== C0: SVD CF Baseline ====
print("\n=== C0: SVD CF Baseline ===")
metrics_rows = []
tuning_rows = []

for comp in C0_COMPONENT_VALUES:
    t0 = time.time()
    users_arr = np.sort(train["user_id"].unique())
    items_arr = np.sort(train["recipe_id"].unique())
    u2i_svd = {u: idx for idx, u in enumerate(users_arr)}
    r2i_svd = {r: idx for idx, r in enumerate(items_arr)}
    i2r_svd = {idx: r for r, idx in r2i_svd.items()}
    rows_idx = train["user_id"].map(u2i_svd).values
    cols_idx = train["recipe_id"].map(r2i_svd).values
    vals = train["rating"].astype(float).values
    mat = csr_matrix((vals, (rows_idx, cols_idx)), shape=(len(users_arr), len(items_arr)), dtype=np.float32)
    n_comp = min(comp, min(mat.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    user_emb = svd.fit_transform(mat)
    item_emb = svd.components_.T
    svd_score_mat = user_emb @ item_emb.T

    recs = {}
    for u in eval_users:
        if u not in u2i_svd:
            recs[u] = popularity_recs_for_users(pop_ranked, [u])[u]
            continue
        uidx = u2i_svd[u]
        seen = user_train_all_items.get(u, set())
        s = svd_score_mat[uidx]
        idx_rank = np.argsort(-s)
        out = []
        for idx in idx_rank:
            item = i2r_svd[int(idx)]
            if item in seen:
                continue
            out.append((item, float(s[idx])))
            if len(out) >= K:
                break
        recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
    runtime = time.time() - t0
    agg, _ = evaluate_model(f"C0_svd{comp}", f"SVD CF baseline {comp}", "C0", "filtered_temporal", recs, runtime)
    agg["parameters"] = json.dumps({"n_components": comp})
    agg["notes"] = "SVD collaborative filtering baseline"
    metrics_rows.append(agg)
    tuning_rows.append(agg)
    print(f"  C0_svd{comp}: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

c0_recs = recs

# ==== C1: TF-IDF Content Baseline ====
print("\n=== C1: TF-IDF Content Baseline ===")
text_cols = [col for col in ["name", "description", "tags_text", "ingredients_text"] if col in recipe_model_table.columns]
if not text_cols:
    text_cols = [col for col in recipe_model_table.columns if recipe_model_table[col].dtype == object][:4]
print("  Text columns:", text_cols)

recipe_model_table["combined_text_v"] = recipe_model_table[text_cols].fillna("").agg(" ".join, axis=1)
# Column is "id" not "recipe_id"
recipe_id_col = "recipe_id" if "recipe_id" in recipe_model_table.columns else "id"
recipe_id_to_idx = {rid: idx for idx, rid in enumerate(recipe_model_table[recipe_id_col].values)}
recipe_ids_array = recipe_model_table[recipe_id_col].values

for mf in C1_MAX_FEATURES_VALUES:
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=mf, stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(recipe_model_table["combined_text_v"])

    recs = {}
    for u in eval_users:
        liked = user_train_positive_items.get(u, set())
        liked_in_model = [recipe_id_to_idx[r] for r in liked if r in recipe_id_to_idx]
        if not liked_in_model:
            recs[u] = popularity_recs_for_users(pop_ranked, [u])[u]
            continue
        user_vec = tfidf_matrix[liked_in_model].mean(axis=0)
        user_vec = np.asarray(user_vec).reshape(1, -1)
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        seen = user_train_all_items.get(u, set())
        idx_rank = np.argsort(-sims)
        out = []
        for idx in idx_rank:
            rid = int(recipe_ids_array[idx])
            if rid in seen:
                continue
            out.append((rid, float(sims[idx])))
            if len(out) >= K:
                break
        recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
    runtime = time.time() - t0
    agg, _ = evaluate_model(f"C1_tfidf{mf}", f"TF-IDF content baseline {mf}", "C1", "filtered_temporal", recs, runtime)
    agg["parameters"] = json.dumps({"max_features": mf})
    agg["notes"] = "TF-IDF user-profile content baseline"
    metrics_rows.append(agg)
    tuning_rows.append(agg)
    print(f"  C1_tfidf{mf}: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

c1_recs = recs

# ==== C2: Weighted Hybrid ====
print("\n=== C2: Weighted Hybrid ===")
for alpha in C2_ALPHA_VALUES:
    t0 = time.time()
    recs = {}
    for u in eval_users:
        seen = user_train_all_items.get(u, set())
        # CF scores
        cf_scores = {}
        if u in u2i_svd:
            uidx = u2i_svd[u]
            s = svd_score_mat[uidx]
            for idx in range(len(s)):
                item = i2r_svd[idx]
                if item not in seen:
                    cf_scores[item] = float(s[idx])
        # Content scores
        ct_scores = {}
        liked = user_train_positive_items.get(u, set())
        liked_in_model = [recipe_id_to_idx[r] for r in liked if r in recipe_id_to_idx]
        if liked_in_model:
            user_vec = tfidf_matrix[liked_in_model].mean(axis=0)
            user_vec = np.asarray(user_vec).reshape(1, -1)
            sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
            for idx in range(len(sims)):
                rid = int(recipe_ids_array[idx])
                if rid not in seen:
                    ct_scores[rid] = float(sims[idx])

        all_items = set(cf_scores.keys()) | set(ct_scores.keys())
        if not all_items:
            recs[u] = popularity_recs_for_users(pop_ranked, [u])[u]
            continue
        items_list = list(all_items)
        cf_vals = np.array([cf_scores.get(r, 0.0) for r in items_list])
        ct_vals = np.array([ct_scores.get(r, 0.0) for r in items_list])
        cf_min, cf_max = cf_vals.min(), cf_vals.max()
        ct_min, ct_max = ct_vals.min(), ct_vals.max()
        cf_norm = (cf_vals - cf_min) / (cf_max - cf_min) if cf_max > cf_min else np.zeros_like(cf_vals)
        ct_norm = (ct_vals - ct_min) / (ct_max - ct_min) if ct_max > ct_min else np.zeros_like(ct_vals)
        hybrid = alpha * cf_norm + (1 - alpha) * ct_norm
        ranked = np.argsort(-hybrid)
        recs[u] = [(items_list[idx], float(hybrid[idx])) for idx in ranked[:K]]
    runtime = time.time() - t0
    agg, _ = evaluate_model(f"C2_a{int(alpha*100)}", f"Weighted hybrid alpha={alpha}", "C2", "filtered_temporal", recs, runtime)
    agg["parameters"] = json.dumps({"alpha_cf": alpha, "alpha_content": round(1-alpha, 2)})
    agg["notes"] = "Weighted hybrid: alpha*norm(CF) + (1-alpha)*norm(content)"
    metrics_rows.append(agg)
    tuning_rows.append(agg)
    print(f"  C2_a{int(alpha*100)}: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

c2_recs = recs

# ==== C3: Switching Hybrid ====
print("\n=== C3: Switching Hybrid ===")
for thr in C3_THRESHOLD_VALUES:
    t0 = time.time()
    recs = {}
    cf_count = 0
    ct_count = 0
    for u in eval_users:
        n_hist = user_train_count.get(u, 0)
        seen = user_train_all_items.get(u, set())
        if n_hist >= thr and u in u2i_svd:
            uidx = u2i_svd[u]
            s = svd_score_mat[uidx]
            idx_rank = np.argsort(-s)
            out = []
            for idx in idx_rank:
                item = i2r_svd[int(idx)]
                if item in seen:
                    continue
                out.append((item, float(s[idx])))
                if len(out) >= K:
                    break
            recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
            cf_count += 1
        else:
            liked = user_train_positive_items.get(u, set())
            liked_in_model = [recipe_id_to_idx[r] for r in liked if r in recipe_id_to_idx]
            if not liked_in_model:
                recs[u] = popularity_recs_for_users(pop_ranked, [u])[u]
                ct_count += 1
                continue
            user_vec = tfidf_matrix[liked_in_model].mean(axis=0)
            user_vec = np.asarray(user_vec).reshape(1, -1)
            sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
            idx_rank = np.argsort(-sims)
            out = []
            for idx in idx_rank:
                rid = int(recipe_ids_array[idx])
                if rid in seen:
                    continue
                out.append((rid, float(sims[idx])))
                if len(out) >= K:
                    break
            recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
            ct_count += 1
    runtime = time.time() - t0
    agg, _ = evaluate_model(f"C3_t{thr}", f"Switching hybrid threshold={thr}", "C3", "filtered_temporal", recs, runtime)
    agg["parameters"] = json.dumps({"switch_threshold": thr})
    agg["notes"] = "Switching hybrid: CF if history>=threshold, else content"
    metrics_rows.append(agg)
    tuning_rows.append(agg)
    print(f"  C3_t{thr}: NDCG@{K}={agg['ndcg_at_k']:.6f}, CF={cf_count}, content={ct_count}, runtime={runtime:.1f}s")

c3_recs = recs

# ==== C4: Reciprocal Rank Fusion ====
print("\n=== C4: Reciprocal Rank Fusion ===")
N_CANDIDATES = 200
for rrf_k in C4_RRF_K_VALUES:
    t0 = time.time()
    recs = {}
    for u in eval_users:
        seen = user_train_all_items.get(u, set())
        # CF ranking
        cf_ranked = []
        if u in u2i_svd:
            uidx = u2i_svd[u]
            s = svd_score_mat[uidx]
            idx_rank = np.argsort(-s)
            for idx in idx_rank:
                item = i2r_svd[int(idx)]
                if item in seen:
                    continue
                cf_ranked.append(item)
                if len(cf_ranked) >= N_CANDIDATES:
                    break
        # Content ranking
        ct_ranked = []
        liked = user_train_positive_items.get(u, set())
        liked_in_model = [recipe_id_to_idx[r] for r in liked if r in recipe_id_to_idx]
        if liked_in_model:
            user_vec = tfidf_matrix[liked_in_model].mean(axis=0)
            user_vec = np.asarray(user_vec).reshape(1, -1)
            sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
            idx_rank_ct = np.argsort(-sims)
            for idx in idx_rank_ct:
                rid = int(recipe_ids_array[idx])
                if rid in seen:
                    continue
                ct_ranked.append(rid)
                if len(ct_ranked) >= N_CANDIDATES:
                    break

        if not cf_ranked and not ct_ranked:
            recs[u] = popularity_recs_for_users(pop_ranked, [u])[u]
            continue
        cf_rank_map = {item: rank for rank, item in enumerate(cf_ranked, start=1)}
        ct_rank_map = {item: rank for rank, item in enumerate(ct_ranked, start=1)}
        default_rank = N_CANDIDATES + 1
        all_items = set(cf_ranked) | set(ct_ranked)
        rrf_scores = {}
        for item in all_items:
            r_cf = cf_rank_map.get(item, default_rank)
            r_ct = ct_rank_map.get(item, default_rank)
            rrf_scores[item] = 1.0 / (rrf_k + r_cf) + 1.0 / (rrf_k + r_ct)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:K]
        recs[u] = ranked
    runtime = time.time() - t0
    agg, _ = evaluate_model(f"C4_rrf{rrf_k}", f"RRF hybrid k={rrf_k}", "C4", "filtered_temporal", recs, runtime)
    agg["parameters"] = json.dumps({"rrf_k": rrf_k, "n_candidates": N_CANDIDATES})
    agg["notes"] = "Reciprocal Rank Fusion of CF and content top-N lists"
    metrics_rows.append(agg)
    tuning_rows.append(agg)
    print(f"  C4_rrf{rrf_k}: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

c4_recs = recs

# ==== Final comparison ====
print("\n" + "=" * 60)
print("FINAL RESULTS (DEBUG MODE)")
print("=" * 60)
metrics_df = pd.DataFrame(metrics_rows)
print(metrics_df[["model_id", "model_name", "precision_at_k", "recall_at_k",
                   "ndcg_at_k", "catalog_coverage_at_k", "runtime_seconds"]].to_string(index=False))

best = metrics_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
print(f"\nBest model: {best['model_id']}  NDCG@{K}: {best['ndcg_at_k']:.6f}")

# Save
metrics_df.to_csv(os.path.join(RESULTS_DIR, "debug_version_c_metrics.csv"), index=False)
pd.DataFrame(tuning_rows).to_csv(os.path.join(RESULTS_DIR, "debug_version_c_tuning_results.csv"), index=False)
print("\nDebug results saved to", RESULTS_DIR)
print("Done!")
