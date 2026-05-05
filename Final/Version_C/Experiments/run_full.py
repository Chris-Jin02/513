"""Version C FULL RUN - lightweight version with best params only."""
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

POSITIVE_THRESHOLD = 4
K = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- Load data ----
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "interactions_train_filtered.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "interactions_test_filtered.csv"))
interactions_filtered = pd.read_csv(os.path.join(DATA_DIR, "interactions_filtered.csv"))
recipe_model_table = pd.read_csv(os.path.join(DATA_DIR, "recipe_model_table.csv"))
print("train shape:", train.shape)
print("test shape:", test.shape)

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
print("Eval users:", len(eval_users))

# ---- Evaluation functions ----
def precision_at_k(rec, rel, k):
    rec = rec[:k]
    if k == 0 or len(rec) == 0: return 0.0
    return sum(1 for x in rec if x in rel) / k

def recall_at_k(rec, rel, k):
    if len(rel) == 0: return 0.0
    return sum(1 for x in rec[:k] if x in rel) / len(rel)

def ndcg_at_k(rec, rel, k):
    rec = rec[:k]
    if len(rec) == 0 or len(rel) == 0: return 0.0
    dcg = sum(1.0 / np.log2(rank + 1) for rank, item in enumerate(rec, start=1) if item in rel)
    ideal = min(len(rel), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(rec, rel, k):
    return 1.0 if any(x in rel for x in rec[:k]) else 0.0

def evaluate_model(model_id, model_name, phase, recommendations, runtime_seconds):
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
           "split": "filtered_temporal", "k": K, "evaluated_users": eval_count,
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
            if rid in seen: continue
            recs.append((rid, float(score)))
            if len(recs) >= K: break
        out[u] = recs
    return out

print("Popularity fallback ready.")

# ---- Recipe text setup ----
text_cols = [col for col in ["name", "description", "tags_text", "ingredients_text"] if col in recipe_model_table.columns]
if not text_cols:
    text_cols = [col for col in recipe_model_table.columns if recipe_model_table[col].dtype == object][:4]
print("Text columns:", text_cols)

recipe_id_col = "recipe_id" if "recipe_id" in recipe_model_table.columns else "id"
recipe_model_table["combined_text_v"] = recipe_model_table[text_cols].fillna("").agg(" ".join, axis=1)
recipe_id_to_idx = {rid: idx for idx, rid in enumerate(recipe_model_table[recipe_id_col].values)}
recipe_ids_array = recipe_model_table[recipe_id_col].values

metrics_rows = []
per_user_frames = []
runtime_rows = []

# ==== C0: SVD CF (64 components - best from B4) ====
print("\n=== C0: SVD CF Baseline (64 components) ===")
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
svd = TruncatedSVD(n_components=64, random_state=RANDOM_STATE)
user_emb = svd.fit_transform(mat)
item_emb = svd.components_.T
svd_score_mat = user_emb @ item_emb.T
print(f"  SVD fit done in {time.time()-t0:.1f}s")

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
        if item in seen: continue
        out.append((item, float(s[idx])))
        if len(out) >= K: break
    recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
runtime = time.time() - t0
agg, pu = evaluate_model("C0_svd64", "SVD CF baseline 64", "C0", recs, runtime)
agg["parameters"] = json.dumps({"n_components": 64})
agg["notes"] = "SVD collaborative filtering baseline"
metrics_rows.append(agg)
per_user_frames.append(pu)
runtime_rows.append({"phase": "C0", "model_id": agg["model_id"], "runtime_seconds": runtime, "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c0_recs = recs
print(f"  C0: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

# ==== C1: TF-IDF Content (10000 features) ====
print("\n=== C1: TF-IDF Content Baseline (10000 features) ===")
t0 = time.time()
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(recipe_model_table["combined_text_v"])
print(f"  TF-IDF fit done in {time.time()-t0:.1f}s")

recs = {}
batch_size = 500
for batch_start in range(0, len(eval_users), batch_size):
    batch_users = eval_users[batch_start:batch_start + batch_size]
    for u in batch_users:
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
            if rid in seen: continue
            out.append((rid, float(sims[idx])))
            if len(out) >= K: break
        recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
    print(f"  C1 progress: {min(batch_start+batch_size, len(eval_users))}/{len(eval_users)} users")
runtime = time.time() - t0
agg, pu = evaluate_model("C1_tfidf10k", "TF-IDF content baseline 10000", "C1", recs, runtime)
agg["parameters"] = json.dumps({"max_features": 10000})
agg["notes"] = "TF-IDF user-profile content baseline"
metrics_rows.append(agg)
per_user_frames.append(pu)
runtime_rows.append({"phase": "C1", "model_id": agg["model_id"], "runtime_seconds": runtime, "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c1_recs = recs
print(f"  C1: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

# ==== C2: Weighted Hybrid (alpha=0.7) ====
print("\n=== C2: Weighted Hybrid (alpha=0.7) ===")
alpha = 0.7
t0 = time.time()
recs = {}
for i, u in enumerate(eval_users):
    seen = user_train_all_items.get(u, set())
    # CF scores - only get top candidates instead of all items
    cf_top = []
    if u in u2i_svd:
        uidx = u2i_svd[u]
        s = svd_score_mat[uidx]
        idx_rank = np.argsort(-s)
        for idx in idx_rank:
            item = i2r_svd[int(idx)]
            if item in seen: continue
            cf_top.append((item, float(s[idx])))
            if len(cf_top) >= 200: break
    # Content scores - only get top candidates
    ct_top = []
    liked = user_train_positive_items.get(u, set())
    liked_in_model = [recipe_id_to_idx[r] for r in liked if r in recipe_id_to_idx]
    if liked_in_model:
        user_vec = tfidf_matrix[liked_in_model].mean(axis=0)
        user_vec = np.asarray(user_vec).reshape(1, -1)
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        idx_rank = np.argsort(-sims)
        for idx in idx_rank:
            rid = int(recipe_ids_array[idx])
            if rid in seen: continue
            ct_top.append((rid, float(sims[idx])))
            if len(ct_top) >= 200: break

    # Merge candidates
    cf_dict = dict(cf_top)
    ct_dict = dict(ct_top)
    all_items = set(cf_dict.keys()) | set(ct_dict.keys())
    if not all_items:
        recs[u] = popularity_recs_for_users(pop_ranked, [u])[u]
        continue
    items_list = list(all_items)
    cf_vals = np.array([cf_dict.get(r, 0.0) for r in items_list])
    ct_vals = np.array([ct_dict.get(r, 0.0) for r in items_list])
    cf_min, cf_max = cf_vals.min(), cf_vals.max()
    ct_min, ct_max = ct_vals.min(), ct_vals.max()
    cf_norm = (cf_vals - cf_min) / (cf_max - cf_min) if cf_max > cf_min else np.zeros_like(cf_vals)
    ct_norm = (ct_vals - ct_min) / (ct_max - ct_min) if ct_max > ct_min else np.zeros_like(ct_vals)
    hybrid = alpha * cf_norm + (1 - alpha) * ct_norm
    ranked = np.argsort(-hybrid)
    recs[u] = [(items_list[idx], float(hybrid[idx])) for idx in ranked[:K]]
    if (i + 1) % 1000 == 0:
        print(f"  C2 progress: {i+1}/{len(eval_users)} users, {time.time()-t0:.0f}s elapsed")
runtime = time.time() - t0
agg, pu = evaluate_model("C2_a70", "Weighted hybrid alpha=0.7", "C2", recs, runtime)
agg["parameters"] = json.dumps({"alpha_cf": 0.7, "alpha_content": 0.3})
agg["notes"] = "Weighted hybrid: alpha*norm(CF) + (1-alpha)*norm(content), top-200 candidates"
metrics_rows.append(agg)
per_user_frames.append(pu)
runtime_rows.append({"phase": "C2", "model_id": agg["model_id"], "runtime_seconds": runtime, "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c2_recs = recs
print(f"  C2: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

# ==== C3: Switching Hybrid (threshold=10) ====
print("\n=== C3: Switching Hybrid (threshold=10) ===")
thr = 10
t0 = time.time()
recs = {}
cf_count = 0; ct_count = 0
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
            if item in seen: continue
            out.append((item, float(s[idx])))
            if len(out) >= K: break
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
            if rid in seen: continue
            out.append((rid, float(sims[idx])))
            if len(out) >= K: break
        recs[u] = out if out else popularity_recs_for_users(pop_ranked, [u])[u]
        ct_count += 1
runtime = time.time() - t0
agg, pu = evaluate_model("C3_t10", "Switching hybrid threshold=10", "C3", recs, runtime)
agg["parameters"] = json.dumps({"switch_threshold": 10})
agg["notes"] = f"Switching hybrid: CF if history>=10, else content. CF={cf_count}, content={ct_count}"
metrics_rows.append(agg)
per_user_frames.append(pu)
runtime_rows.append({"phase": "C3", "model_id": agg["model_id"], "runtime_seconds": runtime, "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c3_recs = recs
print(f"  C3: NDCG@{K}={agg['ndcg_at_k']:.6f}, CF={cf_count}, content={ct_count}, runtime={runtime:.1f}s")

# ==== C4: Reciprocal Rank Fusion (k=60) ====
print("\n=== C4: Reciprocal Rank Fusion (k=60) ===")
rrf_k = 60
N_CANDIDATES = 200
t0 = time.time()
recs = {}
for i, u in enumerate(eval_users):
    seen = user_train_all_items.get(u, set())
    # CF ranking
    cf_ranked = []
    if u in u2i_svd:
        uidx = u2i_svd[u]
        s = svd_score_mat[uidx]
        idx_rank = np.argsort(-s)
        for idx in idx_rank:
            item = i2r_svd[int(idx)]
            if item in seen: continue
            cf_ranked.append(item)
            if len(cf_ranked) >= N_CANDIDATES: break
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
            if rid in seen: continue
            ct_ranked.append(rid)
            if len(ct_ranked) >= N_CANDIDATES: break
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
    if (i + 1) % 1000 == 0:
        print(f"  C4 progress: {i+1}/{len(eval_users)} users, {time.time()-t0:.0f}s elapsed")
runtime = time.time() - t0
agg, pu = evaluate_model("C4_rrf60", "RRF hybrid k=60", "C4", recs, runtime)
agg["parameters"] = json.dumps({"rrf_k": 60, "n_candidates": 200})
agg["notes"] = "Reciprocal Rank Fusion of CF and content top-N lists"
metrics_rows.append(agg)
per_user_frames.append(pu)
runtime_rows.append({"phase": "C4", "model_id": agg["model_id"], "runtime_seconds": runtime, "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c4_recs = recs
print(f"  C4: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s")

# ==== Save results ====
print("\n" + "=" * 60)
print("FINAL RESULTS (FULL RUN)")
print("=" * 60)
metrics_df = pd.DataFrame(metrics_rows)
print(metrics_df[["model_id", "model_name", "precision_at_k", "recall_at_k",
                   "ndcg_at_k", "hit_at_k", "catalog_coverage_at_k", "runtime_seconds"]].to_string(index=False))

best = metrics_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
print(f"\nBest model: {best['model_id']}  NDCG@{K}: {best['ndcg_at_k']:.6f}")

metrics_df.to_csv(os.path.join(RESULTS_DIR, "version_c_metrics.csv"), index=False)
pd.DataFrame(runtime_rows).to_csv(os.path.join(RESULTS_DIR, "version_c_phase_runtime.csv"), index=False)
per_user_df = pd.concat(per_user_frames, ignore_index=True)
per_user_df.to_csv(os.path.join(RESULTS_DIR, "version_c_per_user_metrics.csv"), index=False)

# Save config
config = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "debug_mode": False, "full_run": True,
    "k": K, "positive_threshold": POSITIVE_THRESHOLD,
    "selected_final_model": best["model_id"],
    "c0_n_components": 64, "c1_max_features": 10000,
    "c2_alpha": 0.7, "c3_threshold": 10, "c4_rrf_k": 60,
}
with open(os.path.join(RESULTS_DIR, "version_c_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Save example recommendations
final_recs = {"C0": c0_recs, "C1": c1_recs, "C2": c2_recs, "C3": c3_recs, "C4": c4_recs}
best_recs = final_recs[best["phase"]]
example_users = eval_users[:10]
example_rows = []
for u in example_users:
    hist = sorted(list(user_train_all_items.get(u, set())))[:50]
    rel = sorted(list(user_test_relevant_items.get(u, set())))
    rec_items = best_recs.get(u, [])
    rec_ids = [x[0] if isinstance(x, tuple) else x for x in rec_items]
    example_rows.append({
        "user_id": u, "model_id": best["model_id"],
        "user_history_items": "|".join(map(str, hist)),
        "recommended_items": "|".join(map(str, rec_ids)),
        "relevant_test_items": "|".join(map(str, rel)),
        "explanation": "Hybrid recommendation combining CF and content signals.",
    })
pd.DataFrame(example_rows).to_csv(os.path.join(RESULTS_DIR, "version_c_example_recommendations.csv"), index=False)

# Save top-10 for all users
top_rows = []
for u in eval_users:
    rec_items = best_recs.get(u, [])
    for rk, rs in enumerate(rec_items[:K], start=1):
        rid, score = rs if isinstance(rs, tuple) else (rs, np.nan)
        top_rows.append({"user_id": u, "rank": rk, "recipe_id": rid, "score": float(score),
                         "model_id": best["model_id"], "model_name": best["model_name"]})
pd.DataFrame(top_rows).to_csv(os.path.join(RESULTS_DIR, "version_c_top10_recommendations.csv"), index=False)

# Save model notes
with open(os.path.join(RESULTS_DIR, "version_c_model_notes.md"), "w") as f:
    f.write("# Version C Model Notes\n\n")
    for _, row in metrics_df.iterrows():
        f.write(f"## {row['model_id']} - {row['model_name']}\n")
        f.write(f"- precision@{K}: {row['precision_at_k']:.6f}\n")
        f.write(f"- recall@{K}: {row['recall_at_k']:.6f}\n")
        f.write(f"- ndcg@{K}: {row['ndcg_at_k']:.6f}\n")
        f.write(f"- hit@{K}: {row['hit_at_k']:.6f}\n")
        f.write(f"- coverage@{K}: {row['catalog_coverage_at_k']:.6f}\n")
        f.write(f"- runtime: {row['runtime_seconds']:.1f}s\n")
        f.write(f"- parameters: {row.get('parameters','{}')}\n")
        f.write(f"- notes: {row.get('notes','')}\n\n")

print("\nAll results saved to", RESULTS_DIR)
print("Done!")
