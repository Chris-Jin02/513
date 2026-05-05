"""Version C FULL RUN v2 - memory-efficient, no full score matrix."""
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
print("Loading data...", flush=True)
train = pd.read_csv(os.path.join(DATA_DIR, "interactions_train_filtered.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "interactions_test_filtered.csv"))
recipe_model_table = pd.read_csv(os.path.join(DATA_DIR, "recipe_model_table.csv"))
print(f"train: {train.shape}, test: {test.shape}", flush=True)

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
print(f"Eval users: {len(eval_users)}", flush=True)

# ---- Eval functions ----
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
    dcg = sum(1.0 / np.log2(r + 1) for r, item in enumerate(rec, 1) if item in rel)
    ideal = min(len(rel), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(rec, rel, k):
    return 1.0 if any(x in rel for x in rec[:k]) else 0.0

def evaluate_model(model_id, model_name, phase, recommendations, runtime_seconds):
    per_user_rows = []
    unique_rec = set()
    for u in eval_users:
        rel = user_test_relevant_items.get(u, set())
        rec_items = recommendations.get(u, [])
        rec_ids = [x[0] if isinstance(x, tuple) else x for x in rec_items][:K]
        unique_rec.update(rec_ids)
        per_user_rows.append({"user_id": u, "model_id": model_id,
            "precision_at_k": precision_at_k(rec_ids, rel, K),
            "recall_at_k": recall_at_k(rec_ids, rel, K),
            "ndcg_at_k": ndcg_at_k(rec_ids, rel, K),
            "hit_at_k": hit_at_k(rec_ids, rel, K)})
    per_user_df = pd.DataFrame(per_user_rows)
    n = len(per_user_df)
    agg = {"version": "C", "phase": phase, "model_id": model_id, "model_name": model_name,
           "split": "filtered_temporal", "k": K, "evaluated_users": n,
           "precision_at_k": float(per_user_df["precision_at_k"].mean()),
           "recall_at_k": float(per_user_df["recall_at_k"].mean()),
           "ndcg_at_k": float(per_user_df["ndcg_at_k"].mean()),
           "hit_at_k": float(per_user_df["hit_at_k"].mean()),
           "catalog_coverage_at_k": float(len(unique_rec) / train["recipe_id"].nunique()),
           "runtime_seconds": float(runtime_seconds),
           "seconds_per_user": float(runtime_seconds / n) if n > 0 else 0}
    return agg, per_user_df

# ---- Popularity fallback ----
pop_global_mean = train["rating"].mean()
pop = train.groupby("recipe_id").agg(R=("rating", "mean"), v=("rating", "count")).reset_index()
pop["score"] = (pop["v"] / (pop["v"] + 100)) * pop["R"] + (100 / (pop["v"] + 100)) * pop_global_mean
pop = pop.sort_values("score", ascending=False)
pop_ranked = list(zip(pop["recipe_id"].tolist(), pop["score"].tolist()))

def pop_recs(u):
    seen = user_train_all_items.get(u, set())
    out = []
    for rid, sc in pop_ranked:
        if rid in seen: continue
        out.append((rid, sc))
        if len(out) >= K: break
    return out

# ---- Recipe text setup ----
text_cols = [c for c in ["name", "description", "tags_text", "ingredients_text"] if c in recipe_model_table.columns]
recipe_id_col = "recipe_id" if "recipe_id" in recipe_model_table.columns else "id"
recipe_model_table["combined_text_v"] = recipe_model_table[text_cols].fillna("").agg(" ".join, axis=1)
recipe_id_to_idx = {rid: idx for idx, rid in enumerate(recipe_model_table[recipe_id_col].values)}
recipe_ids_array = recipe_model_table[recipe_id_col].values
print(f"Text cols: {text_cols}, ID col: {recipe_id_col}", flush=True)

metrics_rows = []
per_user_frames = []
runtime_rows = []

# ==== C0: SVD CF (per-user scoring, no full matrix) ====
print("\n=== C0: SVD CF Baseline (64d, per-user) ===", flush=True)
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
user_emb = svd.fit_transform(mat)  # (n_users, 64)
item_emb = svd.components_.T       # (n_items, 64)
print(f"  SVD fit done in {time.time()-t0:.1f}s, user_emb={user_emb.shape}, item_emb={item_emb.shape}", flush=True)

def svd_top_k_for_user(u, k=K):
    """Compute scores per-user without full matrix."""
    if u not in u2i_svd:
        return pop_recs(u)
    uidx = u2i_svd[u]
    scores = user_emb[uidx] @ item_emb.T  # (n_items,) - one row only
    seen = user_train_all_items.get(u, set())
    idx_rank = np.argsort(-scores)
    out = []
    for idx in idx_rank:
        item = i2r_svd[int(idx)]
        if item in seen: continue
        out.append((item, float(scores[idx])))
        if len(out) >= k: break
    return out if out else pop_recs(u)

recs = {}
for i, u in enumerate(eval_users):
    recs[u] = svd_top_k_for_user(u)
    if (i + 1) % 2000 == 0:
        print(f"  C0: {i+1}/{len(eval_users)} users", flush=True)
runtime = time.time() - t0
agg, pu = evaluate_model("C0_svd64", "SVD CF baseline 64", "C0", recs, runtime)
agg["parameters"] = json.dumps({"n_components": 64})
agg["notes"] = "SVD collaborative filtering baseline (per-user scoring)"
metrics_rows.append(agg); per_user_frames.append(pu)
runtime_rows.append({"phase": "C0", "model_id": agg["model_id"], "runtime_seconds": runtime,
                      "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c0_recs = recs
print(f"  C0 done: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s", flush=True)

# ==== C1: TF-IDF Content (10000 features) ====
print("\n=== C1: TF-IDF Content Baseline ===", flush=True)
t0 = time.time()
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(recipe_model_table["combined_text_v"])
print(f"  TF-IDF fit done in {time.time()-t0:.1f}s, shape={tfidf_matrix.shape}", flush=True)

def content_top_k_for_user(u, k=K):
    liked = user_train_positive_items.get(u, set())
    liked_in = [recipe_id_to_idx[r] for r in liked if r in recipe_id_to_idx]
    if not liked_in:
        return pop_recs(u)
    user_vec = tfidf_matrix[liked_in].mean(axis=0)
    user_vec = np.asarray(user_vec).reshape(1, -1)
    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
    seen = user_train_all_items.get(u, set())
    idx_rank = np.argsort(-sims)
    out = []
    for idx in idx_rank:
        rid = int(recipe_ids_array[idx])
        if rid in seen: continue
        out.append((rid, float(sims[idx])))
        if len(out) >= k: break
    return out if out else pop_recs(u)

recs = {}
for i, u in enumerate(eval_users):
    recs[u] = content_top_k_for_user(u)
    if (i + 1) % 1000 == 0:
        print(f"  C1: {i+1}/{len(eval_users)} users, {time.time()-t0:.0f}s", flush=True)
runtime = time.time() - t0
agg, pu = evaluate_model("C1_tfidf10k", "TF-IDF content baseline 10000", "C1", recs, runtime)
agg["parameters"] = json.dumps({"max_features": 10000})
agg["notes"] = "TF-IDF user-profile content baseline"
metrics_rows.append(agg); per_user_frames.append(pu)
runtime_rows.append({"phase": "C1", "model_id": agg["model_id"], "runtime_seconds": runtime,
                      "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c1_recs = recs
print(f"  C1 done: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s", flush=True)

# ==== C2: Weighted Hybrid (top-200 candidates) ====
print("\n=== C2: Weighted Hybrid (alpha=0.7) ===", flush=True)
alpha = 0.7
t0 = time.time()
recs = {}
for i, u in enumerate(eval_users):
    seen = user_train_all_items.get(u, set())
    cf_top = svd_top_k_for_user(u, k=200)
    ct_top = content_top_k_for_user(u, k=200)
    cf_dict = dict(cf_top)
    ct_dict = dict(ct_top)
    all_items = set(cf_dict.keys()) | set(ct_dict.keys())
    if not all_items:
        recs[u] = pop_recs(u); continue
    items_list = list(all_items)
    cf_vals = np.array([cf_dict.get(r, 0.0) for r in items_list])
    ct_vals = np.array([ct_dict.get(r, 0.0) for r in items_list])
    cf_mn, cf_mx = cf_vals.min(), cf_vals.max()
    ct_mn, ct_mx = ct_vals.min(), ct_vals.max()
    cf_n = (cf_vals - cf_mn) / (cf_mx - cf_mn) if cf_mx > cf_mn else np.zeros_like(cf_vals)
    ct_n = (ct_vals - ct_mn) / (ct_mx - ct_mn) if ct_mx > ct_mn else np.zeros_like(ct_vals)
    hybrid = alpha * cf_n + (1 - alpha) * ct_n
    ranked = np.argsort(-hybrid)
    recs[u] = [(items_list[idx], float(hybrid[idx])) for idx in ranked[:K]]
    if (i + 1) % 1000 == 0:
        print(f"  C2: {i+1}/{len(eval_users)} users, {time.time()-t0:.0f}s", flush=True)
runtime = time.time() - t0
agg, pu = evaluate_model("C2_a70", "Weighted hybrid alpha=0.7", "C2", recs, runtime)
agg["parameters"] = json.dumps({"alpha_cf": 0.7, "alpha_content": 0.3})
agg["notes"] = "Weighted hybrid: 0.7*norm(CF) + 0.3*norm(content), top-200 candidates"
metrics_rows.append(agg); per_user_frames.append(pu)
runtime_rows.append({"phase": "C2", "model_id": agg["model_id"], "runtime_seconds": runtime,
                      "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c2_recs = recs
print(f"  C2 done: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s", flush=True)

# ==== C3: Switching Hybrid ====
print("\n=== C3: Switching Hybrid (threshold=10) ===", flush=True)
thr = 10
t0 = time.time()
recs = {}
cf_count = 0; ct_count = 0
for i, u in enumerate(eval_users):
    n_hist = user_train_count.get(u, 0)
    if n_hist >= thr and u in u2i_svd:
        recs[u] = svd_top_k_for_user(u)
        cf_count += 1
    else:
        recs[u] = content_top_k_for_user(u)
        ct_count += 1
    if (i + 1) % 2000 == 0:
        print(f"  C3: {i+1}/{len(eval_users)} users, {time.time()-t0:.0f}s", flush=True)
runtime = time.time() - t0
agg, pu = evaluate_model("C3_t10", "Switching hybrid threshold=10", "C3", recs, runtime)
agg["parameters"] = json.dumps({"switch_threshold": 10})
agg["notes"] = f"Switching: CF if history>=10 else content. CF={cf_count}, content={ct_count}"
metrics_rows.append(agg); per_user_frames.append(pu)
runtime_rows.append({"phase": "C3", "model_id": agg["model_id"], "runtime_seconds": runtime,
                      "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c3_recs = recs
print(f"  C3 done: NDCG@{K}={agg['ndcg_at_k']:.6f}, CF={cf_count}, ct={ct_count}, runtime={runtime:.1f}s", flush=True)

# ==== C4: RRF Hybrid ====
print("\n=== C4: RRF Hybrid (k=60) ===", flush=True)
rrf_k = 60
t0 = time.time()
recs = {}
for i, u in enumerate(eval_users):
    cf_ranked = [x[0] for x in svd_top_k_for_user(u, k=200)]
    ct_ranked = [x[0] for x in content_top_k_for_user(u, k=200)]
    if not cf_ranked and not ct_ranked:
        recs[u] = pop_recs(u); continue
    cf_rank_map = {item: rank for rank, item in enumerate(cf_ranked, 1)}
    ct_rank_map = {item: rank for rank, item in enumerate(ct_ranked, 1)}
    default_rank = 201
    all_items = set(cf_ranked) | set(ct_ranked)
    rrf_scores = {}
    for item in all_items:
        rrf_scores[item] = 1.0 / (rrf_k + cf_rank_map.get(item, default_rank)) + \
                           1.0 / (rrf_k + ct_rank_map.get(item, default_rank))
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:K]
    recs[u] = ranked
    if (i + 1) % 1000 == 0:
        print(f"  C4: {i+1}/{len(eval_users)} users, {time.time()-t0:.0f}s", flush=True)
runtime = time.time() - t0
agg, pu = evaluate_model("C4_rrf60", "RRF hybrid k=60", "C4", recs, runtime)
agg["parameters"] = json.dumps({"rrf_k": 60, "n_candidates": 200})
agg["notes"] = "Reciprocal Rank Fusion of CF and content top-200 lists"
metrics_rows.append(agg); per_user_frames.append(pu)
runtime_rows.append({"phase": "C4", "model_id": agg["model_id"], "runtime_seconds": runtime,
                      "evaluated_users": agg["evaluated_users"], "seconds_per_user": agg["seconds_per_user"]})
c4_recs = recs
print(f"  C4 done: NDCG@{K}={agg['ndcg_at_k']:.6f}, runtime={runtime:.1f}s", flush=True)

# ==== Save ====
print("\n" + "=" * 60, flush=True)
print("FINAL RESULTS (FULL RUN)", flush=True)
print("=" * 60, flush=True)
metrics_df = pd.DataFrame(metrics_rows)
print(metrics_df[["model_id", "model_name", "precision_at_k", "recall_at_k",
                   "ndcg_at_k", "hit_at_k", "catalog_coverage_at_k", "runtime_seconds"]].to_string(index=False), flush=True)

best = metrics_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
print(f"\nBest model: {best['model_id']}  NDCG@{K}: {best['ndcg_at_k']:.6f}", flush=True)

metrics_df.to_csv(os.path.join(RESULTS_DIR, "version_c_metrics.csv"), index=False)
pd.DataFrame(runtime_rows).to_csv(os.path.join(RESULTS_DIR, "version_c_phase_runtime.csv"), index=False)
per_user_df = pd.concat(per_user_frames, ignore_index=True)
per_user_df.to_csv(os.path.join(RESULTS_DIR, "version_c_per_user_metrics.csv"), index=False)

config = {"timestamp": datetime.utcnow().isoformat() + "Z", "debug_mode": False, "full_run": True,
          "k": K, "positive_threshold": POSITIVE_THRESHOLD, "selected_final_model": best["model_id"]}
with open(os.path.join(RESULTS_DIR, "version_c_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Example recs
final_recs = {"C0": c0_recs, "C1": c1_recs, "C2": c2_recs, "C3": c3_recs, "C4": c4_recs}
best_recs = final_recs[best["phase"]]
rows = []
for u in eval_users[:10]:
    hist = sorted(list(user_train_all_items.get(u, set())))[:50]
    rel = sorted(list(user_test_relevant_items.get(u, set())))
    rec_ids = [x[0] if isinstance(x, tuple) else x for x in best_recs.get(u, [])]
    rows.append({"user_id": u, "model_id": best["model_id"],
                 "user_history_items": "|".join(map(str, hist)),
                 "recommended_items": "|".join(map(str, rec_ids)),
                 "relevant_test_items": "|".join(map(str, rel))})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "version_c_example_recommendations.csv"), index=False)

# Model notes
with open(os.path.join(RESULTS_DIR, "version_c_model_notes.md"), "w") as f:
    f.write("# Version C Model Notes\n\n")
    for _, row in metrics_df.iterrows():
        f.write(f"## {row['model_id']} - {row['model_name']}\n")
        f.write(f"- precision@{K}: {row['precision_at_k']:.6f}\n")
        f.write(f"- recall@{K}: {row['recall_at_k']:.6f}\n")
        f.write(f"- ndcg@{K}: {row['ndcg_at_k']:.6f}\n")
        f.write(f"- hit@{K}: {row['hit_at_k']:.6f}\n")
        f.write(f"- coverage@{K}: {row['catalog_coverage_at_k']:.6f}\n")
        f.write(f"- runtime: {row['runtime_seconds']:.1f}s\n\n")

print("\nAll results saved to", RESULTS_DIR, flush=True)
print("Done!", flush=True)
