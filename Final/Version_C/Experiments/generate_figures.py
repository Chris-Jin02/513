"""Generate all Version C figures + cross-version comparison for presentation."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "../Results"
FIGURES_DIR = "../Results/Figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load Version C metrics
c_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "version_c_metrics.csv"))
c_runtime = pd.read_csv(os.path.join(RESULTS_DIR, "version_c_phase_runtime.csv"))
c_per_user = pd.read_csv(os.path.join(RESULTS_DIR, "version_c_per_user_metrics.csv"))

print("Version C metrics loaded:", c_metrics.shape)
print(c_metrics[["model_id", "ndcg_at_k", "catalog_coverage_at_k"]].to_string(index=False))

plot_df = c_metrics.sort_values("model_id")

# ---- 1. Model Metrics Bar Chart ----
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(plot_df))
w = 0.25
ax.bar(x - w, plot_df["precision_at_k"], w, label="Precision@10", color="#4C72B0")
ax.bar(x, plot_df["recall_at_k"], w, label="Recall@10", color="#55A868")
ax.bar(x + w, plot_df["ndcg_at_k"], w, label="NDCG@10", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model_id"], rotation=15)
ax.set_title("Version C: Model Metrics at K=10", fontsize=14)
ax.set_ylabel("Metric Value")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "version_c_model_metrics_at_10.png"), dpi=150)
plt.close()
print("Saved: version_c_model_metrics_at_10.png")

# ---- 2. Coverage Bar Chart ----
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
ax.bar(plot_df["model_id"], plot_df["catalog_coverage_at_k"], color=colors)
ax.set_title("Version C: Catalog Coverage at K=10", fontsize=14)
ax.set_ylabel("Coverage")
for i, v in enumerate(plot_df["catalog_coverage_at_k"]):
    ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "version_c_catalog_coverage_at_10.png"), dpi=150)
plt.close()
print("Saved: version_c_catalog_coverage_at_10.png")

# ---- 3. Quality vs Runtime Scatter ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(plot_df["seconds_per_user"], plot_df["ndcg_at_k"], s=100, c=colors[:len(plot_df)], zorder=5)
for _, r in plot_df.iterrows():
    ax.annotate(r["model_id"], (r["seconds_per_user"], r["ndcg_at_k"]),
                xytext=(6, 6), textcoords="offset points", fontsize=9)
ax.set_title("Version C: Quality vs Runtime at K=10", fontsize=14)
ax.set_xlabel("Seconds per User")
ax.set_ylabel("NDCG@10")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "version_c_quality_vs_runtime_at_10.png"), dpi=150)
plt.close()
print("Saved: version_c_quality_vs_runtime_at_10.png")

# ---- 4. Hit Rate Bar Chart ----
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(plot_df["model_id"], plot_df["hit_at_k"], color=colors)
ax.set_title("Version C: Hit Rate at K=10", fontsize=14)
ax.set_ylabel("Hit Rate")
for i, v in enumerate(plot_df["hit_at_k"]):
    ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "version_c_hit_rate_at_10.png"), dpi=150)
plt.close()
print("Saved: version_c_hit_rate_at_10.png")

# ---- 5. Phase Runtime ----
fig, ax = plt.subplots(figsize=(8, 4.5))
rt = c_runtime.sort_values("phase")
ax.bar(rt["phase"], rt["runtime_seconds"], color=colors[:len(rt)])
ax.set_title("Version C: Phase Runtime", fontsize=14)
ax.set_ylabel("Runtime (seconds)")
for i, v in enumerate(rt["runtime_seconds"]):
    ax.text(i, v + 5, f"{v:.0f}s", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "version_c_phase_runtime.png"), dpi=150)
plt.close()
print("Saved: version_c_phase_runtime.png")

# ---- 6. Per-user NDCG Boxplot ----
if not c_per_user.empty:
    box_models = sorted(c_per_user["model_id"].unique())
    box_data = [c_per_user.loc[c_per_user["model_id"] == m, "ndcg_at_k"].values for m in box_models]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(box_data, labels=box_models, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(box_models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Version C: Per-user NDCG@10 Distribution", fontsize=14)
    ax.set_ylabel("NDCG@10")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "version_c_per_user_ndcg_boxplot.png"), dpi=150)
    plt.close()
    print("Saved: version_c_per_user_ndcg_boxplot.png")

# ---- 7. NDCG vs Coverage Trade-off ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(plot_df["catalog_coverage_at_k"], plot_df["ndcg_at_k"], s=150, c=colors[:len(plot_df)], zorder=5)
for _, r in plot_df.iterrows():
    ax.annotate(r["model_id"], (r["catalog_coverage_at_k"], r["ndcg_at_k"]),
                xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
ax.set_title("Version C: NDCG vs Coverage Trade-off", fontsize=14)
ax.set_xlabel("Catalog Coverage@10")
ax.set_ylabel("NDCG@10")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "version_c_ndcg_vs_coverage.png"), dpi=150)
plt.close()
print("Saved: version_c_ndcg_vs_coverage.png")

# ======== CROSS-VERSION COMPARISON ========
print("\n=== Cross-Version Comparison ===")

cross_rows = []

# Version A
a_path = "../../Version_A/Results/version_a_metrics.csv"
if os.path.isfile(a_path):
    a_df = pd.read_csv(a_path)
    # Filter K=10 rows and get best by NDCG
    a10 = a_df[a_df["k"] == 10] if "k" in a_df.columns else a_df
    best_a = a10.sort_values("ndcg_at_k", ascending=False).iloc[0]
    cross_rows.append({"version": "A", "model_id": best_a["model_id"],
                       "model_name": best_a.get("model_name", ""),
                       "ndcg_at_k": best_a["ndcg_at_k"],
                       "recall_at_k": best_a["recall_at_k"],
                       "precision_at_k": best_a["precision_at_k"],
                       "catalog_coverage_at_k": best_a["catalog_coverage_at_k"]})
    print(f"  A best: {best_a['model_id']}, NDCG={best_a['ndcg_at_k']:.6f}")

# Version B
b_path = "../../Version_B/Results/version_b_metrics.csv"
if os.path.isfile(b_path):
    b_df = pd.read_csv(b_path)
    best_b = b_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
    cross_rows.append({"version": "B", "model_id": best_b["model_id"],
                       "model_name": best_b.get("model_name", ""),
                       "ndcg_at_k": best_b["ndcg_at_k"],
                       "recall_at_k": best_b["recall_at_k"],
                       "precision_at_k": best_b["precision_at_k"],
                       "catalog_coverage_at_k": best_b["catalog_coverage_at_k"]})
    print(f"  B best: {best_b['model_id']}, NDCG={best_b['ndcg_at_k']:.6f}")

# Version C - use C2 (hybrid) as the representative, not C0 (which is just SVD again)
c2_row = c_metrics[c_metrics["model_id"] == "C2_a70"].iloc[0]
cross_rows.append({"version": "C (hybrid)", "model_id": c2_row["model_id"],
                   "model_name": c2_row["model_name"],
                   "ndcg_at_k": c2_row["ndcg_at_k"],
                   "recall_at_k": c2_row["recall_at_k"],
                   "precision_at_k": c2_row["precision_at_k"],
                   "catalog_coverage_at_k": c2_row["catalog_coverage_at_k"]})
print(f"  C best hybrid: {c2_row['model_id']}, NDCG={c2_row['ndcg_at_k']:.6f}")

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(os.path.join(RESULTS_DIR, "version_c_cross_comparison.csv"), index=False)

# ---- 8. Cross-Version Metrics Bar Chart ----
if len(cross_df) > 1:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cross_df))
    w = 0.22
    ax.bar(x - w, cross_df["precision_at_k"], w, label="Precision@10", color="#4C72B0")
    ax.bar(x, cross_df["recall_at_k"], w, label="Recall@10", color="#55A868")
    ax.bar(x + w, cross_df["ndcg_at_k"], w, label="NDCG@10", color="#C44E52")
    labels = [f"{r['version']}\n{r['model_id']}" for _, r in cross_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Cross-Version: Best Model Comparison (A vs B vs C)", fontsize=14)
    ax.set_ylabel("Metric Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cross_version_metrics_comparison.png"), dpi=150)
    plt.close()
    print("Saved: cross_version_metrics_comparison.png")

# ---- 9. Cross-Version Coverage Comparison ----
if len(cross_df) > 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ver_colors = ["#55A868", "#4C72B0", "#C44E52"]
    ax.bar(cross_df["version"], cross_df["catalog_coverage_at_k"], color=ver_colors[:len(cross_df)])
    ax.set_title("Cross-Version: Catalog Coverage Comparison", fontsize=14)
    ax.set_ylabel("Coverage@10")
    for i, v in enumerate(cross_df["catalog_coverage_at_k"]):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cross_version_coverage_comparison.png"), dpi=150)
    plt.close()
    print("Saved: cross_version_coverage_comparison.png")

# ---- 10. Cross-Version NDCG vs Coverage Scatter ----
if len(cross_df) > 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(cross_df["catalog_coverage_at_k"], cross_df["ndcg_at_k"],
               s=200, c=ver_colors[:len(cross_df)], zorder=5, edgecolors="black", linewidth=1)
    for _, r in cross_df.iterrows():
        ax.annotate(f"{r['version']}\n{r['model_id']}",
                    (r["catalog_coverage_at_k"], r["ndcg_at_k"]),
                    xytext=(10, 10), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_title("Cross-Version: NDCG vs Coverage Trade-off", fontsize=14)
    ax.set_xlabel("Catalog Coverage@10")
    ax.set_ylabel("NDCG@10")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cross_version_ndcg_vs_coverage.png"), dpi=150)
    plt.close()
    print("Saved: cross_version_ndcg_vs_coverage.png")

# ---- 11. All Models Across Versions Bar Chart ----
all_rows = []
if os.path.isfile(a_path):
    a_df = pd.read_csv(a_path)
    a10 = a_df[a_df["k"] == 10] if "k" in a_df.columns else a_df
    for _, r in a10.iterrows():
        all_rows.append({"version": "A", "model_id": r["model_id"], "ndcg_at_k": r["ndcg_at_k"],
                         "catalog_coverage_at_k": r["catalog_coverage_at_k"]})
if os.path.isfile(b_path):
    b_df = pd.read_csv(b_path)
    for _, r in b_df.iterrows():
        all_rows.append({"version": "B", "model_id": r["model_id"], "ndcg_at_k": r["ndcg_at_k"],
                         "catalog_coverage_at_k": r["catalog_coverage_at_k"]})
for _, r in c_metrics.iterrows():
    all_rows.append({"version": "C", "model_id": r["model_id"], "ndcg_at_k": r["ndcg_at_k"],
                     "catalog_coverage_at_k": r["catalog_coverage_at_k"]})
all_df = pd.DataFrame(all_rows)

if len(all_df) > 1:
    all_df["label"] = all_df["version"] + ":" + all_df["model_id"]
    all_df = all_df.sort_values("ndcg_at_k", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(all_df) * 0.4)))
    color_map = {"A": "#55A868", "B": "#4C72B0", "C": "#C44E52"}
    bar_colors = [color_map[v] for v in all_df["version"]]
    ax.barh(all_df["label"], all_df["ndcg_at_k"], color=bar_colors)
    ax.set_title("All Models: NDCG@10 Comparison", fontsize=14)
    ax.set_xlabel("NDCG@10")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "all_models_ndcg_comparison.png"), dpi=150)
    plt.close()
    print("Saved: all_models_ndcg_comparison.png")

print(f"\nAll figures saved to {FIGURES_DIR}")
print("Done!")
