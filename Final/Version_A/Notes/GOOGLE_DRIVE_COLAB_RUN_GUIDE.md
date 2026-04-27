# Version A Google Drive + Colab Run Guide

这份流程用于把 `513` 项目放到 Google Drive，然后在网页端 Colab 运行 `Version_A` notebook。

主 notebook:

- `Final/Version_A/Experiments/version_a_full_experiment.ipynb`

默认项目路径:

```python
WORKSPACE_ROOT = Path("/content/drive/MyDrive/513")
```

## 1. 上传项目到 Google Drive

目标 Drive 目录结构必须是：

```text
MyDrive/
  513/
    Final/
      Data/
        Pure_Data/
          recipe_model_table.csv
          interactions_train_filtered.csv
          interactions_test_filtered.csv
          preprocessing_summary.json
          temporal_split_summary.json
      Version_A/
        Experiments/
          version_a_full_experiment.ipynb
          model_artifacts/
```

不要上传 `Raw_Data/` 或不需要的中间文件。Version A 只需要上面列出的 `Pure_Data` 文件、`Version_A` notebook、文档和空的 `model_artifacts/` 文件夹。

如果你安装了 Google Drive Desktop，Drive 目录通常会出现在：

```text
/Users/steve/Library/CloudStorage/
```

找到类似下面的目录：

```text
GoogleDrive-你的邮箱/My Drive
```

然后可以用：

```bash
rsync -av --progress /Users/steve/Desktop/513/Final/Version_A/ "/Users/steve/Library/CloudStorage/GoogleDrive-你的邮箱/My Drive/513/Final/Version_A/"
rsync -av --progress /Users/steve/Desktop/513/Final/Data/Pure_Data/ "/Users/steve/Library/CloudStorage/GoogleDrive-你的邮箱/My Drive/513/Final/Data/Pure_Data/"
```

如果没有 Google Drive Desktop，就在浏览器里打开 Google Drive，手动上传 `Final/Version_A` 和 `Final/Data/Pure_Data` 里的必需文件。

## 2. 在 Colab 中选择 GPU

在 Colab 页面：

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

保存后重新连接 runtime。

## 3. 在网页端 Colab 打开 notebook

打开：

```text
Final/Version_A/Experiments/version_a_full_experiment.ipynb
```

可以在 Google Drive 中右键 notebook，选择用 Google Colaboratory 打开。

## 4. 运行 import cell

先运行第一个代码 cell，导入依赖。

如果缺包，可以运行：

```python
!pip install pandas numpy scipy scikit-learn matplotlib seaborn tqdm
```

## 5. Mount Google Drive

运行 notebook 里的：

```text
Mount Google Drive
```

它会执行：

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
```

授权完成后，Colab 应该能访问：

```text
/content/drive/MyDrive/513
```

## 6. GPU Verification

运行：

```text
GPU Verification
```

你应该看到 GPU 信息，例如：

```text
GPU Device Detected
gpu_name: Tesla T4 / L4 / A100
```

并看到：

```text
GPU runtime verified.
```

如果报错：

```text
No GPU detected.
```

回 Colab 切换到 GPU runtime，然后重新连接 runtime。

## 7. 确认 Google Drive 路径

运行路径 cell 后，应该看到：

```text
Workspace: /content/drive/MyDrive/513
Data dir: /content/drive/MyDrive/513/Final/Data/Pure_Data
Results dir: /content/drive/MyDrive/513/Final/Version_A/Results
Model artifact dir: /content/drive/MyDrive/513/Final/Version_A/Experiments/model_artifacts
```

如果你的 Drive 目录不是 `MyDrive/513`，只改这一行：

```python
WORKSPACE_ROOT = Path("/content/drive/MyDrive/513")
```

不要改后面的 `DATA_DIR`、`RESULTS_DIR`、`ARTIFACT_DIR`。

## 8. 第一次先跑 sampled mode

保持默认配置：

```python
fast_dev_mode = True
eval_user_sample_size = 1000
run_full_filtered_eval = False
run_secondary_clean_eval = False
run_svd_model = True
```

这一轮用于确认：

- A0 到 A5 都能运行
- tqdm 进度条正常
- 调参图能生成
- 指标 CSV 能生成
- 模型 artifacts 能保存

## 9. 检查 sampled 输出

跑完后检查：

```text
Final/Version_A/Results/
Final/Version_A/Results/Figures/
Final/Version_A/Experiments/model_artifacts/
```

重点输出：

```text
version_a_metrics.csv
version_a_per_user_metrics.csv
version_a_tuning_results.csv
version_a_example_recommendations.csv
version_a_phase_runtime.csv
version_a_config.json
```

重点模型 artifacts：

```text
a0_bayesian_score_vector.npy
tfidf_vectorizer.pkl
tfidf_matrix.npz
a4_rerank_weights.json
A5_*_truncated_svd.pkl
A5_*_item_embeddings.npy
model_artifact_manifest.json
```

## 10. 如果 sampled 太慢

先降低参数：

```python
max_features_default = 20000
svd_components_grid = (64,)
```

也可以缩小 TF-IDF tuning grid：

```python
tfidf_tuning_grid = (
    {"max_features": 20000, "min_df": 2, "ngram_range": (1, 1)},
    {"max_features": 30000, "min_df": 2, "ngram_range": (1, 1)},
)
```

A5 保持必跑：

```python
run_svd_model = True
```

## 11. 跑正式 full filtered evaluation

sampled mode 没问题后，改配置：

```python
fast_dev_mode = False
run_full_filtered_eval = True
run_secondary_clean_eval = False
run_svd_model = True
```

如果 runtime 没断开，改完配置后从配置 cell 开始 `Run cell and below`。如果 Colab runtime 重启过，就从头 `Run all`。

这一轮结果用于 Version A 的正式指标。

## 12. 写结果分析

建议创建：

```text
Final/Version_A/Results/version_a_model_notes.md
```

推荐结构：

```md
# Version A Result Summary

## Best Model

## Metric Comparison

## Tuning Findings

## Runtime Findings

## Recommendation Examples

## Limitations
```

重点回答：

- A4 是否优于 A0/A2/A3
- A5 是否优于 A3
- 哪个模型 NDCG@10 最好
- 哪个模型 coverage 最好
- 哪个模型 runtime 最合理
- metadata-first 方法为什么适合 sparse users/items

## 13. Git 提交注意

提交 notebook 和文档：

```bash
git add Final/Version_A/Experiments/version_a_full_experiment.ipynb
git add Final/Version_A/Experiments/model_artifacts/README.md
git add Final/Version_A/Experiments/model_artifacts/.gitignore
git add Final/Version_A/Notes/GOOGLE_DRIVE_COLAB_RUN_GUIDE.md
git add Final/Version_A/README.md
git add Final/Version_A/Notes
git add Final/Version_A/Experiments/README.md
git add Final/Version_A/Results/README.md
```

不要提交训练生成的大文件：

```text
*.npy
*.npz
*.pkl
version_a_example_recommendations.csv
version_a_per_user_metrics.csv
```

`model_artifacts/.gitignore` 已经默认忽略这些文件。
`version_a_example_recommendations.csv` 可能超过 100MB，直接提交到 GitHub 会失败。正式提交时保留小的 metric CSV、config、figures 和 README 即可。
