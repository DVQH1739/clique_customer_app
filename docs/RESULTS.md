# Tóm tắt kết quả — Online Retail II (CLIQUE)

Nguồn: `data/raw/online_retail_ii.xlsx` · Khách sau lọc `frequency >= 2`: **3,876** · Train/test 80/20.

Pipeline: winsorize (1–99%) → log1p → MinMaxScaler → grid `(xi, tau)` → chọn best theo objective (`quality` / `balanced` / `coverage`).

---

## 1. Chất lượng cụm (run gần nhất)

| Giai đoạn | Silhouette | Ghi chú |
|-----------|------------|---------|
| Grid (40 cấu hình) | −0.21 … +0.14 | tau thấp → cụm quá phân mảnh, silhouette âm |
| **CLIQUE đã chọn (train)** | **+0.137** | 6 cụm, coverage 77.2% |
| **CLIQUE (test)** | **+0.128** | coverage 76.8%, noise 23.2% |
| KMeans k=4 (baseline) | +0.248 | Cụm toàn cục, không giải thích subspace |

**Kết luận:** Silhouette tăng đáng kể so với cấu hình cũ, đổi lại coverage giảm. Đây là trade-off đặc trưng của CLIQUE khi tăng độ “chặt” cụm.

---

## 2. So sánh thuật toán (train, intrinsic)

| Thuật toán | Số cụm | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|------------|-------:|-----------:|---------------:|------------------:|
| **CLIQUE** (xi=16, τ=0.20) | 6 | 0.137 | 1.76 | 296 |
| KMeans k=4 | 4 | 0.248 | 1.35 | 871 |
| KMeans k=5 | 5 | 0.194 | 1.47 | 786 |
| DBSCAN eps=0.3 | 4 | −0.033 | 1.07 | 12 |
| Agglomerative k=5 | 5 | 0.166 | 1.74 | 603 |

---

## 3. Cụm CLIQUE (train, objective=balanced)

| cluster_id | Subspace chính | Size | Ý nghĩa gợi ý |
|-----------:|----------------|-----:|----------------|
| 3 | weekend_ratio | 1824 | Nhóm ít mua cuối tuần |
| 4 | return_rate | 1472 | Nhóm tỷ lệ trả hàng rất thấp |
| 0 | return_rate + weekend_ratio | 935 | Trả thấp và rất ít cuối tuần |
| 5 | frequency | 688 | Tần suất mua thấp |
| 1 | frequency + weekend_ratio | 513 | Mua ít và không thiên về cuối tuần |
| 2 | frequency + return_rate | 510 | Mua ít và hầu như không trả hàng |

---

## 4. Sản phẩm đầu ra (đầy đủ sau `python src/pipelines/run.py retail`)

### CSV — `results/metrics/`

| File | Nội dung |
|------|----------|
| `retail_grid_search.csv` | 40 tổ hợp (xi, tau) + silhouette, coverage, quality_score |
| `best_params.csv` | Bộ tham số được chọn theo objective |
| `baseline_comparison.csv` | CLIQUE vs KMeans / DBSCAN / Agglomerative |
| `cluster_descriptions.csv` | Mô tả cụm + business_label + mean features |
| `test_predictions.csv` | Gán cụm 776 khách test |
| `clique_subspace_coverage.csv` | Coverage theo subspace |

### PNG — `results/figures/`

| File | Nội dung |
|------|----------|
| `EDA_distributions.png` | Phân phối trước/sau transform |
| `baseline_comparison.png` | So sánh silhouette / DB / CH |
| `subspace_heatmap.png` | Heatmap subspace 2D |
| `cluster_sizes.png` | Kích thước cụm |
| `pareto_frontier.png` | Trade-off coverage vs silhouette |
| `grid_*.png` | Lưới dense units (top 2D subspaces) |

### Khác

- `results/RUN_SUMMARY.md` — tóm tắt run gần nhất (top cấu hình)
- `results/retail_run.log` — log chạy pipeline
- `data/processed/` — CSV đã xử lý
- `models/` — `clique_model.pkl`, `scaler.pkl`, `profiles.pkl`

Chạy lại: `python src/pipelines/run.py retail`.

---

## 5. Validation (tự động)

Cuối pipeline, `validate.validate_retail_pipeline()` kiểm tra:

- File raw `data/raw/online_retail_ii.xlsx` tồn tại
- Hợp đồng dữ liệu 8 features, scaled ∈ [0, 1]
- CLIQUE ≥ 2 cụm, silhouette train khớp `baseline_comparison.csv`
- Đủ 5 CSV metrics + PNG bắt buộc (EDA, baseline, heatmap, cluster sizes)

Chi tiết: `docs/VALIDATION.md`.
