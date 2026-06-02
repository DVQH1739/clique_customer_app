# Tóm tắt kết quả — Online Retail II (CLIQUE)

Nguồn: `data/raw/online_retail_ii.xlsx` · Khách sau lọc `frequency >= 2`: **3,876** · Train/test 80/20.

Pipeline: winsorize (1–99%) → log1p → MinMaxScaler → grid `(xi, tau)` → chọn **xi=12, tau=0.18**.

---

## 1. Chất lượng cụm (đã cải thiện so với cấu hình cũ)

| Giai đoạn | Silhouette | Ghi chú |
|-----------|------------|---------|
| Grid (nhiều cấu hình âm) | −0.07 … +0.05 | tau thấp → quá nhiều cụm, silhouette âm |
| **CLIQUE đã chọn (train)** | **+0.049** | 7 cụm, coverage 96.4% |
| **CLIQUE (test)** | **+0.038** | coverage 96.5%, noise 3.5% |
| KMeans k=4 (baseline) | +0.248 | Cụm toàn cục, không giải thích subspace |

**Kết luận:** Silhouette **dương** trên train và test (trước đây thường âm khi tau thấp / quá nhiều cụm). CLIQUE vẫn thấp hơn KMeans trên RFM — bình thường vì CLIQUE ưu tiên **mật độ theo từng subspace**, không tối ưu silhouette toàn cục.

---

## 2. So sánh thuật toán (train, intrinsic)

| Thuật toán | Số cụm | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|------------|-------:|-----------:|---------------:|------------------:|
| **CLIQUE** (xi=12, τ=0.18) | 7 | 0.049 | 4.39 | 203 |
| KMeans k=4 | 4 | 0.248 | 1.35 | 871 |
| KMeans k=5 | 5 | 0.194 | 1.47 | 786 |
| DBSCAN eps=0.3 | 4 | −0.033 | 1.07 | 12 |
| Agglomerative k=5 | 5 | 0.166 | 1.74 | 603 |

---

## 3. Cụm CLIQUE (train)

| cluster_id | Subspace chính | Size | Ý nghĩa gợi ý |
|-----------:|----------------|-----:|----------------|
| 5 | repeat_category_rate | 1993 | Trung thành danh mục thấp–trung bình |
| 4 | weekend_ratio | 1893 | Ít mua cuối tuần |
| 3 | return_rate | 1481 | Tỉ lệ trả hàng thấp |
| 6 | return_rate + weekend_ratio | 948 | Trả thấp & ít cuối tuần |
| 0, 1 | frequency | 688 / 622 | Tần suất mua khác nhau |
| 2 | avg_basket | 584 | Giỏ hàng (scaled) trung bình |

---

## 4. Sản phẩm đầu ra (đầy đủ sau `python src/pipelines/run.py retail`)

### CSV — `results/metrics/`

| File | Nội dung |
|------|----------|
| `retail_grid_search.csv` | 15 tổ hợp (xi, tau) + silhouette, coverage, quality_score |
| `baseline_comparison.csv` | CLIQUE vs KMeans / DBSCAN / Agglomerative |
| `cluster_descriptions.csv` | Mô tả 7 cụm + business_label |
| `test_predictions.csv` | Gán cụm 776 khách test |
| `clique_subspace_coverage.csv` | Coverage theo subspace |

### PNG — `results/figures/`

| File | Nội dung |
|------|----------|
| `EDA_distributions.png` | Phân phối trước/sau transform |
| `baseline_comparison.png` | So sánh silhouette / DB / CH |
| `subspace_heatmap.png` | Heatmap subspace 2D |
| `cluster_sizes.png` | Kích thước cụm |
| `grid_*.png` | Lưới dense units (top 2D subspaces) |

### Khác

- `results/retail_run.log` — log chạy pipeline
- `data/processed/` — CSV đã xử lý
- `models/` — `clique_model.pkl`, `scaler.pkl`, `profiles.pkl`

Chạy lại: `python src/pipelines/run.py retail`.
