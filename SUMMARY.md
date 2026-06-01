# Tóm tắt: Phân cụm không gian con CLIQUE cho phân khúc khách hàng

Tài liệu này tóm tắt **thuật toán** và **kết quả đạt được** của bài toán phân cụm
khách hàng bằng CLIQUE (CLustering In QUEst).

---

## 1. Bài toán

Phân khúc khách hàng dựa trên 8 đặc trưng hành vi (RFM mở rộng), tìm ra các nhóm
khách hàng có ý nghĩa kinh doanh mà **không cần nhãn cho trước** (học không giám sát).

8 đặc trưng (xem `DATA_CONTRACT.md`):
`recency, frequency, monetary, avg_basket, product_diversity, return_rate,
weekend_ratio, repeat_category_rate`.

---

## 2. Thuật toán CLIQUE (Agrawal et al., SIGMOD 1998)

CLIQUE là thuật toán phân cụm **theo không gian con** (subspace clustering): nó tự
động tìm các cụm nằm trong những không gian con (tập con các chiều) khác nhau, phù
hợp cho dữ liệu nhiều chiều. Triển khai gồm 3 pha (`src/clique/algorithm.py`):

### Pha 1 — Xác định không gian con (bottom-up, kiểu Apriori)
- Chia mỗi chiều thành `xi` khoảng đều nhau trên `[0, 1]` (dữ liệu đã MinMax-scale).
- Tìm các **ô đơn vị dày đặc** (dense unit) 1 chiều: ô có mật độ `count/N ≥ tau`.
- Sinh ứng viên k chiều bằng cách mở rộng các ô dày đặc (k-1) chiều.
- **Cắt tỉa theo tính đóng đi xuống (downward-closure)**: chỉ giữ ứng viên mà *mọi*
  hình chiếu (k-1) chiều của nó đều dày đặc (`_all_projections_dense`).

### Pha 2 — Xác định cụm
- Trong mỗi không gian con, hai ô dày đặc **kề nhau** nếu chúng chỉ khác đúng một
  khoảng ở đúng một chiều (`_units_are_adjacent`).
- Dùng **BFS** tìm thành phần liên thông → mỗi thành phần là một cụm
  (`_bfs_connected_components`).
- Gán nhãn điểm: một điểm có thể thuộc nhiều cụm ở các không gian con khác nhau;
  nhãn cuối lấy theo cụm ở **không gian con nhiều chiều nhất**.

### Pha 3 — Mô tả tối giản
- Sinh mô tả DNF dễ đọc cho mỗi cụm, ví dụ:
  `monetary ∈ [0.75, 1.00] AND recency ∈ [0.00, 0.12]`.
- Dùng **greedy cover** gộp các ô liền kề để giảm số hình hộp.

### Tham số
- `xi`: số khoảng chia mỗi chiều.
- `tau`: ngưỡng mật độ tối thiểu (theo tỉ lệ tổng số điểm).
- **Tính tái lập:** `RANDOM_STATE = 42` dùng thống nhất; CLIQUE bản chất tất định.

---

## 3. Quy trình (pipeline) — 4 bước

Mã nguồn nằm gọn trong 4 thư mục lớn: `data/`, `src/`, `models/`, `results/`.

| Bước | File | Mô tả |
|------|------|-------|
| 1 | `src/pipelines/generate_data.py` | Sinh dữ liệu tổng hợp 600 KH, 3 phân khúc (có nhãn thật) |
| 2 | `src/pipelines/preprocess.py` | Log-transform → tách train/test (42) → MinMax scale |
| 3 | `src/pipelines/train.py` | Grid search `(xi, tau)` → chọn theo ARI → huấn luyện |
| 4 | `src/pipelines/evaluate.py` | Tính metric (CSV) + vẽ biểu đồ (PNG) |

Chạy toàn bộ: `python src/scripts/run_all.py`.

**Tiền xử lý quan trọng:** các đặc trưng lệch (`monetary, frequency, avg_basket,
product_diversity`) được **log-transform** trước khi scale; nếu không, CLIQUE chia
ô đều sẽ dồn 95% dữ liệu vào 1–2 ô đầu → không tìm được ô dày đặc.

**Lưu ý feature engineering (đã sửa lỗi):** hóa đơn hủy (`InvoiceNo` bắt đầu bằng
`C`) mang số lượng âm, nên phải tách *trước* bộ lọc `Quantity > 0`; nếu không
`return_rate` luôn bằng 0.

---

## 4. Chọn mô hình

Grid search trên `xi ∈ {5, 8, 10}`, `tau ∈ {0.02, 0.05, 0.08}`.

Tiêu chí chọn là **Adjusted Rand Index (ARI)**, không dùng F1. Lý do: F1 theo
majority-vote sẽ "thưởng" cho việc phân mảnh quá mức (hàng trăm cụm nhỏ, mỗi cụm
khớp gọn vào một phân khúc → F1 ≈ 1.0 giả tạo). ARI phạt cả việc chia quá nhỏ lẫn
gộp quá lớn.

→ Tham số được chọn: **`xi = 8, tau = 0.08`** (54 cụm trên các không gian con, 31
nhãn thực sự được dùng).

---

## 5. Kết quả

### So sánh với baseline (trên dữ liệu tổng hợp, có nhãn thật)

| Thuật toán | n_clusters | Accuracy | F1-macro | ARI | NMI | Silhouette |
|------------|-----------:|---------:|---------:|----:|----:|-----------:|
| **CLIQUE** (xi=8, tau=0.08) | 31 | 0.992 | 0.992 | 0.345 | 0.565 | -0.103 |
| KMeans (k=4) | 4 | 1.000 | 1.000 | **0.870** | **0.905** | **0.387** |
| KMeans (k=5) | 5 | 1.000 | 1.000 | 0.832 | 0.863 | 0.390 |
| KMeans (k=6) | 6 | 1.000 | 1.000 | 0.685 | 0.790 | 0.277 |
| DBSCAN (eps=0.3) | 2 | 0.667 | 0.556 | 0.540 | 0.701 | 0.600 |
| DBSCAN (eps=0.5) | 2 | 0.667 | 0.556 | 0.571 | 0.734 | 0.600 |
| Agglomerative (k=5) | 5 | 1.000 | 1.000 | 0.844 | 0.870 | 0.366 |

### Báo cáo phân loại của CLIQUE (sau khi gán cụm → phân khúc)

| Phân khúc | Precision | Recall | F1 | Support |
|-----------|----------:|-------:|---:|--------:|
| high_value | 1.000 | 0.995 | 0.997 | 200 |
| at_risk | 0.980 | 1.000 | 0.990 | 200 |
| loyal_mid | 0.995 | 0.980 | 0.987 | 200 |
| **Accuracy** | | | **0.992** | 600 |

### Không gian con nổi bật (độ phủ cao nhất)
- 1 chiều: `weekend_ratio`, `repeat_category_rate` (phủ 1.00), `monetary` (0.93),
  `recency` (0.91).
- 2 chiều: `recency & monetary` (0.40), `recency & return_rate` (0.35) — phản ánh
  rõ nhóm khách giá trị cao (recency thấp, monetary cao).

---

## 6. Nhận xét & kết luận

- **CLIQUE tách đúng 3 phân khúc** sau khi gán nhãn (Accuracy/F1 ≈ 0.99): nó tìm
  được các vùng dày đặc đúng với cấu trúc dữ liệu, và cho **mô tả cụm dễ hiểu** theo
  từng chiều — đây là ưu điểm lớn của CLIQUE so với KMeans/DBSCAN.
- **Nhưng CLIQUE phân mảnh** không gian (ARI 0.34, silhouette âm): cùng một điểm
  xuất hiện trong cụm của nhiều không gian con, dẫn đến nhiều nhãn. Đây là **đặc tính
  vốn có** của CLIQUE trên dữ liệu dạng cầu (globular) như RFM.
- Với dữ liệu này, **KMeans (k=4)** và **Agglomerative** cho phân cụm gọn nhất
  (ARI ≈ 0.85–0.87, silhouette cao). CLIQUE phù hợp hơn khi cần **giải thích cụm**
  và khi cụm chỉ tồn tại trong **một vài chiều con** của không gian nhiều chiều.
- **ROC/AUC đã được loại bỏ** một cách có chủ đích: với phân cụm "cứng" nó không có
  định nghĩa hợp lý, và proxy theo tâm cụm cho AUC = 1.0 ở mọi thuật toán (gây hiểu
  nhầm). Bộ metric giữ lại đều chính xác và có ý nghĩa.

### Sản phẩm đầu ra
- `results/metrics/`: `model_comparison.csv`, `clique_classification_report.csv`,
  `clique_grid_search.csv`, `clique_subspace_coverage.csv`.
- `results/figures/`: `confusion_matrix_clique.png`, `metric_comparison.png`,
  `subspace_heatmap.png`, `cluster_sizes.png`, và 2 biểu đồ lưới không gian con.
- `models/`: `clique_model.pkl`, `scaler.pkl`, `profiles.pkl`.
- Ứng dụng Streamlit: `streamlit run src/app.py`.
