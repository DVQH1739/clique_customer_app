# Tóm tắt kết quả & ý nghĩa — Phân khúc khách hàng bằng CLIQUE

Tài liệu này **không mô tả cách thuật toán hoạt động**, mà chỉ trình bày **các chỉ
số kết quả** và **giải thích ý nghĩa** của chúng đối với bài toán phân khúc khách hàng.

---

## 1. Chỉ số kết quả

### 1.1. So sánh các thuật toán (dữ liệu 600 khách hàng, có nhãn phân khúc thật)

| Thuật toán | Số cụm | Accuracy | F1-macro | ARI | NMI | Silhouette |
|------------|------:|---------:|---------:|----:|----:|-----------:|
| **CLIQUE** (xi=8, tau=0.08) | 31 | 0.992 | 0.992 | 0.345 | 0.565 | -0.103 |
| KMeans (k=4) | 4 | 1.000 | 1.000 | 0.870 | 0.905 | 0.387 |
| KMeans (k=5) | 5 | 1.000 | 1.000 | 0.832 | 0.863 | 0.390 |
| KMeans (k=6) | 6 | 1.000 | 1.000 | 0.685 | 0.790 | 0.277 |
| DBSCAN (eps=0.3) | 2 | 0.667 | 0.556 | 0.540 | 0.701 | 0.600 |
| DBSCAN (eps=0.5) | 2 | 0.667 | 0.556 | 0.571 | 0.734 | 0.600 |
| Agglomerative (k=5) | 5 | 1.000 | 1.000 | 0.844 | 0.870 | 0.366 |

### 1.2. Kết quả CLIQUE theo từng phân khúc

| Phân khúc | Precision | Recall | F1 | Số khách |
|-----------|----------:|-------:|---:|---------:|
| high_value | 1.000 | 0.995 | 0.997 | 200 |
| at_risk | 0.980 | 1.000 | 0.990 | 200 |
| loyal_mid | 0.995 | 0.980 | 0.987 | 200 |
| **Tổng (accuracy)** | | | **0.992** | 600 |

---

## 2. Ý nghĩa của từng chỉ số trong bài toán

- **Accuracy = 0.992**: trong 600 khách hàng, mô hình **gán đúng phân khúc cho
  ~99%**. Nói cách khác, gần như mọi khách hàng đều được xếp vào đúng nhóm
  (giá trị cao / có nguy cơ rời bỏ / trung thành trung bình).

- **Precision (độ chính xác) theo nhóm**: trong những khách hàng *được dự đoán*
  thuộc một nhóm, bao nhiêu phần trăm thực sự đúng.
  - `high_value` precision = 1.000 → **mọi khách được gắn nhãn "giá trị cao" đều
    đúng**, không "báo nhầm". Rất quan trọng vì nếu sai sẽ tốn ưu đãi VIP cho nhầm người.

- **Recall (độ bao phủ) theo nhóm**: trong những khách hàng *thực sự* thuộc một
  nhóm, mô hình tìm ra được bao nhiêu phần trăm.
  - `at_risk` recall = 1.000 → **bắt được 100% khách có nguy cơ rời bỏ**, không bỏ
    sót ai → không lỡ cơ hội giữ chân khách.

- **F1-macro = 0.992**: trung bình hài hòa giữa precision và recall, cân bằng đều
  cho cả 3 nhóm → mô hình **không thiên lệch** về bất kỳ nhóm nào, kể cả nhóm nhỏ.

- **ARI = 0.345 (Adjusted Rand Index)**: đo mức độ *cách chia nhóm* của mô hình
  giống với cách chia thật. CLIQUE thấp (0.34) cho thấy nó **chia khách hàng thành
  quá nhiều mảnh nhỏ** (31 nhãn thay vì 3), dù vẫn xếp đúng nhóm lớn. KMeans cao
  (0.87) vì chia gọn đúng 4 nhóm.

- **NMI = 0.565 (Normalized Mutual Information)**: lượng thông tin mà cách chia của
  mô hình "nói lên" về phân khúc thật. Trung bình-khá → kết quả của CLIQUE vẫn phản
  ánh đúng cấu trúc khách hàng, nhưng bị "loãng" do phân mảnh.

- **Silhouette = -0.103 (âm)**: đo độ "gọn và tách bạch" của các cụm. Giá trị âm
  nghĩa là **ranh giới giữa các cụm con của CLIQUE bị chồng lấn** — hệ quả trực tiếp
  của việc một khách hàng có thể rơi vào nhiều ô dày đặc ở các chiều khác nhau.
  KMeans dương (0.39) → cụm gọn hơn.

> Tóm lại: CLIQUE **đúng về việc khách thuộc nhóm nào** (Accuracy/F1 ≈ 0.99) nhưng
> **không gọn về ranh giới cụm** (ARI/Silhouette thấp). Phù hợp khi cần *giải thích*
> chân dung khách hàng theo từng đặc trưng hơn là khi cần các cụm tách bạch tuyệt đối.

---

## 3. Chân dung khách hàng của từng phân khúc

Mô hình tách được 3 nhóm khách hàng rõ ràng, mỗi nhóm mang một hành vi mua sắm
đặc trưng:

### Nhóm "high_value" — Khách hàng giá trị cao (VIP)
- **Mua gần đây, mua thường xuyên, chi tiêu rất nhiều** (recency thấp, frequency &
  monetary cao), giỏ hàng lớn, mua đa dạng sản phẩm, **tỉ lệ trả hàng thấp**, khá
  trung thành với nhóm sản phẩm yêu thích.
- Đây là nhóm **sinh ra phần lớn doanh thu**. Hành động: chăm sóc giữ chân, ưu đãi
  VIP, ra mắt sản phẩm sớm.

### Nhóm "at_risk" — Khách hàng có nguy cơ rời bỏ
- **Lâu rồi không mua** (recency cao), mua ít, chi tiêu thấp, ít loại sản phẩm,
  **tỉ lệ trả hàng cao**, hay mua vào cuối tuần, ít gắn bó với một nhóm sản phẩm.
- Đây là nhóm **dễ mất nhất**. Hành động: chiến dịch win-back, ưu đãi cá nhân hóa,
  cải thiện trải nghiệm để giảm trả hàng.

### Nhóm "loyal_mid" — Khách hàng trung thành mức trung bình
- Các chỉ số **ở mức trung bình** (recency/frequency/monetary vừa phải), mua tương
  đối đều, tỉ lệ trả hàng thấp–vừa.
- Đây là nhóm **tiềm năng nâng cấp lên VIP**. Hành động: cross-sell/up-sell, gợi ý
  sản phẩm bổ sung theo nhóm hàng ưa thích.

---

## 4. Kết luận theo góc độ kinh doanh

- Mô hình **phân loại khách hàng chính xác ~99%** vào 3 phân khúc có ý nghĩa hành
  động rõ ràng (giữ chân VIP / win-back nhóm rủi ro / nâng cấp nhóm trung bình).
- Các chỉ số cho thấy điểm mạnh của CLIQUE là **chỉ ra đặc trưng nào tạo nên mỗi
  nhóm** (ví dụ "recency thấp + monetary cao" = VIP), giúp marketing biết *tại sao*
  một khách thuộc nhóm đó, chứ không chỉ biết nhãn.

### Sản phẩm đầu ra
- `results/metrics/`: 4 file CSV chỉ số.
- `results/figures/`: ma trận nhầm lẫn, biểu đồ so sánh, heatmap không gian con, kích thước cụm, biểu đồ lưới.
- Ứng dụng tra cứu khách hàng: `streamlit run src/app.py`.
