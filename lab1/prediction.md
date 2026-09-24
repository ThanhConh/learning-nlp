# Part C — Dự đoán trước khi thực nghiệm

## Prediction 1 — Vocabulary

Corpus gồm khoảng 30.000 documents được thu thập từ các trang web khác nhau.

Vì vậy, em dự đoán số lượng từ duy nhất trong corpus sẽ nằm trong khoảng **50.000 đến 200.000 từ**.

---

## Prediction 2 — Sparsity

Em dự đoán ma trận TF-IDF sẽ là một **ma trận rất thưa (highly sparse)**.

Bởi vì:
- Vocabulary có thể chứa hàng chục nghìn từ khác nhau.
- Mỗi document chỉ chứa một phần rất nhỏ trong toàn bộ vocabulary.
- Phần lớn các từ không xuất hiện trong một document cụ thể.

Do đó, đa số các phần tử trong ma trận TF-IDF sẽ có giá trị bằng 0.

Em dự đoán:
- Ma trận TF-IDF nên được lưu dưới dạng sparse matrix để tiết kiệm bộ nhớ.
- Tỷ lệ phần tử bằng 0 sẽ lớn hơn **95%**.

---

## Prediction 3 — Search
Với một query bất kỳ, các document đứng đầu kết quả tìm kiếm **không nhất thiết là các document gần nghĩa nhất**.

TF-IDF chủ yếu dựa trên sự xuất hiện của các từ trong document và trọng số IDF của chúng. Do đó, TF-IDF có thể đánh giá cao một document chỉ vì document đó chứa các từ giống với query, ngay cả khi ý nghĩa thực tế không liên quan.

Ví dụ:

**Query:**

> the bank account

Giả sử từ `bank` xuất hiện trong tương đối ít document. Khi đó `bank` có thể có giá trị IDF cao, vì:

\[
IDF(bank)=\log\frac{N}{df(bank)}
\]

Do đó, một document chứa từ `bank` có thể nhận được trọng số TF-IDF đáng kể.

Tuy nhiên, xét hai document:

> the bank account

và:

> river bank

Hai văn bản đều chứa từ `bank`, nhưng từ `bank` được sử dụng trong hai ngữ cảnh khác nhau:

- `bank account`: **ngân hàng / tổ chức tài chính**
- `river bank`: **bờ sông**

TF-IDF chỉ nhìn thấy sự trùng khớp về mặt từ vựng mà không hiểu được hai nghĩa khác nhau của từ `bank`.

Vì vậy, một document chứa từ khóa giống query có thể được xếp hạng cao dù **không thực sự liên quan về mặt ngữ nghĩa**.

Em dự đoán:

- Các document đứng đầu thường có mức độ tương đồng cao về **từ khóa** với query.
- Document có từ khóa quan trọng với IDF cao có thể được đánh giá cao.
- Tuy nhiên, kết quả đứng đầu **không nhất thiết là document gần nghĩa nhất**, đặc biệt đối với các từ đa nghĩa như `bank`.
- TF-IDF có thể gặp khó khăn trong việc phân biệt các nghĩa khác nhau của cùng một từ vì phương pháp này không mô hình hóa ngữ cảnh và ngữ nghĩa.