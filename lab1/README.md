## Giới thiệu tổng quan lab1

- **Biểu diễn văn bản truyền thống**: Tiền xử lý (Tokenization, Lemmatization, Stopwords), Thống kê tần suất (Count Vectorizer, Bag-of-Words, TF-IDF), Khảo sát tính thưa (Sparsity).
- **Truy vấn thông tin & Tìm kiếm**: Xây dựng Search Engine dựa trên không gian vector (Vector Space Model), Cosine Similarity, Đánh giá định lượng ($P@K$, $R@K$, $MRR$), Phân tích lỗi (Error Analysis).
- **Biểu diễn ngữ nghĩa phân tán**: Word Embeddings (Word2Vec, GloVe, FastText), Contextual Embeddings (ELMo, BERT).
- **Kiến trúc Hiện đại & Ứng dụng**: Transformer, Attention Mechanism, Dense Retrieval (Semantic Search), RAG (Retrieval-Augmented Generation).

---

## 📂 Cấu trúc thư mục Lab 1

Thư mục `lab1/` chứa toàn bộ mã nguồn, dữ liệu thực nghiệm và báo cáo của Bài thực hành 1:

```text
lab1/
├── experiments.ipynb       # Notebook chính: toàn bộ thực nghiệm từ Part D đến Part J
├── implementation.py       # Module chứa các hàm tính toán cốt lõi (TF, IDF, TF-IDF, Cosine Similarity)
├── prediction.md           # Báo cáo dự đoán trước khi thực nghiệm (Part C — Predictions)
├── reflection.md           # Báo cáo phản tư sau thực nghiệm (Section 16 — Reflection)
├── results.csv             # Bảng kết quả truy vấn và đo lường định lượng (P@5, R@5, MRR)
├── .gitignore              # Cấu hình bỏ qua file nháp (lab1_v2.ipynb), file cũ và bộ nhớ đệm
└── README.md               # Tài liệu hướng dẫn và báo cáo chi tiết bài thực hành Lab 1
```

---

## Chi tiết Bài thực hành: Lab 1 — TF-IDF & Search Engine

### 1. Các nội dung chính:

- **Part C — Dự đoán trước thực nghiệm (Prediction)**: Phân tích dự đoán về kích thước từ điển ($|V|$), độ thưa ma trận ($>95\%$), và giới hạn của TF-IDF đối với các từ đa nghĩa/đồng nghĩa.
- **Part D — Sparse Representation**:
  - Trích xuất đặc trưng trên tập dữ liệu thực tế ($1.000$ documents từ tập C4-30K).
  - Ma trận $TF\text{-}IDF$ có kích thước $1000 \times 7154$ với độ thưa đạt **$98.01\%$**.
  - Phân tích 20 terms có Document Frequency cao nhất và IDF cao nhất.
- **Part E — Unit Tests & Reference Comparison**:
  - Tự cài đặt các hàm toán học: `build_vocabulary`, `compute_counts`, `compute_tf`, `compute_idf`, `compute_tfidf`, `cosine_similarity`.
  - Bộ kiểm thử tự động (Unit Tests) đạt chuẩn $100\%$.
  - Đối chiếu và giải thích sự khác biệt công thức quy ước IDF với `scikit-learn` ($+1$ smooth).
- **Part F — Preprocessing Ablation**:
  - Khảo sát 3 pipeline tiền xử lý: Pipeline A (Regex cơ bản), Pipeline B (Lọc Stopwords), Pipeline C (Subword segmentation).
  - So sánh kích thước từ vựng, độ dài token trung bình và tỷ lệ Out-Of-Vocabulary (OOV).
- **Part G — Document Search Engine**:
  - Xây dựng công cụ tìm kiếm tài liệu bằng vector TF-IDF kết hợp thước đo góc Cosine Similarity.
- **Part H — Quantitative Evaluation**:
  - Thiết lập Benchmark gồm 6 queries đại diện kèm tập nhãn ground-truth relevance.
  - Cài đặt và đo lường 3 metrics chuẩn:
    - **Mean Precision@5**: **$0.6333$** ($63.33\%$)
    - **Mean Recall@5**: **$0.8750$** ($87.50\%$)
    - **Mean Reciprocal Rank (MRR)**: **$1.0000$** (100% tài liệu đúng xuất hiện ngay tại Rank 1)
  - Tự động xuất toàn bộ bảng kết quả chi tiết ra file [`results.csv`](results.csv).
- **Part I — Error Analysis**:
  - Phân tích chuyên sâu 2 queries kết quả tốt (`transformer language model`, `natural language processing`) và 2 queries kết quả kém (`climate change global warming`, `database sql query`).
  - Mổ xẻ **Failure Case kinh điển (Vocabulary Mismatch Problem)**: minh chứng query y khoa `"myocardial infarction prevention"` nhận điểm tương đồng $0.000000$ đối với tài liệu `"heart attack"` do hai vector trực giao hoàn toàn.
- **Part J — From Failure to Next NLP Representation**:
  - Trả lời câu hỏi lớn về biểu diễn ngữ nghĩa thay cho trùng khớp từ vựng rời rạc.
  - Dẫn giải **Giả thuyết phân bố (Distributional Hypothesis - J.R. Firth, 1957)**.
  - Phân tích lộ trình tiến hóa:
    $$\mathbf{TF\text{-}IDF} \longrightarrow \mathbf{Distributional\ Representation} \longrightarrow \mathbf{Word\ Embedding} \longrightarrow \mathbf{Contextual\ Embedding} \longrightarrow \mathbf{Transformer}$$

---
