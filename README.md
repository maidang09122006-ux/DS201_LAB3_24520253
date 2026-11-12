# DS201 - Lab 3: Mạng neural hồi quy cho phân loại văn bản và gán nhãn chuỗi

Bài làm nằm trong một notebook: `Lab3_24520253.ipynb` (đã chạy xong, giữ nguyên output).

| Bài | Nội dung | Vị trí trong notebook |
|---|---|---|
| 1 | LSTM 5 lớp, hidden 256, phân loại cảm xúc UIT-VSFC, kèm baseline TF-IDF + Logistic Regression | B.3, B.4 |
| 2 | GRU 5 lớp, hidden 256, so sánh với LSTM và TF-IDF | B.5 |
| 3 | Encoder BiLSTM 5 lớp, hidden 256 mỗi chiều, NER trên PhoNER | B.6 |
| 4* | So sánh LSTM 1, 2, 5 lớp | mục "Bài 4*" |

Bảng tổng hợp và phần phân tích lỗi nằm ở cuối notebook.

## Dữ liệu

Dữ liệu không đưa lên repo. Notebook tự tải về thư mục `data/` và kiểm tra SHA-256.

- UIT-VSFC (Vietnamese Student Feedback Corpus): https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk (11,426 train, 1,583 dev, 3,166 test). Chỉ dùng `sentence` và `sentiment`.
- PhoNER_COVID19, bản syllable: https://github.com/VinAIResearch/PhoNER_COVID19 (5,027 train, 2,000 dev, 3,000 test). PhoNER không cho phân phối lại nên không có dữ liệu trong repo.

## Chạy

```bash
pip install torch numpy pandas matplotlib scikit-learn seqeval jupyter
jupyter nbconvert --to notebook --execute Lab3_24520253.ipynb
```

Hoặc mở notebook và chọn Restart Kernel, Run All. Cấu hình ở ô `cfg`: embedding 300, hidden 256, 5 lớp, dropout 0.3, batch 32, Adam lr 1e-3, `clip_grad_norm_` 1.0, seed 42, tối đa 10 epoch. Chạy trên CPU mất khoảng 30 phút.

## Kết quả

Chọn epoch theo F1 dev, tập test chỉ chấm một lần ở cuối.

| Mô hình | Dev F1 tốt nhất | Test F1 | Tham số | Giây/epoch |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.6894 | 0.6483 | unigram + bigram, min_df=2 | 0.4 (thời gian fit) |
| LSTM 5 lớp | 0.8100 (epoch 6) | 0.7521 | 3,138,007 | 35.3 |
| GRU 5 lớp | 0.8049 (epoch 7) | 0.7513 | 2,468,823 | 31.1 |
| BiLSTM NER 5 lớp | 0.9074 (epoch 10) | 0.8966 | 8,305,297 | 79.4 |

F1 của ba dòng phân loại là macro-F1, chỉ so sánh với nhau. Dòng NER là entity-F1 (seqeval, strict IOB2).

Các hình và bảng số nằm trong `checkpoints/`. Checkpoint `.pt` không đưa lên repo.
