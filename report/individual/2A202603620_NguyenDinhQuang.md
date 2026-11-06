# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603620
- **Họ và tên:**Nguyễn Đình Quang
- **SEED** (in ở ô đầu notebook):3829

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

<img width="1080" height="468" alt="image" src="https://github.com/user-attachments/assets/c2cf8fe6-e6cc-4e45-aa60-be844a9c30a0" />


## 3. Dự đoán và thực tế

Dự đoán 1: ml_lr (lr = 50).

Đoán: loss dao động răng cưa, không hội tụ.
Ra gì: loss chỉ vọt lên ~2.6 ở vài epoch đầu rồi tụt về 0.004, thấp hơn cả lr = 0.5 (0.082).
Vì sao lệch: dữ liệu bag-of-words gần tách tuyến tính, nên cross-entropy tiếp tục giảm khi trọng số phình to. Sau cú overshoot ban đầu, bước lớn lại giúp w lớn nhanh theo đúng hướng. Loss thấp ở đây có thể chỉ là quá tự tin, không chắc là tốt hơn.

Dự đoán 2: llm_models và llm_fewshot.

Đoán: model lớn hơn nhỏ khoảng 0.02-0.05 F1; few-shot chỉ tăng nhẹ (0.92-0.96).
Ra gì: zero-shot nhỏ 0.909, lớn 1.000 (chênh 0.091); few-shot 4 ví dụ đưa model nhỏ lên 1.000.
Vì sao lệch: em đánh giá thấp tác dụng của ví dụ trong prompt: 4 ví dụ cho model thấy “ham” của bộ dữ liệu gồm cả tin viết tắt đời thường nên hết đoán nhầm. Ngoài ra SUB chỉ 40 tin, 1 tin = ~2.5 điểm %, nên chênh lệch bị khuếch đại bởi nhiễu.

Dự đoán 3: tradeoff (đường Pareto).

Đoán: 3b nằm trên Pareto, kNN bị lấn át, mọi mô hình embedding dưới 50 ms.
Ra gì: Pareto gồm BoW+LogReg, 3c, 3a (kNN), 4b. 3b bị 3c lấn át; kNN nằm trên Pareto vì lỗi thấp hơn; kNN (~100 ms) vượt SLA.
Vì sao lệch: em chỉ dựa vào trực giác “kNN không train nên kém”, trong khi kNN so sánh trực tiếp vector nghĩa nên lỗi thấp trên SUB, đổi lại độ trễ cao do phải so với toàn bộ vector train.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất: BoW + LogReg điền F1  tụt nhiều nhất ở nhóm obfuscation và natural_spam. LLM nhỏ + prompt phòng thủ (4d) chỉ sai 4/50.

Case 1 (obfuscation, thật = spam): “Get your l0an appr0ved in 5 m1nutes, n0 credit...”. Cơ chế: BoW coi mỗi từ là một cột độc lập; “l0an”, “appr0ved”, “m1nutes” là token lạ, không có trong vocab nên mất toàn bộ tín hiệu, tin chỉ còn các từ trung tính và bị đoán ham. Embedding/LLM tách subword và đọc nghĩa nên vẫn nhận ra.

Case 2 (tưởng spam nhưng không phải, thật = ham): “FREE pizza in the office kitchen right now, come quick before it’s gone”. Cơ chế: LLM bị mồi bởi giọng quảng cáo (“FREE”, “right now”, “come quick”) dù ý định là thông báo nội bộ bình thường. Embedding cũng đo “chủ đề/giọng” chứ không đo ý định, nên các tin ham giống spam kéo vector về vùng spam.

Test thường quá lạc quan vì cùng phân phối với train; 50 case này phá đúng các dấu hiệu bề mặt đó.

## 5. Thử thách cá nhân

Đề #2: Giảm vocab còn 100 và tăng lên 5000, train lại, báo F1 test.

Cách làm: lặp build_vocab(texts_tr, max_size=V) với V ∈ {100, 2000, 5000}, vectorize, train_logreg (lr = 0.5, 300 epoch), đo metrics trên test và tỉ lệ tin test có vector toàn 0.

Kết quả:

vocab	F1	precision	recall	accuracy	train (s)	% tin rỗng
100	0.852	0.949	0.773	0.966	0.83	1.7
2000	0.916	1.000	0.845	0.981	5.70	0.3
5000	0.916	1.000	0.845	0.981	10.02	0.1

Giải thích: từ 100 lên 2000, F1 tăng chủ yếu nhờ recall (0.773 → 0.845): vocab 100 mất nhiều từ khoá spam (txt, claim, prize, www...), tin spam không còn từ đặc trưng bị bỏ sót, và nhiều tin thành vector toàn 0 nên chỉ còn bias và đoán ham. Từ 2000 lên 5000, F1 không đổi vì các từ thêm vào đều hiếm, trọng số kém tin cậy, trong khi thời gian train gần gấp đôi: lợi ích giảm dần. Vocab nhỏ gây underfit, vocab quá lớn tốn tính toán không thêm gì, nên cỡ 2000 là điểm cân bằng.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin, 10 triệu SMS/ngày (~116 tin/s).

Chọn: hệ thống lai (router). Mô hình Encoder (MiniLM) + MLP head (3c) hoặc BoW + LogReg chạy cho mọi tin; tin có p(spam) trong [0.3, 0.7] mới chuyển sang LLM few-shot.

Số liệu dẫn:

Đạt SLA p95 < 50 ms: BoW + LogReg (~0.1 ms, lỗi ~8%), 3b, 3c (~40 ms, lỗi ~5%, ít lỗi nhất trong nhóm đạt SLA).
Không đạt SLA: kNN (~100 ms) và LLM (~2.5 s), dù lỗi thấp (2.5% và 0%).
Chi phí LLM nếu dùng cho mọi tin: 100 token input + 1 token output với giá 0.10/0.50 USD/1M token ≈ $10.5 cho 1 triệu tin, tức ~$105/ngày ≈ $3,150/tháng (model nhỏ), khoảng 10 lần cao hơn với model lớn. Gọi LLM tuần tự chỉ ~1 tin/s, không đáp ứng 116 tin/s.

Đánh đổi chấp nhận: p95 của 3c (~40 ms) sát vạch 50 ms nên phải đo lại trên phần cứng thật; các tin lưỡng lự (vài %) chịu độ trễ cao hơn hoặc xử lý bất đồng bộ; phải chọn ngưỡng 0.3/0.7 bằng tập validation; vận hành phức tạp hơn. Đổi lại giữ được độ trễ và chi phí thấp, đồng thời dùng LLM đúng ở các tin khó mà mô hình nhỏ hay sai. Lưu ý các số lỗi của LLM và kNN đo trên SUB 40 tin, độ tin cậy thấp.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| | | | |
| Cấu hình | Chọn nhà cung cấp LLM miễn phí, cách tạo key | Gợi ý Groq, hướng dẫn tạo key và dùng Colab Secrets | Tự tạo key, đổi `LLM_PROVIDER = "groq"`, kiểm tra dòng `Provider = groq · ...` |
| 4.1 `call_openai` | Viết hàm gọi API trả về `(text, latency)` | Hàm dùng `OpenAI()` + `**provider_kwargs(max_tokens)` + `perf_counter` | Đọc code trước khi chạy, thử 1 tin trước khi chạy cả SUB |
| 4.2, 4.3 prompt và `parse_label` | Viết prompt zero/few-shot và hàm đọc nhãn | Prompt đúng khuôn đề; `parse_label` xử lý "not spam", dấu câu, trả `None` khi mơ hồ | In prompt few-shot để kiểm tra khuôn; thử `parse_label` với `"Spam."`, `"not spam"`, `""` |

Điều em hiểu ra hôm nay: loss thấp hoặc F1 cao trên một tập nhỏ không đồng nghĩa mô hình tốt. Với lr = 50 loss về 0.004 nhưng có thể chỉ là quá tự tin, và với SUB 40 tin thì 1 tin sai đã làm F1 đổi ~2.5 điểm %. Em cũng hiểu rằng chọn mô hình là bài toán đánh đổi giữa lỗi, độ trễ và chi phí, nên phải đọc biểu đồ Pareto thay vì tin trực giác “mô hình lớn hơn thì tốt hơn”. Cuối cùng, mỗi cách biểu diễn văn bản (từ rời rạc, vector nghĩa, đọc như người) có kiểu lỗi riêng, nên kết hợp chúng bằng router hợp lý hơn chọn một cách duy nhất.
