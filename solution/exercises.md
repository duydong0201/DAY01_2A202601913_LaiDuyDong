# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng gợi ý bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature ở mức thấp (0.0), mô hình đưa ra câu trả lời cố định và trực diện nhất, không thay đổi ở các lần chạy khác nhau. Khi tăng lên 0.5 và 1.0, các phản hồi trở nên phong phú hơn, cấu trúc câu đa dạng hơn và văn phong tự nhiên hơn. Từ mức 1.5 trở đi, mô hình bắt đầu bị ảo tưởng (hallucination), tạo ra những câu chữ lộn xộn, lặp từ vô nghĩa và thông tin sai lệch.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn temperature ở mức thấp (0.0 đến 0.2) cho chatbot hỗ trợ khách hàng. Lý do là hệ thống cần đảm bảo tính chính xác và nhất quán tuyệt đối về thông tin sản phẩm, chính sách giá và hướng dẫn kỹ thuật, đồng thời hạn chế tối đa các phản hồi ngẫu hứng hoặc bịa đặt gây bối rối cho khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần (xấp xỉ 17 lần) cho cả token đầu vào và đầu ra. Một trường hợp GPT-4o xứng đáng với chi phí là khi cần phân tích hợp đồng pháp lý phức tạp hoặc gỡ lỗi các hệ thống mã nguồn lớn yêu cầu khả năng suy luận logic chuyên sâu. Một trường hợp nên dùng GPT-4o-mini là khi triển khai các chatbot dịch vụ trả lời tự động câu hỏi thường gặp (FAQs) hoặc trích xuất/phân loại dữ liệu thô khối lượng lớn để tiết kiệm chi phí tối đa.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi của hai vai trò có sự khác biệt rõ rệt về từ vựng, độ dài và các ví dụ minh họa. System prompt với vai giáo viên tiểu học đưa ra câu trả lời ngắn gọn, sử dụng từ ngữ dễ hiểu và ví dụ sinh động như "đoàn tàu chở các thùng hàng". Trong khi đó, system prompt chuyên gia tài chính trả lời dài hơn, sử dụng nhiều thuật ngữ chuyên môn như "phi tập trung", "mã hóa", "sổ cái". Điều này cho thấy system prompt có ảnh hưởng sâu sắc đến hành vi của mô hình bằng cách đặt ra các giới hạn về văn phong, độ sâu kiến thức và đối tượng hướng đến.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Sau khi so sánh một đoạn văn tiếng Việt ~100 từ, số lượng token thực tế đếm bởi tiktoken thường cao hơn từ 30% đến 50% so với ước lượng thô (số từ / 0.75). Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài là do các mô hình tokenizer hiện tại (như cl100k_base hoặc o200k_base) được xây dựng dựa trên tần suất xuất hiện của các từ tiếng Anh. Đối với các ngôn ngữ có dấu thanh và ít phổ biến hơn như tiếng Việt, tokenizer thường phải chia cắt một từ đơn hoặc từ ghép thành nhiều mảnh ký tự/byte nhỏ hơn để biểu diễn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming đặc biệt quan trọng trong các ứng dụng tương tác trực tiếp với người dùng như chatbot, trợ lý ảo hoặc hệ thống dịch thuật thời gian thực, vì nó giúp giảm cảm giác chờ đợi, cải thiện đáng kể UX nhờ thời gian phản hồi đầu tiên (Time to First Token) rất thấp. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý hàng loạt ở nền (background/batch processing), lập trình API tích hợp giữa các hệ thống (system-to-system), hoặc khi cần lấy toàn bộ kết quả văn bản để xử lý/phân tích cấu trúc (như trích xuất JSON) trước khi hiển thị cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff giúp phân phối lại và giãn cách thời gian gửi yêu cầu của các client, tránh tình trạng hàng loạt client cùng dồn dập "tấn công" (thực chất là tự DDoS) máy chủ tại cùng một thời điểm khi máy chủ đang bị quá tải tạm thời. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ luôn chờ 1 giây), nó sẽ tạo ra những "làn sóng" yêu cầu lặp đi lặp lại đè nặng lên server, làm trầm trọng thêm tình trạng nghẽn mạng và khiến server không có cơ hội hồi phục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Trợ lý ảo được thiết kế với persona là "Một trợ giảng lập trình thân thiện, ngắn gọn và nói tiếng Việt". System prompt cụ thể: "Bạn là trợ giảng lập trình thân thiện, chuyên môn cao. Hãy trả lời ngắn gọn, tập trung thẳng vào câu hỏi bằng tiếng Việt." Lựa chọn "trả lời ngắn gọn" là cực kỳ quan trọng để đảm bảo chi phí API thấp và người dùng dễ đọc nhanh trên console; chỉ định "tiếng Việt" để đồng bộ ngôn ngữ phản hồi và tránh việc AI tự động trả lời bằng tiếng Anh khi nhận được các thuật ngữ kỹ thuật trong lập trình.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là thiếu khả năng ghi nhớ dài hạn (long-term memory) giữa các phiên chạy khác nhau và bộ nhớ lịch sử hội thoại (short-term history) bị giới hạn cứng ở 3 lượt chat gần nhất để kiểm soát chi phí. Để cải thiện, ta có thể tích hợp một hệ thống cơ sở dữ liệu vector (như FAISS hoặc ChromaDB) để lưu trữ lịch sử cuộc hội thoại lâu dài dưới dạng embeddings. Khi người dùng đặt câu hỏi mới, hệ thống sẽ thực hiện tìm kiếm ngữ nghĩa (Semantic Search) để tìm ra các ngữ cảnh liên quan nhất trong quá khứ và đưa vào prompt bổ trợ cho LLM (kỹ thuật RAG).

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
