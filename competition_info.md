# DataMate – Zalo AI Challenge 2026

## Mô tả (Description)

Bảng tính và các cơ sở dữ liệu đơn giản là những hệ thống dữ liệu được sử dụng rộng rãi nhất trên thế giới. Một chủ cửa hàng lưu doanh số và tồn kho trong Excel; một giáo viên lưu điểm số và chuyên cần; một phòng khám lưu các lượt khám bệnh và các lô thuốc; một cán bộ nhân sự lưu bảng chấm công; một hợp tác xã lưu năng suất và chi phí đầu vào; một văn phòng xã lưu thống kê hộ gia đình; một nhà báo tải dữ liệu mở từ Tổng cục Thống kê; một nhân viên của công ty lớn muốn lấy một con số từ kho dữ liệu mà không phải chờ đội dữ liệu. Tất cả họ đều có dữ liệu. Gần như không ai trong số họ có thể viết truy vấn. Điều họ muốn là hỏi bằng tiếng Việt thông thường — "Tháng này bán chạy nhất mặt hàng gì?", "Tuần rồi bán được bao nhiêu ly trà sữa?", "Còn ai nợ tiền chưa trả?", "Thuốc nào sắp hết hạn trong kho?" — và nhận được câu trả lời đúng.

DataMate thách thức bạn xây dựng một AI agent trả lời các câu hỏi tiếng Việt tự nhiên trực tiếp trên dữ liệu có cấu trúc (CSV, Excel, SQLite): cho trước các tệp dữ liệu thô và một câu hỏi, agent trả về một câu trả lời đúng — hoặc từ chối trả lời khi dữ liệu không đủ để hỗ trợ câu hỏi. Dữ liệu cuộc thi được neo trong thế giới của các doanh nghiệp nhỏ Việt Nam — nhóm người dùng bảng tính lớn nhất cả nước và là nhóm có dữ liệu thực tế phong phú, lộn xộn nhất — nhưng năng lực mà bạn xây dựng mang tính tổng quát: hỏi đáp ngôn ngữ tự nhiên trên dữ liệu dạng bảng.

### Đầu vào (Input)

- Một gói dữ liệu của một doanh nghiệp: một hoặc nhiều tệp (.csv, .xlsx, .sqlite) cùng một tệp README.md ngắn mô tả doanh nghiệp đó.
- Một câu hỏi bằng tiếng Việt thường ngày.
- Một ngày tham chiếu (as-of date): "hôm nay" dùng để quy đổi các biểu thức thời gian tương đối ("tháng này", "tuần rồi", "cùng kỳ năm ngoái").
- Một rulebook và một lịch âm–dương (solar–lunar calendar), dùng chung cho mọi doanh nghiệp của cùng một split (xem phần Dữ liệu).

### Đầu ra (Output)

- Một câu trả lời cuối cùng thuộc một trong các kiểu ở phần Đánh giá: một giá trị (number, percent, string, date, boolean, list, table), unanswerable khi dữ liệu không đủ hỗ trợ câu hỏi, hoặc abstain khi agent chọn không trả lời.

### Ví dụ

Đầu vào — một dòng của `questions.jsonl`:

```json
{"id": "q_06de33fa21cc", "business_id": "biz_118", "question": "sp bán dc nhiều nhất tuần rồi là gì", "as_of_date": "2026-03-04"}
```

Đầu ra — một dòng của `submission.jsonl`:

```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg"}
```

---

## Đánh giá (Evaluation)

### Chấm điểm từng câu hỏi

Mỗi câu trả lời được chấm là đúng (correct), sai (wrong) hoặc bỏ qua (abstain). Câu trả lời kiểu abstain (hoặc thiếu câu trả lời) được tính là abstain. Mọi câu trả lời khác chỉ đúng nếu kiểu của nó bằng kiểu tham chiếu và giá trị của nó khớp với tham chiếu theo quy tắc dưới đây; ngược lại là sai. Không có điểm từng phần.

| Kiểu (Type) | Nội dung câu trả lời | Ví dụ | Quy tắc khớp |
|---|---|---|---|
| `number` | Một giá trị số (số tiền VND, số đếm, số lượng) | `48250000` | Khớp chính xác. Giá trị tham chiếu là số nguyên (số tiền VND, số đếm, số lượng) hoặc được làm tròn đến 2 chữ số thập phân. |
| `percent` | Một giá trị phần trăm, tính theo phần trăm | `-12.45` | Khớp chính xác sau khi làm tròn đến 2 chữ số thập phân; dấu phải đúng. |
| `string` | Một tên (sản phẩm, khách hàng, chi nhánh, thứ trong tuần, ...) | `"Sting dâu 330ml"` | Cả câu trả lời và từng dạng được chấp nhận của tham chiếu đều được chuẩn hóa, rồi so sánh bằng nhau tuyệt đối. Chuẩn hóa: (1) Unicode NFC; (2) chuyển về chữ thường; (3) mọi ký tự không phải chữ cái, chữ số hoặc khoảng trắng được thay bằng một khoảng trắng; (4) bỏ khoảng trắng đầu/cuối và gộp các chuỗi khoảng trắng liên tiếp thành một khoảng trắng. Dấu tiếng Việt (diacritics) không bị loại bỏ. Mỗi tham chiếu liệt kê các dạng được chấp nhận của nó: mọi cách viết của tên xuất hiện trong dữ liệu của doanh nghiệp đó, cộng với mã của nó — sau khi chuẩn hóa, ví dụ `sting dâu 330ml`, `sp0042`. |
| `date` | Một ngày theo định dạng ISO | `"2026-08-15"`, `"2025-09"` | Khớp chính xác. yyyy-mm-dd; các câu hỏi về một tháng: yyyy-mm (ví dụ `"2025-09"`). |
| `boolean` | `true` / `false` | `true` | Khớp chính xác. |
| `list` | Một danh sách có thứ tự gồm các tên hoặc số, theo thứ tự mà câu hỏi yêu cầu (hoặc mặc định của rulebook) | `["Cà phê sữa", "Bạc xỉu", "Trà đào"]` | Luôn có thứ tự, và được khớp theo vị trí. Thứ tự là thứ tự mà câu hỏi yêu cầu, hoặc mặc định của rulebook khi câu hỏi không nêu (xem phần Câu hỏi xếp hạng bên dưới). Các phần tử được khớp theo quy tắc của `string` hoặc `number`. |
| `table` | Một bảng nhỏ: tên cột và các hàng | `{"columns": [...], "rows": [[...], ...]}` | Các cột được khớp theo tên đã chuẩn hóa; nếu tên không khớp thì khớp theo vị trí. Các hàng được khớp như một tập hợp; các ô tên theo quy tắc `string`, các ô số theo quy tắc `number`. |
| `unanswerable` | Dữ liệu không chứa điều cần thiết để trả lời | — | Đúng khi và chỉ khi tham chiếu là unanswerable. |
| `abstain` | Agent chọn không trả lời | — | Được chấm là abstain bất kể tham chiếu là gì. |

**Câu hỏi xếp hạng.** Các bảng xếp hạng dùng tiêu chí nêu trong câu hỏi, hoặc mặc định của rulebook khi câu hỏi không nêu tiêu chí nào, theo thứ tự giảm dần của tiêu chí đó, chỉ xét trên các đơn hàng hợp lệ (valid orders); các phần tử có giá trị bằng nhau được sắp theo mã, tăng dần. Những câu hỏi mà bảng xếp hạng tham chiếu có đồng hạng tại ranh giới k (hoặc tại hạng 1 đối với các câu hỏi "… nhất") bị loại khỏi các bộ test. Khi có ít hơn k ứng viên, câu trả lời tham chiếu là tất cả các ứng viên đó.

### Điểm số

Chỉ số của bảng xếp hạng là Penalized Accuracy (PA) trên N câu hỏi của một bộ test:

$$
\mathrm{PA} = \frac{1}{N}\sum_{i=1}^{N} s_i, \qquad
s_i =
\begin{cases}
\;\;\,1 & \text{câu trả lời } i \text{ đúng (correct)} \\
\;\;\,0 & \text{câu trả lời } i \text{ là abstain} \\
-1 & \text{câu trả lời } i \text{ sai (wrong)}
\end{cases}
$$

Mọi câu trả lời được đưa ra cho một câu hỏi unanswerable đều được tính là sai. Mọi câu hỏi có trọng số như nhau. Khi đồng hạng ở bảng xếp hạng cuối cùng, thứ hạng được phân định theo tổng thời gian suy luận (inference) trong lần chạy lại của ban tổ chức, sau đó theo thời điểm nộp bài sớm hơn.

---

## Dữ liệu (Data)

### Các loại doanh nghiệp

Dữ liệu bao gồm nhiều loại doanh nghiệp nhỏ của Việt Nam: tạp hóa, quán cà phê và trà sữa, nhà hàng, nhà thuốc, cửa hàng vật liệu xây dựng, tiệm sửa xe máy, cửa hàng mẹ và bé, và nhiều loại khác. Bộ private test bao gồm các loại doanh nghiệp không xuất hiện trong dữ liệu training hoặc public test.

Mỗi doanh nghiệp có 12–30 tháng hoạt động, có tính mùa vụ (Tết, Trung thu, các ngày lễ, ngày phát lương, cuối tuần) và có quy mô từ vài trăm đến khoảng 300.000 dòng.

### Cấu trúc gói dữ liệu

```
<split>/
├── data/<business_id>/
│   ├── *.csv | *.xlsx | *.sqlite        # một hoặc nhiều tệp dữ liệu
│   └── README.md                     # mô tả doanh nghiệp và ghi chú
├── rulebook.md                          # các quy tắc và từ vựng mặc định
├── calendar.csv                          # lịch âm–dương (Tết)
├── questions.jsonl                       # mỗi dòng một câu hỏi
└── answers.jsonl                          # chỉ có ở tập training
```

Bản ghi mẫu trong `questions.jsonl`:

```json
{"id": "q_01234", "business_id": "biz_017", "question": "thang nay hang nao ban cham nhat, ton nhieu ma ban ko dc", "as_of_date": "2026-08-27"}
```

Các doanh nghiệp training còn đi kèm `answers.jsonl`, chứa câu trả lời tham chiếu cho mọi câu hỏi training, liên kết qua `"id"`. Mỗi dòng theo định dạng của bài nộp, cộng thêm trường `"alias"` liệt kê các dạng khác được chấp nhận của một câu trả lời kiểu tên (cách viết khác tìm thấy trong dữ liệu và mã mặt hàng). Các câu hỏi unanswerable có `"answer": null`. Bài nộp không cần bao gồm trường `"alias"`.

Bản ghi mẫu trong `answers.jsonl` (câu trả lời tham chiếu của câu hỏi ví dụ ở phần Mô tả):

```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
```

Chất lượng của README.md khác nhau: một số doanh nghiệp đi kèm từ điển dữ liệu đầy đủ, số khác chỉ có vài ghi chú không chính thức, một số gần như không có gì.

### Các split

| Split | Số doanh nghiệp | Số câu hỏi | Câu trả lời tham chiếu |
|---|---|---|---|
| Training | 30 | ~3.000 | Có |
| Public test | 10–20 | ~1.000 | Ẩn; được chấm trên public leaderboard |
| Private test | 10–20 | ~500 | Ẩn; được công bố ở giai đoạn private test; bao gồm các loại doanh nghiệp mới |

Dữ liệu Train: Download
Dữ liệu Public Test: Download

### Quy tắc nghiệp vụ (diễn giải mặc định)

Mỗi split đi kèm rulebook riêng (`rulebook.md`), quy định nghĩa mặc định của các thuật ngữ thông dụng khi câu hỏi không nói khác. Rulebook có thể khác nhau giữa các split nên agent của bạn phải đọc và áp dụng rulebook của split mà nó đang trả lời, không phải bản sao của rulebook training. Các quy tắc điển hình: doanh thu = tổng thành tiền của các dòng sau khi trừ chiết khấu cấp dòng (voucher cấp đơn hàng không bị trừ), không gồm VAT và phí giao hàng, không gồm các đơn đã hủy và các đơn có bất kỳ mặt hàng nào bị trả lại; "bán chạy" = theo số lượng; một tuần chạy từ thứ Hai đến Chủ nhật; "tháng này" = tháng chứa ngày as-of, tính đến và bao gồm ngày đó; giá vốn mặc định = giá vốn nhập hàng bình quân gia quyền theo lô; khoảng thời gian Tết = từ 23 tháng Chạp đến mùng 10 (âm lịch).

### Định dạng bài nộp

Tệp bài nộp. Ở cả hai vòng, bạn tải lên một tệp `submission.jsonl`, mỗi dòng một đối tượng JSON:

```json
{"id": "q_01234", "type": "string",  "answer": "Sting dâu 330ml"}
{"id": "q_01235", "type": "number",  "answer": 48250000}
{"id": "q_01236", "type": "percent", "answer": -12.45}
{"id": "q_01237", "type": "date",    "answer": "2026-08-15"}
{"id": "q_01238", "type": "boolean", "answer": true}
{"id": "q_01239", "type": "list",    "answer": ["Cà phê sữa", "Bạc xỉu", "Trà đào"]}
{"id": "q_01240", "type": "table",   "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}}
{"id": "q_01241", "type": "unanswerable", "reason": "Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
{"id": "q_01242", "type": "abstain"}
```

Các câu hỏi thiếu trong tệp được chấm là abstain. Trường `reason` là tùy chọn. Số lượt nộp mỗi ngày tuân theo quy chế chung.

Docker image. Image (xem Các vòng thi trong phần Luật) phải chứa mã nguồn (training và inference), trọng số mô hình và một `predict.sh` đọc bộ test từ `/data` và ghi `/result/submission.jsonl`, cùng với một README. Image cũng nên ghi nhật ký theo từng câu hỏi vào `/result/logs/`, ở định dạng bất kỳ, chứa tối thiểu id câu hỏi, câu trả lời cuối cùng và thời gian đã trôi qua — cộng với bất kỳ thông tin nào khác cho thấy câu trả lời được đưa ra như thế nào (ví dụ mã được sinh ra, kết quả thực thi). Ban tổ chức dùng các nhật ký này để giải quyết chênh lệch giữa kết quả bạn tải lên và lần chạy lại của họ (Luật 8).

---

## Luật (Rule)

### Các vòng thi

Cuộc thi diễn ra trên một bảng xếp hạng (leaderboard) với hai vòng:

- Vòng 1 — Public test: các đội phát triển giải pháp, chạy trên bộ public test và tải lên `submission.jsonl`. Điểm số xuất hiện trên public leaderboard và chỉ mang tính tham khảo trong quá trình phát triển.
- Nộp giải pháp cuối cùng: trước hạn nộp giải pháp, giải pháp cuối cùng được nộp dưới dạng Docker image, theo hướng dẫn Docker riêng của ban tổ chức. Để chấm điểm, ban tổ chức gắn (mount) bộ test vào container tại `/data` và thu thập `/result/submission.jsonl`.
- Vòng 2 — Private test: sau khi các đội nộp và đóng băng giải pháp, ban tổ chức công bố bộ private test. Các đội chạy giải pháp đã đóng băng trên bộ này và tải lên `submission.jsonl`. Xếp hạng cuối cùng dựa trên lần chạy lại các Docker image đã nộp của ban tổ chức (Luật 8).

### Yêu cầu kỹ thuật

Mọi giải pháp phải đáp ứng tất cả các yêu cầu sau:

1. Các đội và điều kiện tham gia tuân theo quy chế chung của Zalo AI Challenge 2026.
2. Mô hình. Bạn có thể dùng bất kỳ mô hình nào có trọng số được công bố công khai trước ngày 1 tháng 10 năm 2026 theo giấy phép cho phép việc sử dụng đó, và có thể tinh chỉnh (fine-tune) tự do. Trên tất cả các mô hình được giải pháp của bạn dùng khi suy luận (mô hình ngôn ngữ, mô hình embedding, reranker, bộ phân loại), dù được nạp cùng lúc hay không, tổng số tham số không được vượt quá 15 tỷ, tính tất cả tham số của các mô hình mixture-of-experts. Cho phép mọi hình thức lượng tử hóa (quantization). Các mô hình được dùng phải được khai báo trong README của giải pháp.
3. Không dùng dịch vụ bên ngoài. Quá trình suy luận phải chạy hoàn toàn offline bên trong Docker image của bạn: không gọi API LLM thương mại, không tải xuống, không truy cập mạng dưới bất kỳ hình thức nào. Image được chạy khi tắt mạng.
4. Không có con người trong vòng lặp. Các câu trả lời phải do hệ thống của bạn tạo ra mà không có can thiệp thủ công. Cấm hard-code câu trả lời, hoặc logic dựa trên định danh của câu hỏi hoặc doanh nghiệp.
5. Thực thi mã. Agent của bạn có thể sinh và thực thi mã bên trong container; điều này được kỳ vọng nhưng không bắt buộc. Thư mục dữ liệu được gắn ở chế độ chỉ đọc; hãy ghi các tệp tạm ở nơi khác.
6. Dữ liệu. Bạn có thể dùng dữ liệu training được cung cấp, dữ liệu do bạn tự tổng hợp, và các bộ dữ liệu cùng mô hình công khai. Bạn không được dùng dữ liệu public hay private test để huấn luyện, cũng không được thu thập câu trả lời của các câu hỏi test bằng bất kỳ cách nào.
7. Tham chiếu về tài nguyên tính toán. Các giải pháp được kiểm chứng trên một máy có 2 × NVIDIA RTX 5090 (32 GB VRAM mỗi card, tổng 64 GB), 512 GB RAM. Giải pháp của bạn phải xử lý toàn bộ bộ private test trong vòng 6 giờ thời gian thực (wall-clock) trên máy này, bao gồm cả thời gian nạp mô hình. Các GPU Blackwell yêu cầu CUDA ≥ 12.8 và bản build PyTorch/vLLM tương ứng; hướng dẫn Docker quy định base image cần dùng.
8. Điểm số cuối cùng và khả năng tái lập. Điểm của lần chạy lại Docker image đã nộp của ban tổ chức trên bộ private test là điểm số cuối cùng của bạn. Tệp `submission.jsonl` bạn tải lên trong giai đoạn private test chỉ quyết định bảng xếp hạng tạm thời và những đội nào được chạy lại. Giải pháp của bạn phải đủ tất định (deterministic) để hai lần chạy khớp nhau: một giải pháp được coi là có thể tái lập nếu |PA(uploaded) − PA(re-run)| ≤ 0,03. Nếu chênh lệch vượt quá 0,03, ban tổ chức sẽ xem xét các nhật ký do image của bạn ghi (xem Định dạng bài nộp, phần Dữ liệu), có thể chạy image thêm một lần nữa, và có thể yêu cầu đội cung cấp thêm thông tin; nếu không giải thích được chênh lệch, bài dự thi bị loại. Nếu lần chạy lại vượt quá giới hạn thời gian của Luật 7 hoặc kết thúc sớm, chỉ các câu trả lời đã được ghi vào `/result/submission.jsonl` trước khi bị dừng mới được chấm; phần còn lại được tính là abstain.

---

## Hướng dẫn nộp bài (Submission Guidance)

### I. Vòng Public Test (26 tháng 10 – 16 tháng 11)

- Các đội tham gia tự xây dựng giải pháp, chạy trên bộ public test của ban tổ chức, và nộp các tệp kết quả (JSONL) qua mục Submission trên cổng thông tin Zalo AI Challenge. Mỗi đội được nộp tối đa 5 lần mỗi ngày, và hạn mức được đặt lại vào cuối mỗi ngày. Điểm số và thứ hạng sẽ được hiển thị trên public leaderboard.
- Vào ngày 16 tháng 11, các đội tham gia phải nộp giải pháp hoàn chỉnh, bao gồm mã training, mã inference và dữ liệu, dưới dạng một gói Docker cho ban tổ chức qua biểu mẫu "Solution Submission" trên cổng thông tin Zalo AI Challenge. Vì tệp Docker thường khá lớn, các đội nên cung cấp checksum và một đường dẫn Google Drive công khai để ban tổ chức có thể tải xuống. Mỗi đội được nộp tối đa 2 lần. Ban tổ chức chỉ đánh giá lần nộp cuối cùng. Hạn nộp: 23:59 (GMT+7).

### II. Vòng Private Test (20 tháng 11 – 23 tháng 11)

- Lúc 9:00 sáng ngày 20 tháng 11, ban tổ chức sẽ công bố bộ private test. Các đội tham gia sẽ chạy giải pháp đã nộp của mình trên bộ này và nộp các tệp kết quả (JSONL) qua mục Submission trên cổng thông tin Zalo AI Challenge, với hạn chót là 23:59 (GMT+7) ngày 23 tháng 11. Private leaderboard sẽ được hiển thị theo thứ tự thời gian (dựa trên thời điểm nộp).

Ban tổ chức sẽ đánh giá các giải pháp đã nộp để kiểm chứng kết quả, rà soát mã nguồn và phát hiện gian lận. Top 2 đội sẽ được công bố vào ngày 2 tháng 12.