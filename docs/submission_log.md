# Nhật ký nộp bài (public test)

Theo kế hoạch: mỗi lượt ghi **ngày giờ, commit, thay đổi, điểm**. Điểm public chỉ để tham khảo; thước đo
chính là điểm trên tập train của C. Hạn mức: 5 lượt mỗi ngày (đặt lại cuối ngày).

Cách ghi một lượt mới:
1. Trước khi nộp, lấy `<file kết quả>.manifest.json` (do `run_all.py` tự ghi) và dán vào cột "Cấu hình" các giá trị
   chính: commit, adapter, `max_input_tokens`, `rules_chars`, chính sách unanswerable.
2. Mỗi lượt chỉ đổi **một thứ** so với lượt trước (lượt 1 đến 3 trong ngày là thí nghiệm, lượt 4 là hồi quy,
   lượt 5 là dự phòng).
3. Chọn **SLM** ở Step 2 của form (mô hình mã nguồn mở tự chạy).

## Lượt đã nộp

| # | Ngày giờ | File | Model (form) | Điểm | Số câu có đáp án |
|---|---|---|---|---|---|
| 1 | 07/11/2026 11:41:28 | `final.jsonl` | SLM | **0,00000** | 52 |
| 2 | 07/11/2026 11:43:48 | `diag_A_number.jsonl` | SLM | **−0,01400** | 30 (chỉ `number`) |
| 3 | 07/11/2026 11:44:07 | `diag_B_other.jsonl` | **APIs/Others** (chọn nhầm) | **−0,01200** | 22 (13 percent, 4 list, 4 date, 1 string) |

### Cấu hình chung của ba lượt (chạy local, một lần, shard 1)

- Nhánh `test/D-with-B`, commit gốc `190c727f`; code của D chưa được commit lúc chạy (xem manifest từ các lượt sau).
- Pipeline: `run_local.ps1 -Shards 1`, engine `hf` (transformers), Qwen2.5-Coder-7B-Instruct 4-bit NF4 +
  adapter của B (`qwen25coder7b_semantic_v1_len2048/final_adapter`).
- `max_input_tokens` 3072, `fallback_input_tokens` 2048, `rules_chars` 1600, `max_attempts` 2.
- Câu model báo unanswerable được nộp là `abstain`. **Chưa có** chốt chặn đáp án đáng ngờ (0, rỗng).
- Máy: laptop Windows, RTX 2070 8 GB, driver 621.14, torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.2,
  bitsandbytes 0.50.2, duckdb 2.0.0.
- Kết quả chạy shard 1: 189 câu, 52 có đáp án (28%), 137 abstain; trong abstain: 96 model báo unanswerable,
  23 bịa tên cột/bảng, 9 SQL ra NULL, 7 lỗi cú pháp, 2 khác. Trung bình 62 giây mỗi câu (194 phút).
- `final.jsonl` = 52 đáp án + 948 abstain; `diag_A` giữ 30 câu `number`, `diag_B` giữ 22 câu còn lại, phần khác
  đều `abstain`. Đã kiểm tra: A ∪ B = final, không trùng, các dòng giống hệt.

### Đọc kết quả

- Nếu điểm cộng được theo từng câu: A + B = −0,026 nên trong 52 câu có **13 đúng và 39 sai (≈25%)**:
  `number` 8/30 đúng, các loại còn lại 5/22 đúng. Mỗi câu có đáp án trung bình đóng góp khoảng −0,5 điểm: trả lời hết
  còn tệ hơn abstain hết.
- **Bất thường chưa giải thích được:** `final.jsonl` lẽ ra phải ra −0,026 (= A + B) nhưng cổng báo 0,00000.
  Hai khả năng: lượt 1 bị lỗi chấm (hiển thị 0 mặc định), hoặc cổng không cộng điểm theo từng câu. Cách kiểm tra:
  nộp lại `final.jsonl` đúng nguyên bản một lần.
- 19/52 câu có đáp án là loại gần như chắc chắn sai: 16 câu bằng 0 (đáp án thật bằng 0 chỉ chiếm 0,16% ở train),
  2 danh sách rỗng, 1 ngày là chuỗi "None". Đã thêm chốt chặn vào `run_all.py` (`suspicious_reason`) và công cụ
  `filter_submission.py` để áp lên file cũ.
- Nghi vấn lỗi hệ thống: 4 câu `percent` quanh 97 đến 99,7 ở các câu "hơn hay kém bao nhiêu %" (có thể tính tỷ lệ
  mới/cũ × 100 thay vì mức thay đổi); danh sách trả mã sản phẩm thay vì tên.
- Lượt 3 bị ghi loại model "APIs/Others" do chọn nhầm ở form. Giải pháp thật là mô hình mã nguồn mở tự chạy (SLM).
- Hạn mức hôm đó còn lại: 2 lượt.

## Lượt đã chuẩn bị, chưa nộp

| File | Nội dung | Dự đoán |
|---|---|---|
| `outputs/local_run3/final_filtered.jsonl` | `final.jsonl` sau chốt chặn: 33 câu có đáp án, 19 câu đáng ngờ thành `abstain` | Nếu giả thuyết "các câu 0/rỗng đều sai" đúng thì khoảng −0,007 |

## Việc kế tiếp (ưu tiên)

1. C: scorer đúng luật và PA thật trên tập train chưa huấn luyện (`docs/for_C.md`).
2. A: sửa nạp dữ liệu (cột `unnamed_N`), nạp lại cache.
3. Chỉ chạy thêm shard khi độ chính xác trên các câu được trả lời vượt 50%.
4. D: khung Docker (`predict.sh`, `Dockerfile`, README) trước hạn 16/11.
