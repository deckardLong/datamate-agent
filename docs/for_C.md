# Hướng dẫn cho C (Evaluation & Calibration)

Mục tiêu: biết **độ chính xác thật** của pipeline trên bộ train (có đáp án) và tìm quy tắc
"chỉ trả lời khi tin cậy", vì mỗi câu sai bị −1 còn abstain là 0.

## Vì sao việc này gấp

Trên public test, 52 câu model đã trả lời chỉ đúng khoảng 13 (≈25%). Một câu chỉ có lợi khi xác
suất đúng > 50%, nên trả lời hết đang làm điểm **âm**. Public test không có đáp án, nên chỉ bộ
train mới đo được. Hạn nộp Docker: 16/11.

## 1. Lấy code và dữ liệu

- Code: nhánh GitHub mà D đã push (có đủ code_member_a, b, d). `pip install -r code_member_b/requirements-colab.txt`
  rồi `pip install -U "transformers>=4.56,<5" duckdb==2.0.0 unidecode openpyxl`.
- Từ Drive của B (`datamate-agent-current`): thư mục `qlora/qwen25coder7b_semantic_v1_len2048/final_adapter`
  (chỉ cần `adapter_model.safetensors` và `adapter_config.json`) và
  `pseudo_labels/semantic3000_v2/qlora_train.jsonl`.
- Cache train: `corpus/train/` (đã có `db.duckdb`, `profile.json`, `rules.json`, `questions.jsonl`, `answers.jsonl`).
- Cần GPU (T4 trở lên). Không dùng dữ liệu public/private test để huấn luyện hay lấy đáp án (Luật 6).

## 2. Tạo tập kiểm tra không lọt vào huấn luyện

```
python -m code_member_d.make_heldout --questions corpus/train/questions.jsonl \
  --answers corpus/train/answers.jsonl --qlora-train <đường dẫn>/qlora_train.jsonl \
  --n 150 --out-dir outputs/heldout
```
Công cụ lặp lại đúng cách B chia dữ liệu (seed 42), chỉ giữ câu **chưa từng được huấn luyện**, đủ
mọi loại đáp án (kể cả ≥15 câu unanswerable) và rải đều các business.

## 3. Chạy pipeline trên tập đó

```
python -m code_member_d.run_all --questions outputs/heldout/questions_heldout.jsonl \
  --cache-root corpus --split train --out outputs/heldout/sub.jsonl \
  --adapter <đường dẫn>/final_adapter --unanswerable keep --max-input-tokens 4096
```
- Dùng `--unanswerable keep` để **đo** độ chính xác của lời báo unanswerable (mặc định `abstain` che mất nó).
- Kết quả: `sub.jsonl` (đáp án) và `outputs/heldout/logs/sub.jsonl` (mỗi câu: SQL, lỗi, số lần thử, thời gian).
- Chạy được lại nhiều lần: câu đã xong được bỏ qua.

## 4. Việc C cần làm

1. **Scorer đúng luật cuộc thi** (xem `competition_info.md`): chuẩn hóa chuỗi (NFC, chữ thường, ký tự
   không phải chữ/số → khoảng trắng, giữ dấu), `alias`, làm tròn 2 chữ số cho number/percent,
   list khớp theo vị trí, table: cột theo tên, hàng như tập; đúng +1, sai −1, abstain 0; trả lời cho
   câu unanswerable tính sai. Hàm so khớp của B (`answers_match` trong `pseudo_label.py`) **khác** luật thật
   (số lệch ≤ 0,01 vẫn khớp, chuỗi không bỏ dấu câu, bảng so theo thứ tự hàng): nên thay bằng scorer này.
2. **PA thật** trên tập kiểm tra, và bảng theo loại đáp án và theo business.
3. **Độ chính xác của lời báo unanswerable** (model báo ~40–50% câu public là unanswerable, trong khi train chỉ có 2,3%).
4. **Tìm quy tắc "chỉ trả lời khi..."** dựa trên tín hiệu có trong log: loại đáp án, có phải thử lại không
   (`attempts`), business có cột `unnamed_N` không, đáp án ≈ 100 ở câu "hơn/kém bao nhiêu %"... Mục tiêu: tập
   câu được trả lời có độ chính xác > 50%.
5. **Dữ liệu tổng hợp cho 25 luật chỉ có ở rulebook public** (khách sộp nhất, ngày đắt khách nhất, báo cáo tiền
   mặt theo ca, trung bình mỗi ngày thứ X...): train không có đáp án cho các dạng này.
6. Gửi bảng lỗi cho A và B, và quy tắc đề xuất cho D để đưa vào `run_all.py`.

## 5. Các phát hiện đã có (cần kiểm chứng bằng số đo)

- `core.py`: đã bỏ lọc tên business `biz_`/`dev_`. Hard-code `diz_114` nên bỏ trước khi đóng Docker (Luật 4).
- Cột `unnamed_N`: 8/14 business public (61% số câu) và 10/30 business train bị mất tên cột vì file Excel có
  khối tiêu đề phía trên; cần sửa `io_loaders.py` của A.
- Đáp án = 0, danh sách rỗng, ngày "None" gần như luôn sai (đáp án thật = 0 chỉ 0,16% ở train): `run_all.py` đã
  chặn (thử lại rồi abstain); tắt bằng `--allow-suspicious`.
- Câu `percent` kiểu "hơn hay kém bao nhiêu %" ra ~97–99: nghi tính tỷ lệ (mới/cũ×100) thay vì mức thay đổi.
- Rulebook public khác rulebook train: 25 luật chỉ có ở public, 3 luật đổi cách diễn đạt.
- 289 mẫu huấn luyện của B đến từ lượt template (không dùng Qwen); 21% SQL dùng cột `unnamed_N`; boolean (10 mẫu)
  và số nhỏ (23/79) có thể khớp tình cờ.

## 6. Lệnh hữu ích

- Kiểm tra môi trường: `python -m code_member_d.check_env --adapter <final_adapter>`
- Gộp shard và kiểm tra file nộp: `python -m code_member_d.make_baseline_submission --questions <q> --merge-from ... --out ...`
  rồi `--validate`.
- Áp chốt chặn đáp án đáng ngờ lên file đã chạy: `python -m code_member_d.filter_submission --sub ... --out ...`
