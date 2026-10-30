# Code load file

```
from datamate_pipeline.loader import load_business_context, open_db
from datamate_pipeline.time_utils import resolve_time_range

# 1. Load context của 1 business cụ thể
ctx = load_business_context(split="train", business_id="biz_107", cache_root="cache")
print("Schema Card:\n", ctx["schema_card"])

# 2. Mở kết nối DuckDB để test truy vấn
conn = open_db(ctx["db_path"])
df_sample = conn.execute("SHOW TABLES").df()
print("Các bảng trong DB:\n", df_sample)
conn.close()
```
