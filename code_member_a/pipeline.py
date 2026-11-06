
def generate_shards_config(data_dir: Path, questions_path: Path, output_path: Path):
    """
    Tạo shards.json cân bằng tải theo công thức: Dung lượng file (MB) x Số lượng câu hỏi của từng business.
    """
    biz_scores = {}
    q_counts = {}

    if questions_path.exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    q_data = json.loads(line)
                    b_id = q_data.get("business_id")
                    if b_id:
                        q_counts[b_id] = q_counts.get(b_id, 0) + 1
                except Exception:
                    pass

    for b_dir in data_dir.iterdir():
        if b_dir.is_dir() and (b_dir.name.startswith("biz_") or b_dir.name.startswith("business_")):
            b_id = b_dir.name
            total_size = sum(f.stat().st_size for f in b_dir.glob('**/*') if f.is_file())
            num_questions = q_counts.get(b_id, 1)

            score = (total_size / (1024 * 1024)) * num_questions
            biz_scores[b_id] = {
                "size_mb": round(total_size / (1024 * 1024), 3),
                "questions": num_questions,
                "score": round(score, 3)
            }

    sorted_bizs = sorted(biz_scores.items(), key=lambda x: x[1]['score'], reverse=True)

    shards = {"shard_1": [], "shard_2": []}
    toggle = True
    for b_id, metrics in sorted_bizs:
        target_shard = "shard_1" if toggle else "shard_2"
        shards[target_shard].append({"business_id": b_id, **metrics})
        toggle = not toggle

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shards, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Đã khởi tạo shards.json cân bằng tải thành công tại: {output_path}")


# LỚP CLASS CHÍNH: DATAMATE DATA PIPELINE
class DataMateDataPipeline:
    def __init__(self, split_dir: str = "data/train", cache_root: str = "cache"):
        """
        Khởi tạo pipeline cho một tập split (ví dụ: data/train, data/public_test)
        """
        self.split_dir = Path(split_dir)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.rulebook_path = self.split_dir.parent / "rulebook.md"
        if not self.rulebook_path.exists():
            self.rulebook_path = self.split_dir / "rulebook.md"

        self.calendar_path = self.split_dir / "calendar.csv"
        self.questions_path = self.split_dir / "questions.jsonl"

        # Parse quy tắc chung ngay khi khởi tạo
        self.structured_rules = parse_rulebook(self.rulebook_path)

    def process_single_business(self, business_dir: Path):
        """
        Xử lý hoàn chỉnh cho 1 doanh nghiệp:
        1. Nạp CSV, Excel, SQLite vào DuckDB & xuất Parquet tiết kiệm bộ nhớ.
        2. Quét dữ liệu bẩn.
        3. Sinh profile.json tóm tắt schema ngắn gọn cho Agent.
        """
        business_id = business_dir.name
        business_cache_dir = self.cache_root / business_id
        business_cache_dir.mkdir(parents=True, exist_ok=True)

        db_path = business_cache_dir / f"{business_id}.duckdb"
        profile_path = business_cache_dir / "profile.json"
        parquet_dir = business_cache_dir / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)

        conn = duckdb.connect(str(db_path))

        # A. NẠP DỮ LIỆU ĐA ĐỊNH DẠNG SANG DUCKDB
      
        # 1. Nạp CSV
        for file_path in business_dir.glob("*.csv"):
            if file_path.name == "calendar.csv":
                continue
            table_name = clean_name(file_path.stem)
            try:
                df = pd.read_csv(file_path)
                df = smart_clean_dataframe(df)
                df.columns = [clean_name(col) for col in df.columns]
                conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df')
            except Exception as e:
                print(f"[Lỗi CSV] {business_id} - {file_path.name}: {e}")

        # 2. Nạp Excel (.xlsx)
        for file_path in business_dir.glob("*.xlsx"):
            try:
                xls = pd.ExcelFile(file_path, engine="openpyxl")
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df = smart_clean_dataframe(df)
                    df.columns = [clean_name(col) for col in df.columns]
                    table_name = clean_name(f"{file_path.stem}_{sheet_name}")
                    conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df')
            except Exception as e:
                print(f"[Lỗi Excel] {business_id} - {file_path.name}: {e}")

        # 3. Nạp SQLite (.sqlite, .db)
        for file_path in list(business_dir.glob("*.sqlite")) + list(business_dir.glob("*.db")):
            db_alias = None
            try:
                prefix = clean_name(file_path.stem)
                db_alias = f"sqlite_{prefix}"

                conn.execute("INSTALL sqlite; LOAD sqlite;")
                conn.execute(f"ATTACH '{file_path}' AS {db_alias} (TYPE SQLITE);")

                query_tables = "SELECT table_name FROM duckdb_tables() WHERE database_name = ?"
                tables_in_sqlite = conn.execute(query_tables, [db_alias]).fetchall()

                for (tbl_name,) in tables_in_sqlite:
                    if tbl_name.startswith("sqlite_"):
                        continue

                    target_table_name = clean_name(f"{prefix}_{tbl_name}")
                    df = conn.execute(f'SELECT * FROM {db_alias}."{tbl_name}"').df()
                    df = smart_clean_dataframe(df)
                    df.columns = [clean_name(col) for col in df.columns]

                    conn.execute(f'CREATE OR REPLACE TABLE "{target_table_name}" AS SELECT * FROM df;')

            except Exception as e:
                print(f"[Lỗi SQLite] {business_id} - {file_path.name}: {e}")
            finally:
                if db_alias:
                    try:
                        conn.execute(f"DETACH DATABASE {db_alias};")
                    except Exception:
                        pass

        # B. XUẤT PARQUET & PHÁT HIỆN DỮ LIỆU BẨN
        tables = conn.execute("SHOW TABLES").fetchall()
        dirty_report = {}

        for (table_name,) in tables:
            # Xuất sang định dạng Parquet tiết kiệm bộ nhớ
            parquet_file = parquet_dir / f"{table_name}.parquet"
            conn.execute(f"COPY \"{table_name}\" TO '{parquet_file}' (FORMAT PARQUET);")

            # Quét dữ liệu bẩn
            try:
                df_sample = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 1000').df()
                dirty_report[table_name] = detect_dirty_data(df_sample, business_id)
            except Exception:
                pass

        # C. ĐỌC README.MD VÀ TRÍCH XUẤT SCHEMA PROFILE
        readme_path = business_dir / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

        profile = {
            "business_id": business_id,
            "readme_notes": readme_content,
            "rules_summary": self.structured_rules.get("sections", {}),
            "dirty_data_report": dirty_report,
            "tables": {}
        }

        for (table_name,) in tables:
            cols_info = conn.execute(f'PRAGMA table_info("{table_name}");').fetchall()
            columns = {col[1]: {"type": col[2]} for col in cols_info}
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

            try:
                sample_df = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 3;').df()
                samples = [
                    {k: sanitize_value(v) for k, v in row_dict.items()}
                    for row_dict in sample_df.to_dict(orient="records")
                ]
            except Exception as e:
                samples = []

            profile["tables"][table_name] = {
                "row_count": row_count,
                "columns": columns,
                "samples": samples
            }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        conn.close()
        print(f"[X] Hoàn tất Ingest, Parquet & Profile cho business: {business_id}")
        return business_id

    def run(self):
        """
        Thực thi toàn bộ pipeline:
        - Phân loại ưu tiên business nhỏ xử lý trước.
        - Chạy song song nền cho các business còn lại.
        - Sinh shards.json cân bằng tải.
        """
        print(f"=== BẮT ĐẦU CHẠY PIPELINE CHO THƯ MỤC: {self.split_dir} ===")

        data_businesses_dir = self.split_dir / "data"
        if not data_businesses_dir.exists():
            data_businesses_dir = self.split_dir

        business_dirs = [
            d for d in data_businesses_dir.iterdir()
            if d.is_dir() and (d.name.startswith("biz_") or d.name.startswith("business_"))
        ]
        if not business_dirs:
            business_dirs = [d for d in data_businesses_dir.iterdir() if d.is_dir()]

        # Sắp xếp business theo dung lượng thư mục tăng dần (ưu tiên business nhỏ)
        business_dirs = sorted(
            business_dirs,
            key=lambda x: sum(f.stat().st_size for f in x.glob('**/*') if f.is_file())
        )

        print(f"Tìm thấy {len(business_dirs)} doanh nghiệp cần xử lý.")

        priority_count = min(5, len(business_dirs))
        priority_bizs = business_dirs[:priority_count]
        remaining_bizs = business_dirs[priority_count:]

        print(f"\n--- BƯỚC 1: Xử lý ưu tiên {priority_count} business nhỏ cho Agent B & C ---")
        for b_dir in priority_bizs:
            self.process_single_business(b_dir)

        if remaining_bizs:
            print(f"\n--- BƯỚC 2: Chạy nền song song {len(remaining_bizs)} business còn lại ---")
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(self.process_single_business, remaining_bizs)

        # 3. TẠO SHARDS.JSON CÂN BẰNG TẢI TỔNG THỂ
        shards_output = self.cache_root / "shards.json"
        generate_shards_config(data_businesses_dir, self.questions_path, shards_output)

        print(f"\n=== HOÀN TẤT TOÀN BỘ PIPELINE VÀ CHIA SHARDS THÀNH CÔNG! ===")
