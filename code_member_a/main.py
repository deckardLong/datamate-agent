from pipeline import DataMateDataPipeline

if __name__ == "__main__":
    # Thực thi cho tập dữ liệu train
    pipeline = DataMateDataPipeline(split_dir="data/train", cache_root="cache/train")
    pipeline.run()
