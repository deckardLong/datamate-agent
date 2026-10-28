import re
from datetime import datetime, date
import pandas as pd
import numpy as np
from unidecode import unidecode

def clean_name(raw_name: str) -> str:
    """
    Làm sạch tên bảng an toàn, tự động xử lý lỗi font, bỏ dấu tiếng Việt,
    và chuyển ký tự đặc biệt thành snake_case.
    """
    decoded_name = raw_name
    # try:
    #     decoded_name = raw_name.encode('latin1').decode('utf-8')
    # except (UnicodeEncodeError, UnicodeDecodeError):
    #     decoded_name = raw_name
    decoded_name = raw_name
    text = unidecode(decoded_name).lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def smart_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tự động quét tên cột và dữ liệu để làm sạch, ép kiểu thông minh
    mà không cần biết trước tên cột cụ thể của từng bảng.
    """
    # 1. Loại bỏ các dòng hoàn toàn trống hoặc trùng lặp
    df = df.drop_duplicates().dropna(how="all")

    for col in df.columns:
        col_lower = str(col).lower()

        # THƯỚC ĐO 1: NHẬN DIỆN CÁC CỘT TIỀN TỆ / GIÁ TRỊ (Giá, Tiền, Tổng, Doanh thu...)
        money_keywords = ['gia', 'tien', 'tong', 'thanh_tien', 'doanh_thu',
                          'chi_phi', 'loi_nhuan', 'don_gia', 'price', 'amount', 'total']
        if any(kw in col_lower for kw in money_keywords):
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                s = df[col].astype(str).str.strip()
                # Xóa ký hiệu tiền tệ và khoảng trắng thừa trước
                cleaned_series = (
                    s.str.replace(r'[đVND\s]', '', regex=True)
                     .replace(['nan', 'None', '', 'null', 'NAT', 'nan'], pd.NA)
                )

                # Xử lý thông minh:
                cleaned_series = cleaned_series.str.replace(',', '', regex=False)
                # Ép sang kiểu số thực (float)
                df[col] = pd.to_numeric(cleaned_series, errors='coerce')

        # THƯỚC ĐO 2: NHẬN DIỆN CÁC CỘT SỐ LƯỢNG (Số lượng, Sluong, Qty...)
        qty_keywords = ['sluong', 'so_luong', 'quantity', 'qty', 'sl', 'khoi_luong', 'trong_luong']
        if any(kw in col_lower for kw in qty_keywords):
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                # Lọc lấy phần số học đầu tiên nếu cột chứa chuỗi hỗn hợp (vd: "2 xe" -> lấy số 2 hoặc để nguyên nếu có cột quy đổi riêng)
                # Ở đây ta dùng hàm trích xuất số cơ bản hoặc pd.to_numeric
                cleaned_series = (
                    df[col].astype(str)
                    .str.extract(r'([\d\.]+)', expand=False) # Lấy phần số
                    .replace(['nan', 'None', ''], pd.NA)
                )
                # Chỉ ép kiểu nếu chuỗi thực sự là dạng số hợp lệ
                df[col] = pd.to_numeric(cleaned_series, errors='coerce').fillna(df[col]) # Giữ nguyên text gốc nếu extract thất bại

        # THƯỚC ĐO 3: NHẬN DIỆN CÁC CỘT NGÀY THÁNG (Ngày, Date, Time...)
        date_keywords = ['ngay', 'date', 'thoi_gian', 'time', 'created_at', 'updated_at', 'ngay_lap']
        if any(kw in col_lower for kw in date_keywords):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    # Thêm dayfirst=True nếu dữ liệu của bạn chắc chắn dùng chuẩn DD/MM/YYYY
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce').dt.date
            except Exception:
                pass
    return df

def sanitize_value(val):
    """
    Chuyển đổi các kiểu dữ liệu đặc biệt của Pandas/Numpy sang kiểu chuẩn Python
    để đảm bảo tương thích 100% với json.dump().
    """
    if val is None:
        return None
    # Kiểm tra NaT hoặc NaN của Pandas/Numpy
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()
    if isinstance(val, (pd.Timedelta, np.timedelta64)):
        return str(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val

def detect_dirty_data(df: pd.DataFrame, business_id: str) -> dict:
    """
    Phát hiện bất thường trong bảng dữ liệu:
    - Đơn bị hủy / trả hàng
    - Lệch định dạng số/chữ trong cột giá trị
    - Khác biệt tên gọi/tên trùng nhau
    """
    issues = {
        "business_id": business_id,
        "format_conflicts": [],
        "canceled_orders_count": 0,
        "returned_items_count": 0,
        "duplicate_rows_count": 0
    }

    for col in df.columns:
        col_str = str(col).lower()
        # Phát hiện đơn hủy hoặc hàng trả lại
        if any(kw in col_str for kw in ['trang_thai', 'status', 'huy', 'cancel']):
            canceled = df[df[col].astype(str).str.contains('hủy|cancel|canceled', case=False, na=False)]
            issues["canceled_orders_count"] += len(canceled)

        if any(kw in col_str for kw in ['tra_hang', 'return', 'hoan_tien', 'refund']):
            returned = df[df[col].astype(str).str.contains('trả|return|hoàn', case=False, na=False)]
            issues["returned_items_count"] += len(returned)

        # Phát hiện lệch kiểu dữ liệu còn sót
        if df[col].dtype == object:
            sample_vals = df[col].dropna().astype(str).head(30)
            if any(kw in col_str for kw in ['gia', 'tien', 'total']) and sample_vals.str.contains(r'[a-zA-Z]').any():
                issues["format_conflicts"].append(str(col))

    return issues

