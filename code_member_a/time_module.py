from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd

class TimeModule:
    """
    Module hỗ trợ truy vấn các khoảng thời gian theo đúng quy tắc Rulebook:
    as-of date, tuần Thứ Hai–Chủ Nhật, "tháng này", "tuần rồi", "cùng kỳ năm ngoái", cửa sổ Tết.
    """
    def __init__(self, calendar_csv_path: Path, as_of_date: str = None):
        self.as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else date.today()
        self.calendar_df = pd.read_csv(calendar_csv_path) if calendar_csv_path.exists() else None

    def get_current_week(self):
        """Tuần này: Từ Thứ Hai của tuần chứa as_of_date đến hết ngày as_of_date"""
        start_week = self.as_of_date - timedelta(days=self.as_of_date.weekday())
        return start_week, self.as_of_date

    def get_last_week(self):
        """Tuần rồi / Tuần trước: Cả tuần liền trước (Thứ Hai -> Chủ Nhật)"""
        start_last_week = self.as_of_date - timedelta(days=self.as_of_date.weekday() + 7)
        end_last_week = start_last_week + timedelta(days=6)
        return start_last_week, end_last_week

    def get_this_month(self):
        """Tháng này: Từ ngày 1 đến hết ngày as_of_date"""
        start_month = self.as_of_date.replace(day=1)
        return start_month, self.as_of_date

    def get_same_period_last_year(self):
        """Cùng kỳ năm ngoái so với khoảng ngày hiện tại"""
        try:
            as_of_ly = self.as_of_date.replace(year=self.as_of_date.year - 1)
        except ValueError:  # Xử lý năm nhuận (29/02 -> 28/02)
            as_of_ly = self.as_of_date.replace(year=self.as_of_date.year - 1, day=28)
        start_ly = as_of_ly.replace(day=1)
        return start_ly, as_of_ly

    def get_tet_window(self, year: int):
        """Cửa sổ Tết: Từ 23 tháng Chạp đến hết mùng 10 tháng Giêng (Tra từ calendar.csv)"""
        if self.calendar_df is not None:
            tet_rows = self.calendar_df[
                (self.calendar_df['year'] == year) & (self.calendar_df['is_tet_window'] == 1)
            ]
            if not tet_rows.empty:
                return tet_rows['start_date'].values[0], tet_rows['end_date'].values[0]
        return None, None
