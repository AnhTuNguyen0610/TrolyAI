"""
tools_module.py
Bộ công cụ cho TroLy.AI.

Mỗi người dùng có một bộ tools riêng, trỏ vào workspace riêng của họ:
    tools = tao_tools(Path("./workspaces/user_abc"))

Thêm tool mới: viết trong tao_tools() và thêm vào list trả về.
"""

import ast
import math
import operator
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(override=True)

GIOI_HAN_DOC = 8000      # số ký tự tối đa khi đọc một file
TIMEOUT_PYTHON = 15      # giây


# ─── Helpers ─────────────────────────────────────────────────────────

def _duong_dan_an_toan(sandbox: Path, ten_file: str) -> Path:
    """
    Ghép tên file vào sandbox và đảm bảo kết quả vẫn nằm TRONG sandbox.
    Dùng is_relative_to thay vì startswith: startswith sẽ cho phép
    'workspaces/user_1' truy cập nhầm 'workspaces/user_10'.
    """
    goc = sandbox.resolve()
    duong_dan = (goc / ten_file).resolve()
    if not duong_dan.is_relative_to(goc):
        raise PermissionError("Đường dẫn nằm ngoài thư mục làm việc.")
    return duong_dan


_TOAN_TU = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_HAM = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "abs": abs, "round": round, "min": min, "max": max,
}
_HANG = {"pi": math.pi, "e": math.e}


def _tinh_ast(node):
    """Tính biểu thức toán học qua AST — chỉ cho phép số, + - * / % ** và vài hàm."""
    if isinstance(node, ast.Expression):
        return _tinh_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _HANG:
        return _HANG[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _TOAN_TU:
        trai, phai = _tinh_ast(node.left), _tinh_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(phai) > 100:
            raise ValueError("Số mũ quá lớn.")
        return _TOAN_TU[type(node.op)](trai, phai)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _tinh_ast(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _HAM and not node.keywords):
        return _HAM[node.func.id](*[_tinh_ast(a) for a in node.args])
    raise ValueError("Biểu thức không hợp lệ hoặc dùng cú pháp không được phép.")


# ─── Factory ─────────────────────────────────────────────────────────

def tao_tools(thu_muc_lam_viec: Path) -> List:
    """Tạo bộ tools với sandbox trỏ vào thư mục riêng của một người dùng."""
    sandbox = Path(thu_muc_lam_viec)
    sandbox.mkdir(parents=True, exist_ok=True)

    _serper = {}  # khởi tạo lười: import module không cần SERPER_API_KEY

    def lay_serper():
        if "client" not in _serper:
            from langchain_community.utilities import GoogleSerperAPIWrapper
            _serper["client"] = GoogleSerperAPIWrapper()
        return _serper["client"]

    @tool
    def tim_kiem_web(truy_van: str) -> str:
        """
        Tìm kiếm thông tin trên internet.
        Dùng khi cần: giá cả thị trường, tin tức ngành, thông tin đối thủ,
        xu hướng kinh doanh, số liệu thống kê.
        Tham số truy_van: câu hỏi hoặc từ khóa tìm kiếm.
        """
        try:
            return f"Kết quả tìm kiếm cho '{truy_van}':\n{lay_serper().run(truy_van)}"
        except Exception as e:
            return f"Lỗi khi tìm kiếm: {e}"

    @tool
    def doc_file(ten_file: str) -> str:
        """
        Đọc nội dung một file văn bản trong thư mục làm việc.
        Hỗ trợ: .txt, .md, .csv, .json, .py
        Tham số ten_file: chỉ tên file (ví dụ: 'bao_cao.md'), không dùng đường dẫn tuyệt đối.
        """
        try:
            duong_dan = _duong_dan_an_toan(sandbox, ten_file)
        except (PermissionError, OSError):
            return "Lỗi: Không được phép truy cập file ngoài thư mục làm việc."

        if not duong_dan.is_file():
            cac_file = [f.name for f in sandbox.iterdir() if f.is_file()]
            return (f"File '{ten_file}' không tồn tại.\n"
                    f"Các file hiện có: {', '.join(cac_file) if cac_file else 'Chưa có file nào.'}")
        try:
            noi_dung = duong_dan.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Không thể đọc '{ten_file}' — có thể là file nhị phân."
        except Exception as e:
            return f"Lỗi khi đọc file: {e}"

        tong = len(noi_dung)
        if tong > GIOI_HAN_DOC:
            noi_dung = noi_dung[:GIOI_HAN_DOC] + f"\n\n[... Đã cắt bớt. File có {tong} ký tự]"
        return f"Nội dung file '{ten_file}':\n\n{noi_dung}"

    @tool
    def ghi_file(ten_file: str, noi_dung: str, che_do: str = "ghi_moi") -> str:
        """
        Tạo hoặc cập nhật một file văn bản trong thư mục làm việc.
        Tham số:
        - ten_file: tên file (ví dụ: 'bao_cao_q1.md')
        - noi_dung: nội dung cần ghi
        - che_do: 'ghi_moi' (ghi đè) hoặc 'them_vao' (nối vào cuối)
        Khuyến nghị: .md cho báo cáo, .csv cho dữ liệu, .txt cho ghi chú.
        """
        try:
            duong_dan = _duong_dan_an_toan(sandbox, ten_file)
        except (PermissionError, OSError):
            return "Lỗi bảo mật: Không được ghi file ngoài thư mục làm việc."

        try:
            duong_dan.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if che_do == "them_vao" else "w"
            with open(duong_dan, mode, encoding="utf-8") as f:
                f.write(noi_dung)
            hanh_dong = "thêm vào" if che_do == "them_vao" else "ghi"
            return (f"Đã {hanh_dong} file '{ten_file}' thành công. "
                    f"Kích thước: {duong_dan.stat().st_size:,} bytes.")
        except Exception as e:
            return f"Lỗi khi ghi file: {e}"

    @tool
    def liet_ke_files() -> str:
        """Liệt kê các file trong thư mục làm việc. Không cần tham số."""
        files = sorted(f for f in sandbox.iterdir() if f.is_file())
        if not files:
            return "Thư mục làm việc hiện đang trống."
        dong = [f"  {f.name} ({f.stat().st_size:,} bytes)" for f in files]
        return "Files trong thư mục làm việc:\n" + "\n".join(dong)

    @tool
    def chay_python(code: str) -> str:
        """
        Chạy đoạn code Python và trả về output. PHẢI dùng print() để nhận kết quả.
        Dùng khi: tính toán số liệu, xử lý dữ liệu, thống kê.
        Giới hạn: timeout 15 giây; chạy trong thư mục làm việc của người dùng.
        Có sẵn: math, statistics, json, csv, re, datetime, collections.
        """
        code_day_du = (
            "import math, statistics, json, csv, re\n"
            "from datetime import datetime, date, timedelta\n"
            "from collections import Counter, defaultdict\n\n"
            + code
        )
        try:
            kq = subprocess.run(
                [sys.executable, "-c", code_day_du],
                capture_output=True, text=True, timeout=TIMEOUT_PYTHON,
                cwd=sandbox,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return f"Timeout: Code chạy quá {TIMEOUT_PYTHON} giây. Hãy tối ưu lại."
        except Exception as e:
            return f"Không thể chạy code: {e}"

        if kq.returncode == 0:
            return kq.stdout.strip() or "Code chạy xong nhưng không có output. Hãy thêm print()."
        dong_loi = kq.stderr.strip().split("\n")
        return "Lỗi Python:\n" + "\n".join(dong_loi[-3:])

    @tool
    def tinh_toan(bieu_thuc: str) -> str:
        """
        Tính biểu thức toán học: lợi nhuận, %, ROI, các phép tính tài chính đơn giản.
        Ví dụ: '(800000000 - 550000000) / 800000000 * 100'
        Hỗ trợ + - * / % ** ^, ngoặc, sqrt, log, log10, abs, round, min, max, pi, e.
        Phép tính phức tạp hơn thì dùng chay_python.
        """
        try:
            bt = bieu_thuc.replace(",", "").replace("^", "**")
            kq = _tinh_ast(ast.parse(bt, mode="eval"))
        except ZeroDivisionError:
            return "Lỗi: Chia cho 0."
        except Exception as e:
            return f"Không thể tính: {e}. Hãy dùng chay_python cho phép tính phức tạp hơn."
        if isinstance(kq, float):
            return f"{bieu_thuc} = {kq:,.4f}"
        return f"{bieu_thuc} = {kq:,}"

    return [tim_kiem_web, doc_file, ghi_file, liet_ke_files, chay_python, tinh_toan]


def lay_mo_ta_tools(tools: List) -> str:
    """Mô tả ngắn các tools (dòng đầu docstring) — dùng cho system prompt."""
    return "\n".join(f"- {t.name}: {t.description.strip().splitlines()[0]}" for t in tools)
