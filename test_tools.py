# test_tools.py — kiểm tra tools_module (không cần API key; bỏ qua web search)
import tempfile
from pathlib import Path

from tools_module import tao_tools

tmp = Path(tempfile.mkdtemp())
tools = {t.name: t for t in tao_tools(tmp / "user_a")}
tools_b = {t.name: t for t in tao_tools(tmp / "user_b")}


def kiem_tra(ten, dieu_kien):
    print(("PASS" if dieu_kien else "FAIL"), "-", ten)
    assert dieu_kien, ten


print("=== Ghi / đọc / liệt kê file ===")
kq = tools["ghi_file"].invoke({"ten_file": "test.md", "noi_dung": "# Test\nNội dung"})
kiem_tra("ghi file", "thành công" in kq)
kiem_tra("đọc file", "Nội dung" in tools["doc_file"].invoke({"ten_file": "test.md"}))
tools["ghi_file"].invoke({"ten_file": "test.md", "noi_dung": "\nthêm", "che_do": "them_vao"})
kiem_tra("nối thêm", "thêm" in tools["doc_file"].invoke({"ten_file": "test.md"}))
kiem_tra("liệt kê", "test.md" in tools["liet_ke_files"].invoke({}))

print("\n=== Cô lập sandbox ===")
kiem_tra("user_b không thấy file của user_a", "test.md" not in tools_b["liet_ke_files"].invoke({}))
kq = tools["doc_file"].invoke({"ten_file": "../user_b/x.md"})
kiem_tra("chặn path traversal (đọc)", "ngoài thư mục" in kq)
kq = tools["ghi_file"].invoke({"ten_file": "../user_b/x.md", "noi_dung": "hack"})
kiem_tra("chặn path traversal (ghi)", "bảo mật" in kq)
(tmp / "user_a10").mkdir()
kq = tools["doc_file"].invoke({"ten_file": "../user_a10/x.md"})
kiem_tra("chặn thư mục cùng tiền tố (user_a vs user_a10)", "ngoài thư mục" in kq)

print("\n=== Python ===")
kq = tools["chay_python"].invoke({"code": "print(sum([120,145,132])/3)"})
kiem_tra("chạy python", kq == "132.33333333333334")
kiem_tra("báo lỗi python", "Lỗi Python" in tools["chay_python"].invoke({"code": "1/0"}))
kiem_tra("nhắc thêm print()", "print()" in tools["chay_python"].invoke({"code": "x = 1"}))

print("\n=== Tính toán ===")
kq = tools["tinh_toan"].invoke({"bieu_thuc": "(500000000 - 320000000) / 500000000 * 100"})
kiem_tra("biên lợi nhuận", "36.0000" in kq)
kiem_tra("chia cho 0", "Chia cho 0" in tools["tinh_toan"].invoke({"bieu_thuc": "1/0"}))
kiem_tra("chặn code độc", "Không thể tính" in tools["tinh_toan"].invoke(
    {"bieu_thuc": "__import__('os').system('echo hi')"}))
kiem_tra("hàm sqrt", "12" in tools["tinh_toan"].invoke({"bieu_thuc": "sqrt(144)"}))

print("\nTất cả tools đều hoạt động!")
