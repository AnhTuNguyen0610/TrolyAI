# TroLy.AI — Trợ Lý AI Doanh Nghiệp Việt Nam

Một "Operator Agent" có thể:

- Nghiên cứu thị trường và viết báo cáo, lưu thành file thật
- Phân tích số liệu kinh doanh, tính KPI / ROI
- Ghi nhớ ngữ cảnh qua nhiều phiên làm việc (SQLite)
- Tự đánh giá chất lượng câu trả lời trước khi trả về (Worker-Evaluator)
- Phục vụ nhiều người dùng song song, dữ liệu tách biệt

## Kiến trúc

```
TroLy.AI/
├── tools_module.py     # Công cụ cho agent: tao_tools(thu_muc) -> tools gắn với sandbox của 1 user
├── session_manager.py  # Tách biệt phiên làm việc (PhienLamViec, QuanLyPhien)
├── agent_core.py       # Bộ não: Worker, Evaluator, Graph (LangGraph)
├── app.py              # Giao diện Gradio
├── workspaces/         # Không gian riêng cho mỗi người dùng
└── memory.db           # SQLite — bộ nhớ dài hạn (checkpointer), tự tạo khi chạy
```

### Luồng xử lý

```
START → worker ──(có tool_calls)──→ tools ──→ worker
           │
           └─(trả lời)──→ evaluator ──(đạt / cần hỏi user / hết 3 vòng)──→ END
                              └──(chưa đạt)──→ worker  (kèm feedback)
```

- **Worker**: LLM có gắn tools, nhận tiêu chuẩn thành công + feedback của vòng trước.
- **Evaluator**: LLM-as-a-Judge, structured output (`KetQuaDanhGia` — Pydantic).
- **Giới hạn**: tối đa `MAX_VONG_DANH_GIA = 3` vòng cho mỗi yêu cầu.

### Cô lập nhiều người dùng

1. Mỗi user có workspace riêng `workspaces/<ma_user>/`
2. Tools được tạo riêng cho từng user, chặn path traversal
3. Mỗi user một `thread_id` (`trolyai_<ma_user>`) trong checkpointer
4. Mỗi session có agent / kết nối SQLite riêng, được dọn khi đóng tab

## Cài đặt & chạy

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # điền OPENAI_API_KEY và SERPER_API_KEY
python app.py               # mở http://localhost:7860
```

LangSmith (tuỳ chọn): bỏ comment các biến `LANGSMITH_*` trong `.env` để xem trace.

## Kiểm tra

```bash
python test_tools.py        # tools + cô lập sandbox (không cần API key)
python test_graph.py        # graph Worker-Evaluator với LLM giả (không cần API key)
python test_multi_user.py   # mô phỏng 2 người dùng đồng thời
```

## Tools

| Tool | Chức năng |
|---|---|
| `tim_kiem_web` | Tìm kiếm Google qua Serper |
| `doc_file` / `ghi_file` / `liet_ke_files` | Quản lý file trong workspace |
| `chay_python` | Chạy Python trong subprocess, timeout 15s |
| `tinh_toan` | Tính biểu thức toán học (AST, không dùng `eval`) |

