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
- **Evaluator**: LLM-as-a-Judge, structured output (`KetQuaDanhGia` — Pydantic, qua function calling).
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
cp .env.example .env        # chọn LLM_PROVIDER, điền key tương ứng + SERPER_API_KEY
python app.py               # mở http://localhost:7860
```

### Chọn LLM provider (không cần OpenAI)

Đặt `LLM_PROVIDER` trong `.env`. Tất cả đều dùng API tương thích OpenAI nên code không đổi:

| `LLM_PROVIDER` | Key cần điền | Model mặc định | Ghi chú |
|---|---|---|---|
| `gemini` (mặc định) | `GOOGLE_API_KEY` | `gemini-2.5-flash` | Free tier, giới hạn theo phút/ngày |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Free tier, rất nhanh |
| `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.1-8b-instruct:free` | Model `:free` thay đổi thường xuyên |
| `ollama` | không cần | `llama3.2` | Chạy local, `ollama pull llama3.2` trước |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | Trả phí |

Đổi model bằng `LLM_MODEL=...` (tên model free hay bị đổi/ngừng, nếu lỗi *model not found* hãy kiểm tra lại danh sách của nhà cung cấp).

**Lưu ý khi dùng gói miễn phí**
- Agent cần model **hỗ trợ tool calling**. Model nhỏ (ví dụ 8B, `llama3.2` 3B) có thể gọi tool sai hoặc bỏ qua tool; nếu thấy vậy hãy thử model lớn hơn.
- Mỗi yêu cầu có thể gọi LLM nhiều lần (Worker + tools + Evaluator, tối đa 3 vòng) nên dễ chạm rate limit. Code đã tự retry khi dính 429; nếu vẫn lỗi hãy đợi một phút hoặc giảm `MAX_VONG_DANH_GIA` trong `agent_core.py`.
- Nếu Evaluator lỗi (schema hỏng, rate limit), agent bỏ qua đánh giá và trả luôn câu trả lời của Worker.
- Tìm kiếm web cần `SERPER_API_KEY` (serper.dev có gói miễn phí).

LangSmith (tuỳ chọn): bỏ comment các biến `LANGSMITH_*` trong `.env` để xem trace.

## Kiểm tra

```bash
python test_tools.py        # tools + cô lập sandbox (không cần API key)
python test_graph.py        # graph Worker-Evaluator với LLM giả (không cần API key)
python test_llm_config.py   # chọn provider qua .env (không gọi mạng)
python test_multi_user.py   # mô phỏng 2 người dùng đồng thời
```

## Tools

| Tool | Chức năng |
|---|---|
| `tim_kiem_web` | Tìm kiếm Google qua Serper |
| `doc_file` / `ghi_file` / `liet_ke_files` | Quản lý file trong workspace |
| `chay_python` | Chạy Python trong subprocess, timeout 15s |
| `tinh_toan` | Tính biểu thức toán học (AST, không dùng `eval`) |

