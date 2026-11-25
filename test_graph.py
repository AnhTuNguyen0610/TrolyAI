# test_graph.py — kiểm tra graph Worker-Evaluator bằng LLM giả (không cần API key)
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["TROLYAI_DB"] = str(Path(tempfile.mkdtemp()) / "memory.db")

from langchain_core.messages import AIMessage

from agent_core import KetQuaDanhGia, TroLyAI


class LLMGiaWorker:
    """Lần 1 gọi tool ghi_file, sau đó trả lời; đếm số lần được gọi."""
    def __init__(self):
        self.so_lan = 0

    def invoke(self, messages):
        self.so_lan += 1
        if self.so_lan == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "ghi_file", "id": "c1",
                "args": {"ten_file": "bc.md", "noi_dung": "báo cáo"}}])
        return AIMessage(content=f"Đã lưu báo cáo (lần {self.so_lan})")


class LLMGiaEvaluator:
    def __init__(self, dat):
        self.dat = dat
        self.so_lan = 0

    def invoke(self, messages):
        self.so_lan += 1
        return KetQuaDanhGia(phan_hoi="ok", da_dat_tieu_chuan=self.dat, can_them_thong_tin=False)


def tao_agent(ma_user, dat):
    ws = Path(tempfile.mkdtemp()) / ma_user
    ws.mkdir(parents=True)
    agent = TroLyAI(ma_user, ws)
    agent.worker_llm = LLMGiaWorker()
    agent.evaluator_llm = LLMGiaEvaluator(dat)
    return agent, ws


# 1. Worker gọi tool -> ghi file trong workspace đúng user -> evaluator duyệt
agent, ws = tao_agent("u1", dat=True)
kq = agent.xu_ly_yeu_cau("Viết báo cáo")
assert (ws / "bc.md").exists(), "tool không ghi file vào workspace"
assert kq["hoan_thanh"] and "Đã lưu" in kq["tra_loi"]
assert agent.evaluator_llm.so_lan == 1
print("PASS - worker -> tool -> evaluator")

# 2. Evaluator luôn từ chối -> dừng sau MAX_VONG_DANH_GIA vòng
agent, _ = tao_agent("u2", dat=False)
agent.xu_ly_yeu_cau("yêu cầu 1")
assert agent.evaluator_llm.so_lan == 3, agent.evaluator_llm.so_lan
print("PASS - giới hạn 3 vòng")

# 3. Yêu cầu thứ 2 trên cùng thread vẫn được 3 vòng (không bị đếm lịch sử cũ)
agent.xu_ly_yeu_cau("yêu cầu 2")
assert agent.evaluator_llm.so_lan == 6, agent.evaluator_llm.so_lan
print("PASS - bộ đếm vòng tính theo từng yêu cầu")

print("\nGraph hoạt động đúng!")
