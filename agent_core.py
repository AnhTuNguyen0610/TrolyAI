# agent_core.py

import uuid
import sqlite3
from typing import Annotated, TypedDict, Optional, List, Any
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from tools_module import tao_tools, lay_mo_ta_tools

load_dotenv(override=True)

MAX_VONG_DANH_GIA = 3


# ─── State ────────────────────────────────────────────────────────────

class TrangThaiTroLy(TypedDict):
    messages: Annotated[List[Any], add_messages]
    tieu_chuan_thanh_cong: str
    phan_hoi_cong_viec: Optional[str]
    da_hoan_thanh: bool
    can_nguoi_dung: bool
    so_vong: int  # số vòng Worker-Evaluator của YÊU CẦU HIỆN TẠI


# ─── Evaluator Output Schema ──────────────────────────────────────────

class KetQuaDanhGia(BaseModel):
    phan_hoi: str = Field(
        description="Nhận xét chi tiết về câu trả lời của trợ lý"
    )
    da_dat_tieu_chuan: bool = Field(
        description="True nếu câu trả lời đã đáp ứng tiêu chuẩn thành công"
    )
    can_them_thong_tin: bool = Field(
        description="True nếu trợ lý đang hỏi thêm hoặc không thể tiến hành mà không có thêm thông tin"
    )


# ─── TroLyAI Agent ────────────────────────────────────────────────────

class TroLyAI:
    """
    Trợ Lý AI Doanh Nghiệp Việt Nam.
    Kiến trúc: Worker (có tools) + Evaluator (có structured output) + Memory (SQLite).
    """

    def __init__(self, ma_nguoi_dung: str, thu_muc_lam_viec: Path):
        self.ma_nguoi_dung = ma_nguoi_dung
        self.thu_muc_lam_viec = thu_muc_lam_viec
        self.session_id = str(uuid.uuid4())[:8]
        self.thread_id = f"trolyai_{ma_nguoi_dung}"  # Dùng ma_nguoi_dung để nhớ qua session
        self.graph = None
        self._khoi_tao()

    def _khoi_tao(self):
        """Khởi tạo toàn bộ hệ thống."""
        # Tools — với sandbox trỏ đến thư mục riêng của user này
        self.tools = tao_tools(self.thu_muc_lam_viec)

        # LLM cho Worker — có tools
        llm_worker = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.worker_llm = llm_worker.bind_tools(self.tools)

        # LLM cho Evaluator — structured output, nghiêm khắc hơn
        llm_evaluator = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.evaluator_llm = llm_evaluator.with_structured_output(KetQuaDanhGia)

        # SQLite checkpointer — nhớ qua các session
        self.conn = sqlite3.connect("memory.db", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)

        # Build graph
        self._xay_graph()

    def _node_worker(self, state: TrangThaiTroLy) -> dict:
        """Node Worker: LLM với tools, xử lý yêu cầu của người dùng."""
        tieu_chuan = state["tieu_chuan_thanh_cong"]
        phan_hoi_cu = state.get("phan_hoi_cong_viec") or ""

        system_content = f"""Bạn là TroLy.AI — Trợ Lý AI Doanh Nghiệp chuyên nghiệp, được thiết kế đặc biệt cho doanh nghiệp Việt Nam.

Thời gian hiện tại: {datetime.now().strftime("%d/%m/%Y %H:%M")}
Người dùng hiện tại: {self.ma_nguoi_dung}

Công cụ bạn có:
{lay_mo_ta_tools(self.tools)}

Tiêu chuẩn cần đạt cho nhiệm vụ này:
{tieu_chuan}

Nguyên tắc làm việc:
- Dùng tiếng Việt tự nhiên, chuyên nghiệp
- Cần thông tin bên ngoài: dùng tim_kiem_web
- Có số liệu cần tính: dùng tinh_toan (hoặc chay_python nếu phức tạp)
- Khi viết báo cáo: lưu file bằng ghi_file
- Nếu cần thêm thông tin từ người dùng: hỏi rõ ràng — bắt đầu bằng "Câu hỏi:"
- Khi hoàn thành: trả lời trực tiếp, không hỏi thêm"""

        if phan_hoi_cu:
            system_content += f"""

LƯU Ý: Lần trước câu trả lời của bạn chưa đạt tiêu chuẩn.
Phản hồi cụ thể cần khắc phục:
{phan_hoi_cu}

Hãy làm lại và đảm bảo khắc phục được tất cả điểm trên."""

        # Bỏ các tin nhắn "[Evaluator]" khỏi context của Worker: chúng chỉ để hiển thị,
        # nếu để lại Worker sẽ bắt chước format đó và nhiễu lịch sử.
        tin_nhan = [
            m for m in state["messages"]
            if not (isinstance(m, AIMessage) and (m.content or "").startswith("[Evaluator]"))
            and not isinstance(m, SystemMessage)
        ]
        tin_nhan_moi = [SystemMessage(content=system_content)] + tin_nhan

        phan_hoi = self.worker_llm.invoke(tin_nhan_moi)
        return {"messages": [phan_hoi]}

    def _node_evaluator(self, state: TrangThaiTroLy) -> dict:
        """Node Evaluator: Đánh giá output của Worker."""
        phan_hoi_cuoi = state["messages"][-1].content or ""
        tieu_chuan = state["tieu_chuan_thanh_cong"]
        phan_hoi_truoc = state.get("phan_hoi_cong_viec", "")

        # Tóm tắt lịch sử hội thoại
        lich_su = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                lich_su.append(f"Người dùng: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                lich_su.append(f"Trợ lý: {msg.content[:300]}...")
        tom_tat = "\n".join(lich_su[-8:])  # 8 tin nhắn gần nhất

        ghi_chu_lich_su = ""
        if phan_hoi_truoc:
            ghi_chu_lich_su = f"\nBạn đã feedback trước đó: {phan_hoi_truoc}\nNếu trợ lý vẫn mắc lỗi cũ, hãy đặt can_them_thong_tin=True."

        ket_qua = self.evaluator_llm.invoke([
            SystemMessage(content=f"""Bạn là Evaluator — đánh giá chất lượng câu trả lời của Trợ Lý AI.

Tiêu chuẩn thành công:
{tieu_chuan}

Nguyên tắc đánh giá:
- Chỉ đặt da_dat_tieu_chuan=True khi ĐẠT TẤT CẢ tiêu chuẩn
- Feedback phải cụ thể và có thể thực hiện được
- Nếu trợ lý nói đã ghi file, hãy tin là đã làm xong
- Nếu trợ lý hỏi người dùng, đặt can_them_thong_tin=True
{ghi_chu_lich_su}"""),
            HumanMessage(content=f"""Lịch sử hội thoại:
{tom_tat}

Câu trả lời cuối cùng của trợ lý cần đánh giá:
{phan_hoi_cuoi}

Đánh giá và quyết định:""")
        ])

        return {
            "messages": [AIMessage(
                content=f"[Evaluator] Điểm: {'Đạt' if ket_qua.da_dat_tieu_chuan else 'Chưa đạt'} | {ket_qua.phan_hoi}"
            )],
            "phan_hoi_cong_viec": ket_qua.phan_hoi,
            "da_hoan_thanh": ket_qua.da_dat_tieu_chuan,
            "can_nguoi_dung": ket_qua.can_them_thong_tin,
            "so_vong": state.get("so_vong", 0) + 1,
        }

    def _dinh_tuyen_worker(self, state: TrangThaiTroLy) -> str:
        """Sau Worker: đến tools hay evaluator?"""
        msg_cuoi = state["messages"][-1]
        if hasattr(msg_cuoi, "tool_calls") and msg_cuoi.tool_calls:
            return "tools"
        return "evaluator"

    def _dinh_tuyen_evaluator(self, state: TrangThaiTroLy) -> str:
        """Sau Evaluator: xong hay cần làm lại?"""
        if state.get("da_hoan_thanh") or state.get("can_nguoi_dung"):
            return END
        # Giới hạn số vòng của yêu cầu hiện tại để tránh loop vô hạn.
        # (Trước đây đếm "[Evaluator]" trong cả lịch sử thread -> sau vài lượt chat
        # thì vòng lặp tự sửa sai bị tắt luôn.)
        if state.get("so_vong", 0) >= MAX_VONG_DANH_GIA:
            return END
        return "worker"

    def _xay_graph(self):
        """5 bước xây LangGraph."""
        builder = StateGraph(TrangThaiTroLy)

        builder.add_node("worker",    self._node_worker)
        builder.add_node("tools",     ToolNode(tools=self.tools))
        builder.add_node("evaluator", self._node_evaluator)

        builder.add_edge(START, "worker")
        builder.add_conditional_edges(
            "worker",
            self._dinh_tuyen_worker,
            {"tools": "tools", "evaluator": "evaluator"}
        )
        builder.add_edge("tools", "worker")
        builder.add_conditional_edges(
            "evaluator",
            self._dinh_tuyen_evaluator,
            {"worker": "worker", END: END}
        )

        self.graph = builder.compile(checkpointer=self.checkpointer)

    def xu_ly_yeu_cau(self, tin_nhan: str, tieu_chuan: str = "") -> dict:
        """
        Xử lý một yêu cầu từ người dùng.
        Trả về dict với câu trả lời và metadata.
        """
        config = {"configurable": {"thread_id": self.thread_id}}

        state_dau_vao = {
            "messages": [HumanMessage(content=tin_nhan)],
            "tieu_chuan_thanh_cong": tieu_chuan or "Câu trả lời rõ ràng, chính xác, và hữu ích.",
            "phan_hoi_cong_viec": None,
            "da_hoan_thanh": False,
            "can_nguoi_dung": False,
            "so_vong": 0,
        }

        ket_qua = self.graph.invoke(state_dau_vao, config=config)

        # Lấy câu trả lời chính (message trước message evaluator)
        cac_ai_msg = [
            m for m in ket_qua["messages"]
            if isinstance(m, AIMessage)
            and m.content
            and "[Evaluator]" not in m.content
            and (
                m.tool_calls == [] if hasattr(m, "tool_calls") else True
            )
        ]
        
        cau_tra_loi = cac_ai_msg[-1].content if cac_ai_msg else "Không có câu trả lời."

        evaluator_msg = next(
            (m.content for m in reversed(ket_qua["messages"])
             if isinstance(m, AIMessage) and "[Evaluator]" in (m.content or "")),
            ""
        )

        return {
            "tra_loi": cau_tra_loi,
            "danh_gia": evaluator_msg,
            "hoan_thanh": ket_qua.get("da_hoan_thanh", False),
            "can_nguoi_dung": ket_qua.get("can_nguoi_dung", False),
        }

    def don_dep(self):
        """Dọn dẹp tài nguyên."""
        try:
            self.conn.close()
        except Exception:
            pass