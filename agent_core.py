# agent_core.py

import uuid
import sqlite3
from typing import Annotated, TypedDict, List, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)


class TrangThaiTroLy(TypedDict):
    messages: Annotated[List[Any], add_messages]


class TroLyAI:
    """Trợ Lý AI Doanh Nghiệp VN — bản đầu: chỉ có Worker + Tools + Memory."""

    def __init__(self, ma_nguoi_dung: str, thu_muc_lam_viec: Path):
        self.ma_nguoi_dung = ma_nguoi_dung
        self.thu_muc_lam_viec = thu_muc_lam_viec
        self.session_id = str(uuid.uuid4())[:8]
        self.thread_id = f"trolyai_{ma_nguoi_dung}"
        self.graph = None
        self._khoi_tao()

    def _khoi_tao(self):
        self.tools = self._tao_tools()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.worker_llm = llm.bind_tools(self.tools)

        self.conn = sqlite3.connect("memory.db", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        self._xay_graph()

    def _tao_tools(self) -> list:
        from langchain_core.tools import tool
        from langchain_community.utilities import GoogleSerperAPIWrapper

        serper = GoogleSerperAPIWrapper()
        sandbox = self.thu_muc_lam_viec

        @tool
        def tim_kiem_web(truy_van: str) -> str:
            """Tìm kiếm thông tin trên internet. Dùng khi cần thông tin thị trường, tin tức, số liệu."""
            try:
                return f"Kết quả cho '{truy_van}':\n" + serper.run(truy_van)
            except Exception as e:
                return f"Lỗi tìm kiếm: {e}"

        @tool
        def ghi_bao_cao(ten_file: str, noi_dung: str) -> str:
            """Lưu báo cáo/kết quả vào file .md hoặc .txt trong workspace."""
            duong_dan = sandbox / ten_file
            duong_dan.write_text(noi_dung, encoding="utf-8")
            return f"Đã lưu '{ten_file}' ({duong_dan.stat().st_size} bytes)."

        @tool
        def doc_tai_lieu(ten_file: str) -> str:
            """Đọc file đã lưu trước đó trong workspace. Tham số: tên file."""
            duong_dan = sandbox / ten_file
            if not duong_dan.exists():
                files = [f.name for f in sandbox.iterdir() if f.is_file()]
                return f"File không tồn tại. Files có sẵn: {files or 'chưa có'}"
            return duong_dan.read_text(encoding="utf-8")[:6000]

        @tool
        def xem_workspace() -> str:
            """Xem danh sách files trong workspace cá nhân."""
            files = [f"{f.name} ({f.stat().st_size:,} bytes)"
                     for f in sorted(sandbox.iterdir()) if f.is_file()]
            return "\n".join(files) if files else "Workspace đang trống."

        return [tim_kiem_web, ghi_bao_cao, doc_tai_lieu, xem_workspace]

    def _node_worker(self, state: TrangThaiTroLy) -> dict:
        system_content = (
            "Bạn là TroLy.AI — Trợ Lý AI Doanh Nghiệp cho doanh nghiệp Việt Nam.\n"
            f"Thời gian hiện tại: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
            f"Người dùng: {self.ma_nguoi_dung}\n"
            "Dùng tiếng Việt tự nhiên, chuyên nghiệp. Cần tìm thông tin thì dùng tim_kiem_web, "
            "viết báo cáo thì lưu bằng ghi_bao_cao."
        )
        tin_nhan = [SystemMessage(content=system_content)] + list(state["messages"])
        return {"messages": [self.worker_llm.invoke(tin_nhan)]}

    def _dinh_tuyen_worker(self, state: TrangThaiTroLy) -> str:
        msg_cuoi = state["messages"][-1]
        if getattr(msg_cuoi, "tool_calls", None):
            return "tools"
        return END

    def _xay_graph(self):
        builder = StateGraph(TrangThaiTroLy)
        builder.add_node("worker", self._node_worker)
        builder.add_node("tools", ToolNode(tools=self.tools))

        builder.add_edge(START, "worker")
        builder.add_conditional_edges(
            "worker", self._dinh_tuyen_worker, {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "worker")
        self.graph = builder.compile(checkpointer=self.checkpointer)

    def xu_ly_yeu_cau(self, tin_nhan: str) -> dict:
        config = {"configurable": {"thread_id": self.thread_id}}
        ket_qua = self.graph.invoke(
            {"messages": [HumanMessage(content=tin_nhan)]}, config=config
        )
        return {"tra_loi": ket_qua["messages"][-1].content}

    def don_dep(self):
        try:
            self.conn.close()
        except Exception:
            pass
