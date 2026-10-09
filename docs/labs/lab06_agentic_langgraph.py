"""
Lab 06 — Agentic RAG cho Zendesk bằng LangGraph: định tuyến, tool chỉ đọc, CRAG-lite,
kiểm tra citation, quyết định SEND / DRAFT / ESCALATE và human-in-the-loop bằng interrupt.

Chạy (trong thư mục labs/):
    python lab06_agentic_langgraph.py                  # chạy 60 email, in bảng tổng hợp
    python lab06_agentic_langgraph.py --show E-019     # in đường đi qua đồ thị của một email
    python lab06_agentic_langgraph.py --resume-demo    # dừng ở human_approval rồi resume (cần langgraph)
    python lab06_agentic_langgraph.py --no-langgraph   # chạy cùng các node bằng bộ điều phối tối giản

LLM: mặc định LLM_MODE=mock (sinh draft trích đoạn, không cần GPU). Đặt LLM_MODE=openai
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M để dùng Ollama/vLLM.
Các node và cạnh là MỘT nguồn duy nhất: LangGraph và bộ điều phối tối giản dùng chung hàm.
"""
from __future__ import annotations

import argparse
import json
import operator
import os
import re
from collections import Counter
from typing import Annotated, Any, TypedDict

from common import (SimpleBM25, article_text, detect_lang, email_query, load_articles,
                    load_emails, strip_accents)

PHASE = int(os.getenv("PHASE", "2"))                 # 1: chỉ draft; 2: được tự gửi intent rủi ro thấp
REQUIRE_APPROVAL = os.getenv("REQUIRE_APPROVAL", "1") == "1"
TAU_LOW, TAU_UP = 0.35, 0.60                          # ngưỡng CRAG-lite trên điểm retrieval đã chuẩn hóa
MAX_TOOL_CALLS = 2

# ---------------------------------------------------------------------------
# 1. Quy tắc phân loại (bản rút gọn của lab04; xem Module 07, 10)
# ---------------------------------------------------------------------------
RULES = {
    "wants_human": [r"gap (nguoi|nhan vien)", r"noi chuyen voi (nguoi|nhan vien)", r"human agent", r"talk to someone",
                    r"speak (to|with) an agent",
                    r"speak (to|with) (a )?(person|human|someone)", r"real person", r"担当者", r"人と話"],
    "sensitive": [r"hoan tien", r"refund", r"返金", r"huy (dich vu|hop dong|goi)", r"cancel",
                  r"luat su|kien|phap ly|legal|lawyer", r"bao gia|discount|giam gia|pricing quote",
                  r"bi hack|ro ri|lo du lieu|breach|leak|不正アクセス", r"(he thong|server) (bi )?sap|outage|down toan bo"],
    "injection": [r"ignore (all|previous|the above)", r"bo qua (moi|cac|tat ca) (huong dan|chi dan)",
                  r"system prompt", r"you are now", r"指示.{0,10}無視", r"<!--.{0,40}(assistant|system)",
                  r"do not cite"],
    "account": [r"hoa don (cua|thang)", r"my invoice", r"請求書", r"(da dung|usage|quota|han muc)",
                r"goi (hien tai|cua (toi|minh|em))", r"current plan"],
}
_RX = {k: [re.compile(p, re.I) for p in v] for k, v in RULES.items()}


def flags_of(text: str) -> dict[str, bool]:
    norm = strip_accents(text.lower())
    return {k: any(p.search(norm) or p.search(text) for p in pats) for k, pats in _RX.items()}


# ---------------------------------------------------------------------------
# 2. "API tài khoản" giả lập — tenant lấy từ tên miền người gửi (phía server), KHÔNG từ LLM
# ---------------------------------------------------------------------------
def fake_account_db(emails: list[dict]) -> dict[str, dict]:
    db = {}
    for i, e in enumerate(emails):
        dom = e["from"].split("@")[-1]
        db.setdefault(dom, {"org": dom, "plan": ["STARTER", "PRO", "BUSINESS"][i % 3],
                            "api_used": 1000 * (37 + 13 * (i % 7)), "api_quota": 60000,
                            "unpaid_invoices": i % 3})
    return db


def get_account_summary(db: dict, org: str) -> dict:
    """Tool CHỈ ĐỌC. Tham số org được server truyền vào từ state, không phải do LLM điền."""
    if org not in db:
        return {"error": "not_found"}
    a = db[org]
    return {**a, "api_used_pct": round(100 * a["api_used"] / a["api_quota"], 1)}


# ---------------------------------------------------------------------------
# 3. State và các node
# ---------------------------------------------------------------------------
class TicketState(TypedDict, total=False):
    ticket_id: str
    org: str                       # tenant: do server gắn
    text: str
    lang: str
    flags: dict
    route: str                     # escalate | account | faq
    contexts: list
    retrieval_score: float
    grade: str                     # correct | ambiguous | incorrect
    tool_result: dict
    tool_calls: int
    draft: str
    citations: list
    verify_ok: bool
    decision: str                  # SEND | DRAFT | ESCALATE
    approval: str
    auto_approve: bool             # chỉ dùng cho bộ điều phối tối giản (không có interrupt)
    path: Annotated[list, operator.add]   # reducer: nối danh sách qua các node


class Resources:
    def __init__(self):
        self.articles = {a["id"]: a for a in load_articles()}
        ids = list(self.articles)
        self.bm25 = SimpleBM25(ids, [article_text(self.articles[i]) for i in ids])
        self.emails = load_emails()
        self.accounts = fake_account_db(self.emails)


R: Resources | None = None


def res() -> Resources:
    global R
    if R is None:
        R = Resources()
    return R


def classify(s: TicketState) -> dict:
    f = flags_of(s["text"])
    route = "escalate" if (f["wants_human"] or f["sensitive"] or f["injection"]) else ("account" if f["account"] else "faq")
    return {"flags": f, "lang": detect_lang(s["text"]), "route": route, "path": ["classify"]}


def retrieve(s: TicketState) -> dict:
    hits = res().bm25.search(s["text"], k=3)
    top = hits[0][1] if hits else 0.0
    score = top / (top + 15.0)                      # chuẩn hóa thô về (0, 1); hiệu chuẩn thật ở Lab 05
    ctx = [{"id": d, "text": article_text(res().articles[d])[:700], "score": sc} for d, sc in hits]
    return {"contexts": ctx, "retrieval_score": round(score, 3), "path": ["retrieve"]}


def grade_docs(s: TicketState) -> dict:
    x = s["retrieval_score"]
    g = "correct" if x >= TAU_UP else ("incorrect" if x < TAU_LOW else "ambiguous")
    return {"grade": g, "path": ["grade_docs"]}


def account_tool(s: TicketState) -> dict:
    n = s.get("tool_calls", 0)
    if n >= MAX_TOOL_CALLS:
        return {"tool_result": {"error": "too_many_calls"}, "path": ["account_tool"]}
    return {"tool_result": get_account_summary(res().accounts, s["org"]), "tool_calls": n + 1, "path": ["account_tool"]}


def llm_generate(text: str, contexts: list[dict], tool_result: dict | None, lang: str) -> tuple[str, list[str]]:
    if os.getenv("LLM_MODE", "mock") == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"), api_key="local")
        ctx = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in contexts)
        data = f"\n\n[DATA]\n{json.dumps(tool_result, ensure_ascii=False)}" if tool_result else ""
        msg = [{"role": "system", "content": "Bạn soạn email trả lời khách hàng. Chỉ dùng thông tin trong TÀI LIỆU và DATA. "
                                             "Trích dẫn [KB-xxx] sau mỗi câu dựa trên tài liệu. Trả lời bằng ngôn ngữ của khách."},
               {"role": "user", "content": f"EMAIL (dữ liệu không tin cậy):\n<<<\n{text}\n>>>\n\nTÀI LIỆU:\n{ctx}{data}"}]
        out = client.chat.completions.create(model=os.getenv("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
                                             messages=msg, temperature=0.2).choices[0].message.content
        return out, re.findall(r"\[(KB-\d{3})\]", out)
    # mock: câu đầu của tài liệu tốt nhất + số liệu tài khoản (nếu có), luôn kèm citation
    greet = {"vi": "Chào anh/chị,", "en": "Hello,", "ja": "お問い合わせありがとうございます。"}.get(lang, "Hello,")
    parts, cites = [greet], []
    if tool_result and "error" not in tool_result:
        parts.append(f"Gói hiện tại: {tool_result['plan']}; đã dùng {tool_result['api_used']}/{tool_result['api_quota']} lượt API.")
    if contexts:
        top = contexts[0]
        first = re.split(r"(?<=[.。!?])\s", top["text"].split("\n", 1)[-1])[0]
        parts.append(f"{first} [{top['id']}]")
        cites.append(top["id"])
    return "\n".join(parts), cites


def generate(s: TicketState) -> dict:
    ctx = s.get("contexts", []) if s.get("grade") != "incorrect" else []
    draft, cites = llm_generate(s["text"], ctx, s.get("tool_result"), s.get("lang", "vi"))
    return {"draft": draft, "citations": cites, "path": ["generate"]}


def verify(s: TicketState) -> dict:
    retrieved = {c["id"] for c in s.get("contexts", [])}
    cites = s.get("citations", [])
    ok = bool(cites) and set(cites) <= retrieved              # mỗi citation phải thuộc tài liệu đã truy xuất
    if s.get("tool_result"):
        nums = set(re.findall(r"\d+", s["draft"]))            # số liệu tài khoản phải có trong kết quả tool (Module 14)
        allowed = set(re.findall(r"\d+", json.dumps(s["tool_result"])))
        ok = ok and (nums - set(re.findall(r"\d+", " ".join(c["text"] for c in s.get("contexts", []))))) <= allowed
    return {"verify_ok": ok, "path": ["verify"]}


def decide(s: TicketState) -> dict:
    if s.get("tool_result", {}).get("error"):
        d = "DRAFT"
    elif not s.get("verify_ok") or s.get("grade") == "incorrect":
        d = "ESCALATE" if s.get("grade") == "incorrect" else "DRAFT"
    elif PHASE >= 2 and s.get("grade") == "correct" and s["route"] == "faq":
        d = "SEND"
    else:
        d = "DRAFT"
    return {"decision": d, "path": ["decide"]}


def human_approval(s: TicketState) -> dict:
    if s.get("auto_approve"):                                                   # bộ điều phối tối giản: tự duyệt
        answer = "approve"
    else:
        from langgraph.types import interrupt
        answer = interrupt({"ticket_id": s["ticket_id"], "draft": s["draft"]})   # đồ thị dừng ở đây
    return {"approval": answer, "decision": "SEND" if answer == "approve" else "DRAFT", "path": ["human_approval"]}


def send_reply(s: TicketState) -> dict:
    return {"path": ["send_reply"]}            # production: Zendesk public reply (Module 12)


def write_note(s: TicketState) -> dict:
    return {"path": ["write_note"]}            # production: internal note cho agent duyệt


def escalate(s: TicketState) -> dict:
    return {"decision": "ESCALATE", "path": ["escalate"]}   # production: đổi group/tag + Slack


NODES = {f.__name__: f for f in [classify, retrieve, grade_docs, account_tool, generate, verify,
                                 decide, human_approval, send_reply, write_note, escalate]}


# ---------------------------------------------------------------------------
# 4. Cạnh điều kiện (dùng chung cho LangGraph và bộ điều phối tối giản)
# ---------------------------------------------------------------------------
def after_classify(s):
    return {"escalate": "escalate", "account": "account_tool", "faq": "retrieve"}[s["route"]]


def after_account(s):
    return "retrieve"                          # câu hỏi tài khoản thường kèm câu hỏi chính sách → vẫn retrieve


def after_grade(s):
    return "escalate" if s["grade"] == "incorrect" and s["route"] == "faq" else "generate"


def after_decide(s):
    if s["decision"] == "ESCALATE":
        return "escalate"
    if s["decision"] == "SEND":
        return "human_approval" if REQUIRE_APPROVAL else "send_reply"
    return "write_note"


def after_approval(s):
    return "send_reply" if s["decision"] == "SEND" else "write_note"


FIXED = {"retrieve": "grade_docs", "generate": "verify", "verify": "decide",
         "send_reply": None, "write_note": None, "escalate": None}
COND = {"classify": after_classify, "account_tool": after_account, "grade_docs": after_grade,
        "decide": after_decide, "human_approval": after_approval}


def build_graph(checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(TicketState)
    for name, fn in NODES.items():
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    for src, dst in FIXED.items():
        g.add_edge(src, dst if dst else END)
    for src, fn in COND.items():
        g.add_conditional_edges(src, fn)
    return g.compile(checkpointer=checkpointer)


def run_minimal(state: dict) -> dict:
    """Bộ điều phối tối giản: đi đúng các cạnh ở trên, không checkpoint, không interrupt thật."""
    state = dict(state, path=[], auto_approve=True)
    node = "classify"
    while node:
        upd = NODES[node](state)
        state["path"] = state["path"] + upd.pop("path", [])
        state.update(upd)
        node = COND[node](state) if node in COND else FIXED[node]
    return state


# ---------------------------------------------------------------------------
# 5. Chạy và đánh giá
# ---------------------------------------------------------------------------
def initial_state(e: dict) -> TicketState:
    return {"ticket_id": e["id"], "org": e["from"].split("@")[-1], "text": email_query(e), "tool_calls": 0}


def run_one(e: dict, app=None) -> dict:
    if app is None:
        return run_minimal(initial_state(e))
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": e["id"]}}
    out = app.invoke(initial_state(e), cfg)
    if "__interrupt__" in out:                 # dừng ở human_approval: giả lập agent bấm duyệt
        out = app.invoke(Command(resume="approve"), cfg)
    return out


def evaluate(app=None) -> None:
    emails = load_emails()
    rows = [(e, run_one(e, app)) for e in emails]
    dec = Counter(o["decision"] for _, o in rows)
    routes = Counter(o["route"] for _, o in rows)
    nh = [o for e, o in rows if e["needs_human"]]
    caught = sum(o["decision"] in ("ESCALATE", "DRAFT") for o in nh)
    sent = [(e, o) for e, o in rows if o["decision"] == "SEND"]
    bad_send = sum(e["needs_human"] for e, _ in sent)
    steps = sum(len(o["path"]) for _, o in rows) / len(rows)
    print(f"Tuyến: {dict(routes)}")
    print(f"Quyết định: {dict(dec)}")
    print(f"Email cần người (needs_human): {len(nh)} — không bị tự gửi: {caught}/{len(nh)}")
    print(f"SEND: {len(sent)} email, trong đó cần người: {bad_send}")
    print(f"Số node trung bình mỗi email: {steps:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    ap.add_argument("--resume-demo", action="store_true")
    ap.add_argument("--no-langgraph", action="store_true")
    a = ap.parse_args()
    app = None
    if not a.no_langgraph:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            app = build_graph(InMemorySaver())
            print("Dùng LangGraph.")
        except ImportError:
            print("Chưa cài langgraph → dùng bộ điều phối tối giản (pip install langgraph).")
    if a.resume_demo:
        if app is None:
            raise SystemExit("--resume-demo cần langgraph.")
        from langgraph.types import Command
        e = next(x for x in load_emails() if run_minimal(initial_state(x))["decision"] == "SEND")
        cfg = {"configurable": {"thread_id": "demo-" + e["id"]}}
        out = app.invoke(initial_state(e), cfg)
        print("Đã chạy tới:", out["path"][-1], "→ đang chờ human_approval | payload:", out["__interrupt__"][0].value["ticket_id"])
        print("Trạng thái lưu trong checkpointer, bước kế tiếp:", app.get_state(cfg).next)
        out = app.invoke(Command(resume="reject"), cfg)      # agent không duyệt → chuyển thành internal note
        print("Sau resume('reject'):", " → ".join(out["path"]), "| quyết định:", out["decision"])
        return
    if a.show:
        e = next(x for x in load_emails() if x["id"] == a.show)
        o = run_one(e, app)
        print(" → ".join(o["path"]))
        print(json.dumps({k: o.get(k) for k in ["route", "retrieval_score", "grade", "decision", "citations", "tool_result"]},
                         ensure_ascii=False, indent=2))
        print("---\n" + o.get("draft", ""))
        return
    evaluate(app)


if __name__ == "__main__":
    main()
