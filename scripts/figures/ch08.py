"""Hình minh họa cho Module 08 — Kiến trúc RAG nâng cao & Agentic RAG."""
import numpy as np
from matplotlib.patches import Circle, Ellipse, Patch, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


# ---------------------------------------------------------------- 1.1 vòng điều khiển
@figure("control-loop", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    ax.text(2, 43, "Naive RAG: đường ống tĩnh", fontsize=9.4, weight="bold", color=t["fg"])
    for i, s in enumerate(["query", "retrieve\n(1 lần)", "generate", "stop"]):
        box(ax, 2 + i * 13, 30, 11, 8, s, t, fs=8.2)
        if i < 3:
            arrow(ax, 13.3 + i * 13, 34, 14.7 + i * 13, 34, t)
    ax.text(2, 24, "luôn retrieve · một lần · top-k cố định", fontsize=8.2, color=t["fg2"])
    ax.plot([58, 58], [3, 45], color=t["grid"], lw=1)
    ax.text(62, 43, "RAG nâng cao: chính sách π(aₜ | sₜ)", fontsize=9.4, weight="bold", color=t["fg"])
    box(ax, 62, 18, 20, 14, "trạng thái sₜ\ncâu hỏi, tài liệu,\nbản nháp, lịch sử", t,
        color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14), fs=8)
    acts = [("retrieve(q, kho)", BLUE), ("call_tool(·)", ORANGE), ("generate", AQUA), ("escalate", RED), ("stop", GREEN)]
    for i, (a, c) in enumerate(acts):
        y = 36 - i * 7.5
        box(ax, 96, y, 26, 5.6, a, t, color=t["c"][c], fill=tint(t["c"][c], t, .14), fs=8.2)
        arrow(ax, 82.5, 25, 95.5, y + 2.8, t, lw=0.9, color=t["c"][c])
    ax.text(62, 13, "kết quả hành động → sₜ₊₁ → lặp", fontsize=8, color=t["fg2"])
    ax.text(62, 4, "π: classifier (Adaptive-RAG, CRAG),\ntoken đặc biệt (Self-RAG), LLM (agent)", fontsize=8, color=t["fg2"])


# ---------------------------------------------------------------- 1.2 chi phí định tuyến
@figure("routing-cost", size=(8.4, 2.8))
def _(fig, t):
    ax = fig.subplots()
    routes = [("FAQ/how-to", 0.55, 1.3, BLUE), ("nhiều câu hỏi", 0.25, 2.5, AQUA),
              ("điều tra tài khoản", 0.15, 5.0, ORANGE), ("nhạy cảm", 0.05, 0.5, RED)]
    left = 0
    for name, p, n, c in routes:
        v = p * n
        ax.barh(1, v, left=left, color=t["c"][c], height=0.5)
        if v > 0.1:
            ax.text(left + v / 2, 1, f"{p:.0%}×{n:g}\n= {v:.3f}".replace(".", ","), ha="center", va="center",
                    fontsize=7.8, color="#ffffff")
        left += v
    ax.text(left + 0.06, 1, f"E[n] ≈ {left:.1f}".replace(".", ","), va="center", fontsize=9.2, color=t["fg"], weight="bold")
    ax.barh(0, 5, color=t["muted"], height=0.5)
    ax.text(5.06, 0, "mọi ticket qua agent: 5 (≈ 2,4×)", va="center", fontsize=9, color=t["fg"])
    ax.set_yticks([0, 1], ["không định tuyến", "có định tuyến"])
    ax.set_xlim(0, 7.6); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("số lượt gọi LLM kỳ vọng mỗi lượt xử lý (giả định)")
    ax.legend(handles=[Patch(color=t["c"][c], label=n) for n, _, _, c in routes], fontsize=7.8, ncol=4,
              loc="upper center", bbox_to_anchor=(0.45, 1.3))
    xgrid(ax, t)


# ---------------------------------------------------------------- 2.1 Adaptive-RAG
@figure("adaptive-rag", size=(8.8, 3.0))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    ax.text(2, 37, "Tạo nhãn tự động: chạy cả ba chiến lược, nhãn = chiến lược đơn giản nhất trả lời đúng", fontsize=9,
            weight="bold", color=t["fg"])
    box(ax, 2, 14, 18, 12, "câu hỏi mẫu", t, fs=8.6)
    strat = [("A · không retrieve", "sai", RED), ("B · retrieve 1 bước", "đúng", GREEN), ("C · retrieve lặp", "đúng", GREEN)]
    for i, (s, r, c) in enumerate(strat):
        y = 26 - i * 9
        box(ax, 28, y, 28, 7, s, t, fs=8.4)
        arrow(ax, 20.3, 20, 27.7, y + 3.5, t, lw=1)
        box(ax, 60, y + 0.8, 10, 5.4, r, t, color=t["c"][c], fill=tint(t["c"][c], t, .16), fs=8.4)
    arrow(ax, 71, 20, 78, 20, t, lw=1.8)
    box(ax, 78.5, 15, 16, 10, "nhãn = B", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .16), fs=9.4, weight="bold")
    ax.text(97, 26, "classifier nhỏ học", fontsize=8.4, color=t["fg"])
    ax.text(97, 22, "«chi phí tối thiểu", fontsize=8.4, color=t["fg"])
    ax.text(97, 18, "để đúng»", fontsize=8.4, color=t["fg"])
    ax.text(2, 3, "Zendesk: ack · faq · multi_doc · account · sensitive — không dùng lớp A cho câu hỏi về sản phẩm.",
            fontsize=8.3, color=t["fg2"])


# ---------------------------------------------------------------- 2.2 FLARE
@figure("flare-tokens", size=(7.8, 3.0))
def _(fig, t):
    ax = fig.subplots()
    toks = ["Nhập", "Entity", "ID", "Metadata", "URL", "Okta"]
    p = np.array([0.92, 0.81, 0.95, 0.44, 0.88, 0.31])
    cols = [t["c"][RED] if v < 0.4 else (t["c"][YELLOW] if v < 0.5 else t["c"][BLUE]) for v in p]
    x = np.arange(len(toks))
    ax.bar(x, p, color=cols, width=0.6)
    ax.axhline(0.5, color=t["c"][ORANGE], ls="--", lw=1.3)
    ax.axhline(0.4, color=t["c"][RED], ls=":", lw=1.3)
    ax.text(5.5, 0.5, "θ = 0.5\nmin < θ → retrieve", fontsize=8.2, color=t["fg"], ha="left", va="bottom")
    ax.text(5.5, 0.39, "β = 0.4\ntoken dưới β bị\nche khỏi query", fontsize=8.2, color=t["fg"], ha="left", va="top")
    for xi, v in zip(x, p):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.4, color=t["fg"])
    ax.set_xticks(x, toks); ax.set_ylim(0, 1.08); ax.set_xlim(-0.5, 7.3)
    ax.set_ylabel("p(token) trong câu tạm")
    ax.set_title("Query sinh ra: «Nhập Entity ID và Metadata URL do ___ cung cấp»", fontsize=9.6)
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.1 Self-RAG
@figure("selfrag-scores", size=(8.4, 2.9))
def _(fig, t):
    ax = fig.subplots()
    rows = [("d₁ So sánh gói\nw_Sup = 1", 0.9, 0.8, 0.575, 1), ("d₂ ticket cũ\nw_Sup = 1", 0.6, 0.35, 0.2, 1),
            ("d₁\nw_Sup = 2", 0.9, 0.8, 0.575, 2), ("d₂\nw_Sup = 2", 0.6, 0.35, 0.2, 2)]
    y = np.arange(len(rows))
    for i, (name, rel, sup, use, ws) in enumerate(rows):
        parts = [(rel, BLUE), (ws * sup, GREEN), (0.5 * use, VIOLET)]
        left = 0
        for v, c in parts:
            ax.barh(i, v, left=left, color=t["c"][c], height=0.55)
            left += v
        ax.text(left + 0.04, i, f"S = {left:.2f}", va="center", fontsize=8.8, color=t["fg"])
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.2); ax.invert_yaxis()
    ax.set_xlim(0, 3.5)
    ax.axhline(1.5, color=t["line"], lw=1)
    ax.set_xlabel("điểm phê bình S = w_Rel·s_Rel + w_Sup·s_Sup + w_Use·s_Use (w_Use = 0.5)")
    ax.legend(handles=[Patch(color=t["c"][BLUE], label="IsRel"), Patch(color=t["c"][GREEN], label="IsSup"),
                       Patch(color=t["c"][VIOLET], label="IsUse")], fontsize=8, loc="lower right")
    ax.set_title("Self-RAG: tăng w_Sup làm tính có căn cứ chi phối", fontsize=9.6)
    xgrid(ax, t)


# ---------------------------------------------------------------- 3.2 CRAG
@figure("crag-thresholds", size=(8.8, 3.1))
def _(fig, t):
    ax = fig.subplots()
    lo, up = -0.9, 0.5
    ax.axvspan(-1, lo, color=t["c"][RED], alpha=0.12, lw=0)
    ax.axvspan(lo, up, color=t["c"][YELLOW], alpha=0.12, lw=0)
    ax.axvspan(up, 1, color=t["c"][GREEN], alpha=0.12, lw=0)
    ax.axvline(lo, color=t["c"][RED], lw=1.2, ls="--"); ax.axvline(up, color=t["c"][GREEN], lw=1.2, ls="--")
    for y, e, lab in [(1, [0.72, 0.10, -0.40], "ví dụ 1 → Correct (có e ≥ τ_up)"),
                      (0, [0.2, -0.3, -0.5], "ví dụ 2 → Ambiguous")]:
        ax.scatter(e, [y] * 3, s=60, color=t["c"][BLUE], zorder=4)
        for v in e:
            ax.text(v, y + 0.18, f"{v:+.2f}", ha="center", fontsize=8.2, color=t["fg"])
        ax.text(-0.98, y - 0.28, lab, fontsize=8.4, color=t["fg"])
    ax.text((-1 + lo) / 2, 1.75, "Incorrect\n→ escalate\n(không web)", ha="center", fontsize=8, color=t["fg"], va="top")
    ax.text((lo + up) / 2, 1.75, "Ambiguous → mở rộng nội bộ,\nvẫn mơ hồ thì draft một phần + escalate", ha="center",
            fontsize=8, color=t["fg"], va="top")
    ax.text((up + 1) / 2, 1.75, "Correct → tinh lọc\nstrip, generate", ha="center", fontsize=8, color=t["fg"], va="top")
    ax.set_xlim(-1, 1); ax.set_ylim(-0.55, 1.85); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("điểm evaluator eᵢ của từng tài liệu (τ_low = −0.9, τ_up = 0.5)")
    ax.set_title("CRAG cho CS: ba vùng hành động theo hai ngưỡng", fontsize=9.6)


# ---------------------------------------------------------------- 4 multi-hop / IRCoT
@figure("multihop", size=(8.8, 3.2))
def _(fig, t):
    a0 = fig.add_axes([0.0, 0.0, 0.58, 1.0]); a1 = fig.add_axes([0.68, 0.16, 0.31, 0.68])
    a0.set_xlim(0, 70); a0.set_ylim(0, 40); a0.axis("off")
    a0.text(1, 37, "IRCoT / ReAct: mỗi bước suy luận làm query cho bước sau", fontsize=9, weight="bold", color=t["fg"])
    steps = [("Thought 1: cần release notes tháng 9", BLUE), ("search → [R12] v5.8 đổi định dạng ngày", AQUA),
             ("Thought 2: cần workaround cho v5.8", BLUE), ("search → [H331] chọn lại dd/mm/yyyy", AQUA),
             ("Đủ → soạn trả lời [R12][H331]", GREEN)]
    for i, (s, c) in enumerate(steps):
        y = 30 - i * 6.6
        box(a0, 2 + (i % 2) * 6, y, 52, 5, s, t, color=t["c"][c], fill=tint(t["c"][c], t, .13), fs=8, ha="left", radius=0.6)
        if i < 4:
            arrow(a0, 28 + (i % 2) * 6, y - 0.2, 28 + ((i + 1) % 2) * 6, y - 1.4, t, lw=1)
    a0.text(2, 1.5, "giới hạn ≤ 4 bước, ngân sách token, tool allowlist", fontsize=8, color=t["fg2"])
    vals = [0.3, 0.8 * 0.7]
    b = a1.bar(["1 lần\nretrieve", "2 bước"], vals, color=[t["c"][ORANGE], t["c"][GREEN]], width=0.55)
    for bb, v in zip(b, vals):
        a1.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{'≤ ' if v == 0.3 else ''}{v:.2f}", ha="center", fontsize=9, color=t["fg"])
    a1.set_ylim(0, 1); a1.set_ylabel("P(có đủ d⁽¹⁾ và d⁽²⁾)")
    a1.set_title("Recall chuỗi (minh họa)", fontsize=9.4)
    ygrid(a1, t)


# ---------------------------------------------------------------- 5.2 RAPTOR
@figure("raptor-tree", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    leaves = [6 + i * 9 for i in range(8)]
    for x in leaves:
        box(ax, x - 3.5, 4, 7, 5, "chunk", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .15), fs=7.4, radius=0.5)
    mids = [(14, [0, 1, 2]), (37, [2, 3, 4]), (64, [5, 6, 7])]
    for mx, ch in mids:
        box(ax, mx - 6, 18, 12, 6, "tóm tắt", t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .15), fs=7.8, radius=0.6)
        for c in ch:
            ax.plot([leaves[c], mx], [9, 18], color=t["line"], lw=1)
    ax.text(30.5, 12.5, "chunk 3 thuộc\nhai cụm (GMM mềm)", fontsize=7.4, color=t["fg2"], ha="center")
    box(ax, 33, 33, 12, 6, "tóm tắt\ngốc", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=7.8, radius=0.6)
    for mx, _ in mids:
        ax.plot([mx, 39], [24, 33], color=t["line"], lw=1)
    ax.text(2, 43, "Xây cây từ dưới lên: phân cụm → LLM tóm tắt → embed lại", fontsize=9, weight="bold", color=t["fg"])
    ax.plot([80, 80], [2, 44], color=t["grid"], lw=1)
    ax.text(84, 43, "Collapsed tree khi truy vấn", fontsize=9, weight="bold", color=t["fg"])
    items = [("tóm tắt gốc", VIOLET), ("tóm tắt × 3", AQUA), ("chunk × 8", BLUE)]
    for i, (s, c) in enumerate(items):
        box(ax, 84, 30 - i * 8, 36, 6, s, t, color=t["c"][c], fill=tint(t["c"][c], t, .15), fs=8.2)
    ax.text(84, 6.5, "trải phẳng mọi node vào một index;\ncâu hỏi chi tiết trúng lá,\ncâu hỏi tổng hợp trúng tóm tắt", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 5.3 GMM + BIC
@figure("gmm-bic", size=(8.8, 3.2))
def _(fig, t):
    from sklearn.mixture import GaussianMixture
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.1, 1]))
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal([0, 0], 0.6, (60, 2)), rng.normal([3, 0.5], 0.6, (60, 2)), rng.normal([1.5, 2.8], 0.5, (50, 2))]
    gm = GaussianMixture(3, random_state=0).fit(X)
    R = gm.predict_proba(X)
    cols = np.array([t["c"][BLUE], t["c"][ORANGE], t["c"][AQUA]])
    lab = R.argmax(1)
    soft = (R > 0.2).sum(1) > 1
    a1.scatter(X[~soft, 0], X[~soft, 1], s=12, color=cols[lab[~soft]], lw=0, alpha=0.85)
    a1.scatter(X[soft, 0], X[soft, 1], s=40, facecolor="none", edgecolor=t["fg"], lw=1.3, label="γᵢₖ > 0.2 ở ≥ 2 cụm")
    for k in range(3):
        v, w = np.linalg.eigh(gm.covariances_[k])
        ang = np.degrees(np.arctan2(w[1, 1], w[0, 1]))
        a1.add_patch(Ellipse(gm.means_[k], 4 * np.sqrt(v[1]), 4 * np.sqrt(v[0]), angle=ang, fill=False, ec=cols[k], lw=1.4))
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("GMM (2D mô phỏng): phân cụm mềm", fontsize=9.4)
    a1.legend(fontsize=8, loc="upper right")
    K = [3, 5, 8]; fit = [4100, 3300, 2900]; pen = [(k * 65 + k - 1) * 5.30 for k in K]  # ln 200 ≈ 5,30 như trong bài
    x = np.arange(3)
    a2.bar(x, fit, color=t["c"][BLUE], width=0.55, label="−2 ln L̂ (giả định)")
    a2.bar(x, pen, bottom=fit, color=t["c"][ORANGE], width=0.55, label="phạt p·ln n")
    for i in range(3):
        tot = fit[i] + pen[i]
        a2.text(i, tot + 80, f"{tot:,.0f}".replace(",", "."), ha="center", fontsize=8.6, color=t["fg"],
                weight="bold" if i == 1 else "normal")
    a2.set_xticks(x, [f"K = {k}" for k in K]); a2.set_ylim(0, 6800)
    a2.set_title("BIC nhỏ nhất ở K = 5 (n = 200, d′ = 10)", fontsize=9.4)
    a2.legend(fontsize=8, loc="upper left")
    ygrid(a2, t)


# ---------------------------------------------------------------- 6.3 modularity
@figure("modularity", size=(8.4, 3.1))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 42))
    pos = {"SSO": (10, 30), "SAML": (26, 20), "Okta": (10, 10), "Thanh toán": (52, 20), "Hóa đơn": (68, 30), "VAT": (68, 10)}
    E = [("SSO", "SAML"), ("SAML", "Okta"), ("SSO", "Okta"), ("Hóa đơn", "VAT"), ("VAT", "Thanh toán"),
         ("Hóa đơn", "Thanh toán"), ("SAML", "Thanh toán")]
    comm = {"SSO": 0, "SAML": 0, "Okta": 0, "Thanh toán": 1, "Hóa đơn": 1, "VAT": 1}
    for a, b in E:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        bridge = comm[a] != comm[b]
        ax.plot([x1, x2], [y1, y2], color=t["c"][RED] if bridge else t["line"], lw=2 if bridge else 1.5, ls="--" if bridge else "-")
    for n, (x, y) in pos.items():
        c = t["c"][BLUE] if comm[n] == 0 else t["c"][ORANGE]
        ax.add_patch(Circle((x, y), 5.2, color=tint(c, t, .25), ec=c, lw=1.6, zorder=3))
        ax.text(x, y, n, ha="center", va="center", fontsize=7.8, color=t["fg"], zorder=4)
    ax.text(39, 23, "cạnh nối", fontsize=7.6, color=t["c"][RED], ha="center")
    ax.text(84, 33, "m = 7 cạnh, γ = 1", fontsize=8.8, color=t["fg"])
    ax.text(84, 27, "hai cộng đồng: Lc = 3, Kc = 7", fontsize=8.8, color=t["fg"])
    ax.text(84, 21, "Q = 2·[3/7 − (7/14)²] ≈ 0.357", fontsize=9, color=t["fg"], weight="bold")
    ax.text(84, 15, "gộp tất cả: Q = 1 − 1² = 0", fontsize=8.8, color=t["fg2"])
    ax.text(2, 1, "Leiden tìm phân hoạch tối đa hóa Q; đệ quy cho phân cấp cộng đồng của GraphRAG.", fontsize=8.3, color=t["fg2"])


# ---------------------------------------------------------------- 6.5 Personalized PageRank
@figure("ppr", size=(8.4, 3.1))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 42))
    names = ["SSO", "SAML", "Okta", "Azure AD", "Thanh toán", "Hóa đơn", "VAT", "Enterprise"]
    pos = np.array([(12, 30), (30, 22), (12, 12), (32, 36), (56, 22), (74, 32), (74, 12), (52, 36)])
    E = [(0, 1), (1, 2), (0, 2), (1, 3), (0, 3), (1, 4), (4, 5), (5, 6), (4, 6), (7, 0), (7, 4)]
    n = len(names); A = np.zeros((n, n))
    for a, b in E:
        A[a, b] = A[b, a] = 1
    P = A / A.sum(1, keepdims=True)
    alpha = 0.85; e = np.zeros(n); e[0] = 1
    r = e.copy()
    for _ in range(200):
        r = (1 - alpha) * e + alpha * P.T @ r
    for a, b in E:
        ax.plot(*zip(pos[a], pos[b]), color=t["line"], lw=1.2)
    for i in range(n):
        rad = 2.5 + 16 * r[i]
        ax.add_patch(Circle(pos[i], rad, color=tint(t["c"][VIOLET], t, 0.2 + 0.7 * r[i] / r.max()), ec=t["c"][VIOLET], lw=1.4, zorder=3))
        ax.text(pos[i][0], pos[i][1] - rad - 2.2, f"{names[i]} {r[i]:.2f}", ha="center", fontsize=7.6, color=t["fg"], zorder=4)
    ax.text(88, 33, "Seed = «SSO», α = 0.85", fontsize=8.8, color=t["fg"], weight="bold")
    ax.text(88, 27, "r = (1−α)·e_seed + α·Pᵀr", fontsize=8.6, color=t["fg"])
    ax.text(88, 19, "kích thước node ∝ r;", fontsize=8.4, color=t["fg2"])
    ax.text(88, 15, "node gần seed theo nhiều", fontsize=8.4, color=t["fg2"])
    ax.text(88, 11, "đường được điểm cao", fontsize=8.4, color=t["fg2"])
    ax.text(2, 1, "HippoRAG: multi-hop trong một bước retrieve (đồ thị đồ chơi, r tính thật bằng lặp lũy thừa).",
            fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 7.1 Self-Route
@figure("selfroute-cost", size=(6.6, 3.0))
def _(fig, t):
    ax = fig.subplots()
    rho = np.linspace(0, 1, 200)
    ax.plot(rho, 1 + 20 * rho, color=t["c"][BLUE], label="Self-Route: c_RAG + ρ·c_LC")
    ax.axhline(20, color=t["c"][ORANGE], ls="--", lw=1.4, label="long-context cho mọi câu")
    ax.axhline(1, color=t["c"][AQUA], ls=":", lw=1.4, label="chỉ RAG")
    ax.scatter([0.15], [4], color=t["fg"], s=30, zorder=4)
    ax.text(0.17, 3.2, "ρ = 0.15 → 4·c_RAG (rẻ hơn LC 5×)", fontsize=8.4, color=t["fg"])
    ax.set_xlabel("ρ — tỷ lệ câu hỏi bị chuyển sang long-context")
    ax.set_ylabel("chi phí kỳ vọng (đơn vị c_RAG)")
    ax.set_ylim(0, 22)
    ax.set_title("c_LC = 20·c_RAG (giả định)", fontsize=9.6)
    ax.legend(fontsize=8, loc="lower right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 7.2 KV cache cho CAG
@figure("cag-kv-memory", size=(8.0, 2.9))
def _(fig, t):
    ax = fig.subplots()
    per = 2 * 32 * 8 * 128 * 2  # byte/token
    rows = [("Chính sách ~30K token", 30_000), ("300 macro ~75K token", 75_000), ("800 bài HC ~960K token", 960_000)]
    y = np.arange(3)
    gib = [n * per / 2 ** 30 for _, n in rows]
    ax.barh(y, gib, color=[t["c"][GREEN], t["c"][AQUA], t["c"][RED]], height=0.55)
    for i, g in enumerate(gib):
        ax.text(g * 1.1, i, f"{g:.1f} GiB", va="center", fontsize=8.8, color=t["fg"])
    for v, lab in [(6, "RTX 4050: 6 GB (cả model)"), (80, "GPU datacenter 80 GB")]:
        ax.axvline(v, color=t["muted"], ls="--", lw=1)
        ax.text(v * 1.05, -0.62, lab, fontsize=7.8, color=t["fg2"])
    ax.set_xscale("log"); ax.set_xlim(1, 400)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_ylim(2.5, -0.85)
    ax.set_xlabel("KV cache (log) — model 8B GQA: 32 lớp, 8 head KV, d_h = 128, FP16 → 128 KiB/token")
    ax.set_title("CAG chỉ khả thi cho kho nhỏ, ổn định", fontsize=9.6)
    xgrid(ax, t)


# ---------------------------------------------------------------- 8.1 dải mức tự chủ
@figure("agent-spectrum", size=(8.8, 2.8))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 36))
    levels = [("Pipeline\ncố định", "naive RAG"), ("Router", "Adaptive-RAG"), ("Vòng lặp\ncó giới hạn", "CRAG, repair"),
              ("Agent có tool\ntrong đồ thị", "ReAct trong\nnode LangGraph"), ("Agent\ntự do", "autonomous")]
    for i, (a, b) in enumerate(levels):
        x = 2 + i * 24.4
        rec = i == 3
        c = t["c"][GREEN] if rec else t["c"][BLUE]
        k = 0.1 + 0.08 * i
        box(ax, x, 13, 22, 14, a, t, color=c, fill=tint(t["c"][BLUE], t, k) if not rec else tint(c, t, .22), fs=8.6,
            weight="bold" if rec else "normal", lw=2.4 if rec else 1.2)
        ax.text(x + 11, 9, b, ha="center", fontsize=7.8, color=t["fg2"], va="top")
    ax.annotate("", xy=(121, 31), xytext=(2, 31), arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1.3))
    ax.text(2, 32.5, "LLM quyết định càng nhiều →", fontsize=8.3, color=t["fg2"])
    ax.text(121, 32.5, "← kiểm soát càng ít", fontsize=8.3, color=t["fg2"], ha="right")
    ax.text(86.2, 0.5, "khuyến nghị cho CS production", fontsize=8.2, color=t["c"][GREEN] if t["name"] == "light" else t["fg"], ha="center")


# ---------------------------------------------------------------- 8.2 LangGraph state machine
@figure("langgraph-state", size=(8.6, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 42))
    box(ax, 2, 13, 22, 12, "state s\n(TypedDict)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=8.6)
    box(ax, 34, 13, 20, 12, "node n\nfₙ(s) → cập nhật", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .15), fs=8.4)
    box(ax, 64, 13, 20, 12, "s ⊕ fₙ(s)\n(reducer)", t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .15), fs=8.4)
    box(ax, 94, 13, 26, 12, "cạnh điều kiện\ng(s) → node tiếp", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .15), fs=8.4)
    for x1, x2 in [(24.3, 33.7), (54.3, 63.7), (84.3, 93.7)]:
        arrow(ax, x1, 19, x2, 19, t, lw=1.6)
    ax.annotate("", xy=(13, 25.5), xytext=(107, 25.5), arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1.2,
                connectionstyle="arc3,rad=0.25"))
    ax.text(60, 39.5, "chu trình được phép (rewrite, repair) — có bộ đếm trong state", fontsize=8.2, color=t["fg2"], ha="center")
    ax.text(2, 5, "Checkpointer lưu s sau mỗi bước theo thread_id = «zd-<ticket_id>» → chạy tiếp sau crash, interrupt() chờ người duyệt.",
            fontsize=8, color=t["fg2"])
    ax.text(2, 1, "Tập hành động khả dĩ bị giới hạn bởi đồ thị, không phải bởi điều LLM «muốn».", fontsize=8.2, color=t["fg"])


# ---------------------------------------------------------------- 8.5 MCP + tenant
@figure("mcp-tenant", size=(8.8, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    box(ax, 2, 26, 26, 12, "load_context\norg_id từ Zendesk API", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .14), fs=8.2)
    box(ax, 2, 6, 26, 12, "LLM agent (ReAct)\nchọn tool + tham số", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14), fs=8.2)
    box(ax, 42, 6, 26, 32, "", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .08))
    ax.text(44, 34, "MCP adapter", fontsize=8.8, weight="bold", color=t["fg"])
    ax.text(44, 29, "• allowlist tool chỉ đọc", fontsize=8, color=t["fg"])
    ax.text(44, 25, "• inject org_id từ state", fontsize=8, color=t["fg"])
    ax.text(44, 21, "• LLM không điền org_id", fontsize=8, color=t["fg"])
    ax.text(44, 17, "• spotlight kết quả", fontsize=8, color=t["fg"])
    ax.text(44, 13, "  trước khi trả LLM", fontsize=8, color=t["fg"])
    arrow(ax, 28.5, 32, 41.5, 26, t, color=t["c"][GREEN], lw=1.6)
    arrow(ax, 28.5, 12, 41.5, 16, t, color=t["c"][VIOLET], lw=1.6)
    tools = ["get_account_status", "get_recent_invoices (≤5)", "get_error_logs (≤7 ngày)", "search_kb"]
    for i, s in enumerate(tools):
        box(ax, 82, 32 - i * 7, 38, 5.5, s, t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .12), fs=8, radius=0.6)
        arrow(ax, 68.5, 22, 81.5, 34.7 - i * 7, t, lw=0.9)
    ax.text(82, 40, "MCP server chỉ đọc (scope theo org)", fontsize=8.4, color=t["fg2"])
    box(ax, 82, 2, 38, 5, "email: «kiểm tra org 4711 giúp tôi» → bị bỏ qua", t, color=t["c"][RED],
        fill=tint(t["c"][RED], t, .1), fs=7.6, radius=0.6)


# ---------------------------------------------------------------- 9.2 kiến trúc theo loại ticket
@figure("architecture-by-route", size=(8.4, 2.8))
def _(fig, t):
    ax = fig.subplots()
    rows = [("FAQ / how-to", 55, "hybrid + rerank + CRAG-lite", BLUE), ("Nhiều câu hỏi", 25, "decomposition song song (+ RAPTOR nếu đo thấy lợi)", AQUA),
            ("Điều tra tài khoản", 15, "LangGraph + ReAct ≤ 4 bước, tool chỉ đọc", ORANGE), ("Nhạy cảm", 5, "rule → escalate ngay", RED)]
    y = np.arange(len(rows))
    for i, (n, p, a, c) in enumerate(rows):
        ax.barh(i, p, color=t["c"][c], height=0.55)
        ax.text(p + 1, i, f"{p}% · {a}", va="center", fontsize=8.3, color=t["fg"])
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 130); ax.set_xlabel("tỷ lệ ticket (giả định)")
    ax.set_title("Phức tạp hóa có chọn lọc: chỉ 15% ticket chạm tới agent", fontsize=9.6)
    xgrid(ax, t)


if __name__ == "__main__":
    run("08")
