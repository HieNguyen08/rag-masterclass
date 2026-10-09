"""Hình minh họa cho Module 07 — Generation, grounding, citation, guardrails & prompt injection."""
import numpy as np
from matplotlib.patches import Ellipse, Patch, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


# ---------------------------------------------------------------- 1.1 tập đầu ra hợp lệ
@figure("valid-output-set", size=(8.6, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    ax.add_patch(Rectangle((2, 2), 70, 40, fill=False, ec=t["line"], lw=1.2))
    ax.text(4, 38.5, "mọi đầu ra y có thể sinh", fontsize=8.6, color=t["fg2"])
    cons = [("faithfulness", BLUE, (30, 22), (40, 26)), ("format JSON", AQUA, (40, 22), (40, 26)),
            ("policy", ORANGE, (36, 16), (40, 22)), ("security", VIOLET, (34, 26), (40, 22)),
            ("style / ngôn ngữ", MAGENTA, (40, 18), (38, 26))]
    for name, c, (x, y), (w, h) in cons:
        ax.add_patch(Ellipse((x, y), w, h, fill=False, ec=t["c"][c], lw=1.6))
    ax.add_patch(Ellipse((36.5, 21.5), 9, 7, color=t["c"][GREEN], alpha=0.35, lw=0))
    ax.text(36.5, 21.5, "hợp lệ", ha="center", va="center", fontsize=8.6, weight="bold", color=t["fg"])
    labels = [("faithfulness", BLUE, (3, 30)), ("format", AQUA, (58, 31)), ("policy", ORANGE, (55, 7)),
              ("security", VIOLET, (38, 38.4)), ("style", MAGENTA, (60, 19))]
    for name, c, (x, y) in labels:
        ax.text(x, y, name, fontsize=8.4, color=t["c"][c], weight="bold")
    ax.text(78, 33, "Generation tốt =", fontsize=9.4, weight="bold", color=t["fg"])
    ax.text(78, 28.5, "tối đa hóa chất lượng trên", fontsize=8.8, color=t["fg"])
    ax.text(78, 24.5, "giao của mọi ràng buộc", fontsize=8.8, color=t["fg"])
    box(ax, 78, 8, 42, 10, "Giao rỗng (thiếu context,\nrủi ro cao) → KHÔNG sinh, escalate", t,
        color=t["c"][RED], fill=tint(t["c"][RED], t, .14), fs=8.6)


# ---------------------------------------------------------------- 2.1 giải phẫu prompt
@figure("prompt-anatomy", size=(8.6, 3.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 50))
    blocks = [("system", "Vai trò · chính sách bất biến · grounding & citation · schema", "tin cậy", GREEN, 9),
              ("user", "Metadata ticket (ngôn ngữ, gói, tên) — lấy từ API", "tin cậy", GREEN, 6),
              ("user", "<sources> S1, S2, … có ID + type/authority/updated_at", "bán tin cậy", YELLOW, 9),
              ("user", "<customer_email> đã làm sạch, escape, datamark", "KHÔNG tin cậy", RED, 9),
              ("user", "Nhắc lại nhiệm vụ (2–3 dòng, gần vị trí sinh)", "tin cậy", GREEN, 5)]
    y = 47
    for role, s, trust, c, h in blocks:
        y -= h + 1.2
        col = t["c"][c]
        box(ax, 14, y, 80, h, s, t, color=col, fill=tint(col, t, .13), fs=8.5, ha="left")
        ax.text(12, y + h / 2, role, ha="right", va="center", fontsize=8.4, color=t["fg2"], family="monospace")
        ax.text(97, y + h / 2, trust, va="center", fontsize=8.6, color=t["fg"], weight="bold" if c == RED else "normal")
    ax.text(14, 48, "đầu prompt (phần cố định → hưởng prefix caching)", fontsize=8.2, color=t["fg2"])
    ax.text(14, 0.5, "cuối prompt — vị trí sinh", fontsize=8.2, color=t["fg2"])
    ax.text(97, 48, "lệnh chỉ đến từ khối xanh", fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 2.3 sắp xếp theo hình chữ U
@figure("context-order-u", size=(8.8, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32, width_ratios=[1.3, 1]))
    a = np.array([0.80, 0.65, 0.55, 0.62, 0.75]); r = np.array([0.50, 0.25, 0.12, 0.08, 0.05])
    pos = np.arange(1, 6)
    a1.plot(pos, a, color=t["c"][VIOLET], marker="o", lw=2, label="a(j): khả năng «dùng được» vị trí j")
    desc = [1, 2, 3, 4, 5]; sand = [1, 3, 5, 4, 2]  # hạng đặt tại vị trí j
    for j in range(5):
        a1.text(pos[j] - 0.12, a[j] + 0.035, f"#{desc[j]}", fontsize=8.4, color=t["c"][BLUE], ha="center")
        a1.text(pos[j] + 0.14, a[j] + 0.035, f"#{sand[j]}", fontsize=8.4, color=t["c"][ORANGE], ha="center")
    a1.set_ylim(0.45, 0.92); a1.set_xticks(pos); a1.set_xlabel("vị trí trong context j")
    a1.set_title("Hạng chunk đặt ở mỗi vị trí (số giả định)", fontsize=9.6)
    a1.legend(handles=[Patch(color=t["c"][VIOLET], label="a(j) hình chữ U"), Patch(color=t["c"][BLUE], label="giảm dần"),
                       Patch(color=t["c"][ORANGE], label="sandwich")], fontsize=8, loc="lower center", ncol=3)
    ygrid(a1, t)
    v1 = float(np.sum(r * a)); idx = [0, 4, 1, 3, 2]
    v2 = float(sum(r[i] * a[idx[i]] for i in range(5)))
    b = a2.bar(["giảm dần", "sandwich"], [v1, v2], color=[t["c"][BLUE], t["c"][ORANGE]], width=0.55)
    for bb, v in zip(b, [v1, v2]):
        a2.text(bb.get_x() + bb.get_width() / 2, v + 0.015, f"{v:.4f}", ha="center", fontsize=9, color=t["fg"])
    a2.set_ylim(0, 0.88); a2.set_ylabel("P(đúng) ≈ Σ rᵢ·a(σ(i))")
    a2.set_title("+~3 điểm, miễn phí", fontsize=9.6)
    ygrid(a2, t)


# ---------------------------------------------------------------- 2.4 ngân sách token
@figure("token-budget", size=(8.2, 1.8))
def _(fig, t):
    ax = fig.subplots()
    parts = [("system + schema", 1500, VIOLET), ("email", 400, RED), ("6 chunk × 350", 2100, BLUE), ("JSON ra", 600, AQUA)]
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=t["c"][c], height=0.5)
        ax.text(left + v / 2, 0, f"{name}\n{v:,}".replace(",", "."), ha="center", va="center", fontsize=8.2,
                color="#ffffff")
        left += v
    ax.text(left + 60, 0, f"≈ {left:,} token/lượt".replace(",", "."), va="center", fontsize=9, color=t["fg"], weight="bold")
    ax.set_xlim(0, 6000); ax.set_yticks([]); ax.set_ylim(-0.45, 0.45)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("token (ước lượng) — × ~5.250 lượt/ngày ≈ 24 triệu token/ngày")
    ax.set_title("Ngân sách token một lượt sinh")


# ---------------------------------------------------------------- 3.2 ALCE
@figure("alce-citation", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    rows = [("s₁ Business hoàn tiền nếu hủy trong 14 ngày", [("S1", "ok")], True, ""),
            ("s₂ Xuất hóa đơn tại Cài đặt > Thanh toán > Hóa đơn", [("S2", "ok"), ("S3", "irr")], True, "S3 thừa → trừ precision"),
            ("s₃ Hệ thống chưa hỗ trợ xuất hóa đơn gộp", [("S3", "ok")], True, "được hỗ trợ — nhưng S3 có thể lỗi thời"),
            ("s₄ Chúng tôi sẽ hoàn tiền trong 3 ngày làm việc", [("S1", "no")], False, "cam kết không có trong nguồn")]
    for i, (sent, cites, sup, cm) in enumerate(rows):
        y = 36 - i * 9
        c = t["c"][GREEN] if sup else t["c"][RED]
        box(ax, 2, y, 58, 7, sent, t, color=c, fill=tint(c, t, .1), fs=8.1, ha="left")
        for j, (sid, st) in enumerate(cites):
            col = {"ok": t["c"][GREEN], "irr": t["c"][YELLOW], "no": t["c"][RED]}[st]
            box(ax, 62 + j * 9, y + 1, 8, 5, sid, t, color=col, fill=tint(col, t, .2), fs=8.6, radius=0.6)
        ax.text(82, y + 3.5, cm, fontsize=8, color=t["fg2"], va="center")
    ax.text(2, 44, "Câu (xanh: nguồn trích đủ hỗ trợ)", fontsize=8.5, color=t["fg2"])
    ax.text(62, 44, "citation", fontsize=8.5, color=t["fg2"])
    ax.text(2, 1.2, "CitRec = 3/4 = 0.75   ·   CitPrec = 3/4 = 0.75 (tính trên 4 citation của s₁–s₃)", fontsize=9,
            color=t["fg"], weight="bold")


# ---------------------------------------------------------------- 3.3 ba tầng kiểm citation
@figure("citation-check-tiers", size=(8.6, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 34))
    tiers = [("1. Cú pháp", "[S#] thuộc tập ID đã đưa vào\nJSON citations khớp văn bản", "~0", AQUA),
             ("2. Từ vựng / số liệu", "số, ngày, %, tiền, tên gói,\nmenu path phải có trong nguồn", "rẻ", BLUE),
             ("3. Ngữ nghĩa", "NLI / LLM-judge cho từng\ncặp (câu, nguồn) — mục 8", "đắt", VIOLET)]
    for i, (name, d, cost, c) in enumerate(tiers):
        x = 2 + i * 41
        box(ax, x, 6, 37, 22, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .12))
        ax.text(x + 2, 24, name, fontsize=9.4, weight="bold", color=t["fg"])
        ax.text(x + 2, 15.5, d, fontsize=8.2, color=t["fg"], va="center")
        ax.text(x + 2, 8.5, f"chi phí: {cost}", fontsize=8.2, color=t["fg2"])
        if i < 2:
            arrow(ax, x + 37.5, 17, x + 40.5, 17, t, lw=1.6)
    ax.text(2, 1, "Ví dụ: «hủy trong 30 ngày [S1]» — S1 chỉ ghi 14 ngày → tầng 2 bắt được, gần như không tốn gì.",
            fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 4.2 abstention
@figure("abstention-threshold", size=(8.8, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
    p = np.linspace(0.5, 1, 300)
    for ax, cw, name in [(a1, 5, "how-to"), (a2, 50, "hoàn tiền")]:
        ce = 1
        tau = 1 - ce / cw
        ax.plot(p, (1 - p) * cw, color=t["c"][ORANGE], label="gửi: (1 − p̂)·c_w")
        ax.plot(p, np.full_like(p, ce), color=t["c"][BLUE], label="escalate: c_e")
        ax.axvline(tau, color=t["muted"], ls="--", lw=1)
        ax.axvspan(tau, 1, color=t["c"][GREEN], alpha=0.08, lw=0)
        ax.text(tau - 0.01, cw * 0.3, f"τ = {tau:.2f}", ha="right", fontsize=8.6, color=t["fg"])
        ax.scatter([0.9], [(1 - 0.9) * cw], color=t["fg"], s=26, zorder=4)
        dec = "gửi" if 0.9 > tau else "escalate"
        ax.text(0.9, (1 - 0.9) * cw + cw * 0.06, f"p̂ = 0.9 → {dec}", ha="center", fontsize=8.4, color=t["fg"])
        ax.set_xlabel("p̂ (xác suất draft đúng, đã hiệu chuẩn)")
        ax.set_ylabel("chi phí kỳ vọng (tương đối)")
        ax.set_title(f"{name}: c_w = {cw}, c_e = 1", fontsize=9.6)
        ax.set_ylim(0, cw * 0.55)
        ax.legend(fontsize=8, loc="lower left" if cw == 5 else "center left")
        ygrid(ax, t)


# ---------------------------------------------------------------- 4.4 ưu tiên nguồn
@figure("source-priority", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.4, width_ratios=[1, 1.1]))
    dt = np.linspace(0, 720, 300)
    a1.plot(dt, np.exp(-dt / 180), color=t["c"][BLUE])
    for d, lab in [(30, "S1"), (340, "S3")]:
        v = np.exp(-d / 180)
        a1.scatter([d], [v], color=t["c"][ORANGE], s=30, zorder=4)
        a1.text(d + 18, v + 0.04, f"{lab}: {d} ngày → {v:.3f}", fontsize=8.4, color=t["fg"])
    a1.set_xlim(0, 760)
    a1.set_xlabel("tuổi tài liệu Δt (ngày)"); a1.set_ylabel("f(Δt) = exp(−Δt/180)")
    a1.set_title("Độ mới", fontsize=9.6)
    ygrid(a1, t)
    names = ["S1\npolicy", "S3\nticket"]
    wt = np.array([3, 0.5]); fr = np.array([np.exp(-30 / 180), np.exp(-340 / 180)])
    a2.barh(names, wt, color=t["c"][VIOLET], height=0.5, label="w_type")
    a2.barh(names, fr, left=wt, color=t["c"][BLUE], height=0.5, label="w_fresh·f(Δt)")
    for i in range(2):
        a2.text(wt[i] + fr[i] + 0.05, i, f"prio = {wt[i] + fr[i]:.3f}", va="center", fontsize=8.8, color=t["fg"])
    a2.invert_yaxis(); a2.set_xlim(0, 5)
    a2.set_title("prio(c) của ví dụ mục 4.4", fontsize=9.6)
    a2.legend(fontsize=8, loc="lower right")
    xgrid(a2, t)


# ---------------------------------------------------------------- 5.2 constrained decoding
@figure("constrained-decoding", size=(8.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    toks = ["true", "false", '"yes"', "maybe", "khác"]
    before = [0.30, 0.45, 0.15, 0.05, 0.05]
    valid = [1, 1, 0, 0, 0]
    Z = sum(b for b, v in zip(before, valid) if v)
    after = [b / Z if v else 0 for b, v in zip(before, valid)]
    x = np.arange(len(toks)); w = 0.36
    ax.bar(x - w / 2, before, w * 0.92, color=t["muted"], label="p_θ trước khi mask")
    ax.bar(x + w / 2, after, w * 0.92, color=t["c"][BLUE], label="p̃ sau mask + chuẩn hóa lại")
    for i in range(len(toks)):
        ax.text(x[i] - w / 2, before[i] + 0.015, f"{before[i]:.2f}", ha="center", fontsize=8, color=t["fg2"])
        if valid[i]:
            ax.text(x[i] + w / 2, after[i] + 0.015, f"{after[i]:.2f}", ha="center", fontsize=8.4, color=t["fg"], weight="bold")
        else:
            ax.text(x[i] + w / 2, 0.02, "−∞", ha="center", fontsize=8.4, color=t["c"][RED])
    ax.set_xticks(x, toks)
    ax.set_ylim(0, 0.72)
    ax.set_title('Sau tiền tố {"escalate": — chỉ true/false thuộc A_t')
    ax.legend(fontsize=8.3, loc="upper right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.3 méo phân phối
@figure("constrained-distortion", size=(8.8, 3.2))
def _(fig, t):
    a0 = fig.add_axes([0.0, 0.05, 0.5, 0.9]); a1 = fig.add_axes([0.6, 0.17, 0.39, 0.66])
    a0.set_xlim(0, 60); a0.set_ylim(0, 40); a0.axis("off")
    box(a0, 2, 17, 10, 7, "bắt đầu", t, fs=8.4)
    box(a0, 22, 28, 9, 7, "A", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .15), fs=10)
    box(a0, 22, 6, 9, 7, "B", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .15), fs=10)
    arrow(a0, 12.3, 21.5, 21.7, 31.5, t); arrow(a0, 12.3, 19.5, 21.7, 9.5, t)
    a0.text(15, 29, "0.6", fontsize=8.6, color=t["fg"]); a0.text(15, 10, "0.4", fontsize=8.6, color=t["fg"])
    box(a0, 42, 28, 16, 7, "Ax (hợp lệ)", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .15), fs=8.4)
    box(a0, 42, 6, 16, 7, "By (hợp lệ)", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .15), fs=8.4)
    arrow(a0, 31.3, 31.5, 41.7, 31.5, t); arrow(a0, 31.3, 9.5, 41.7, 9.5, t)
    a0.text(33, 33.5, "p(x|A)=0.1", fontsize=8, color=t["fg"]); a0.text(33, 11.5, "p(y|B)=0.9", fontsize=8, color=t["fg"])
    a0.text(2, 38, "Mô hình đồ chơi: chỉ «Ax» và «By» hợp lệ", fontsize=9, weight="bold", color=t["fg"])
    a0.text(2, 1, "Sau A, model «muốn» token khác x (0.9) nhưng bị ép", fontsize=8, color=t["fg2"])
    true = np.array([0.6 * 0.1, 0.4 * 0.9]); true = true / true.sum()
    greedy = np.array([0.6, 0.4])
    x = np.arange(2); w = 0.36
    a1.bar(x - w / 2, true, w * 0.92, color=t["c"][GREEN], label="p_θ(y | y ∈ L) đúng")
    a1.bar(x + w / 2, greedy, w * 0.92, color=t["c"][BLUE], label="masking từng bước")
    for i in range(2):
        a1.text(x[i] - w / 2, true[i] + 0.02, f"{true[i]:.3f}", ha="center", fontsize=8.4, color=t["fg"])
        a1.text(x[i] + w / 2, greedy[i] + 0.02, f"{greedy[i]:.2f}", ha="center", fontsize=8.4, color=t["fg"])
    a1.set_xticks(x, ["Ax", "By"]); a1.set_ylim(0, 1.1)
    a1.set_title("Hai phân phối khác hẳn nhau", fontsize=9.6)
    a1.legend(fontsize=7.8, loc="upper left")
    ygrid(a1, t)


# ---------------------------------------------------------------- 8.1 groundedness
@figure("groundedness-claims", size=(8.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    claims = ["(1) hóa đơn VAT ở\nCài đặt > Thanh toán", "(2) chọn kỳ\ncần xuất", "(3) Business hoàn tiền\ntheo ngày chưa dùng",
              "(4) hủy trong 14 ngày\nkể từ gia hạn", "(5) «tiền về tài khoản\ntrong 3 ngày»"]
    P = [0.97, 0.91, 0.88, 0.95, 0.04]
    cols = [t["c"][GREEN] if p >= 0.7 else t["c"][RED] for p in P]
    x = np.arange(5)
    ax.bar(x, P, color=cols, width=0.55)
    ax.axhline(0.7, color=t["c"][ORANGE], ls="--", lw=1.4)
    ax.text(4.45, 0.73, "δ = 0.7", fontsize=8.4, color=t["fg"], ha="right")
    for xi, p in zip(x, P):
        ax.text(xi, p + 0.02, f"{p:.2f}", ha="center", fontsize=8.6, color=t["fg"])
    ax.set_xticks(x, claims, fontsize=7.8)
    ax.set_ylabel("P_NLI(entail | nguồn, claim)"); ax.set_ylim(0, 1.12)
    ax.set_title("Bốn claim của draft mục 7.3: G = 1; thêm claim (5): G = 4/5 = 0.8")
    ygrid(ax, t)


# ---------------------------------------------------------------- 8.2 cascade verifier
@figure("verifier-cascade", size=(8.6, 2.8))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 38))
    steps = [("Rule", "số, tên gói, menu path", AQUA), ("NLI / MiniCheck nhỏ", "nhanh, chạy cục bộ", BLUE),
             ("LLM-as-judge", "chỉ cho vùng không chắc", VIOLET)]
    for i, (a, b, c) in enumerate(steps):
        x = 2 + i * 30
        box(ax, x, 24, 26, 10, f"{a}\n{b}", t, color=t["c"][c], fill=tint(t["c"][c], t, .14), fs=8.4)
        if i < 2:
            arrow(ax, x + 26.5, 29, x + 29.5, 29, t, lw=1.6)
    ax.text(96, 31, "chi phí / claim ↑", fontsize=8.6, color=t["fg2"])
    # trục xác suất
    x0, x1 = 6, 118
    ax.plot([x0, x1], [10, 10], color=t["fg2"], lw=1.2)
    lo, hi = 0.3, 0.8
    X = lambda p: x0 + (x1 - x0) * p
    ax.add_patch(Rectangle((X(0), 7.5), X(lo) - X(0), 5, color=t["c"][RED], alpha=0.25, lw=0))
    ax.add_patch(Rectangle((X(lo), 7.5), X(hi) - X(lo), 5, color=t["c"][VIOLET], alpha=0.25, lw=0))
    ax.add_patch(Rectangle((X(hi), 7.5), X(1) - X(hi), 5, color=t["c"][GREEN], alpha=0.25, lw=0))
    ax.text(X(lo / 2), 15, "không hỗ trợ", ha="center", fontsize=8.4, color=t["fg"])
    ax.text(X((lo + hi) / 2), 15, "δ_low < P < δ_high → gọi LLM-judge", ha="center", fontsize=8.4, color=t["fg"])
    ax.text(X((hi + 1) / 2), 15, "hỗ trợ", ha="center", fontsize=8.4, color=t["fg"])
    for p in (0, lo, hi, 1):
        ax.text(X(p), 3.5, f"{p:g}" if p in (0, 1) else ("δ_low" if p == lo else "δ_high"), ha="center", fontsize=8, color=t["fg2"])
    ax.text(x0, 0, "điểm của model nhỏ (ngưỡng minh họa)", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 10.2 bề mặt tấn công
@figure("attack-surface", size=(8.8, 3.4))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    srcs = [("Email khách (direct)", RED), ("Ticket lịch sử đã index", ORANGE), ("Đính kèm: ảnh/PDF (OCR, VLM)", ORANGE),
            ("HTML ẩn, quoted reply, chữ ký", ORANGE), ("Tài liệu web qua tool", ORANGE)]
    for i, (s, c) in enumerate(srcs):
        y = 38 - i * 8
        box(ax, 2, y, 34, 6, s, t, color=t["c"][c], fill=tint(t["c"][c], t, .12), fs=8.2)
        arrow(ax, 36.5, y + 3, 47, 22, t, lw=1.1, color=t["c"][c])
    box(ax, 47.5, 15, 22, 14, "Prompt\n(lệnh và dữ liệu\nđều là token)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=8.6)
    arrow(ax, 70, 22, 76, 22, t, lw=1.8)
    goals = ["ép escalate = false, hứa hoàn tiền", "exfiltration qua URL / ảnh markdown", "lộ system prompt, dữ liệu khách khác",
             "lạm dụng tool (khi có agent)"]
    ax.text(77, 41, "Mục tiêu kẻ tấn công", fontsize=9, weight="bold", color=t["fg"])
    for i, g in enumerate(goals):
        box(ax, 77, 31 - i * 8, 45, 6, g, t, color=t["c"][RED], fill=tint(t["c"][RED], t, .08), fs=8.1, ha="left")
    ax.text(2, 45, "Nguồn đưa lệnh vào (đỏ: trực tiếp, cam: gián tiếp)", fontsize=9, weight="bold", color=t["fg"])


# ---------------------------------------------------------------- 10.3 lượng hóa rủi ro
@figure("injection-layers", size=(7.6, 3.0))
def _(fig, t):
    ax = fig.subplots()
    labels = ["tấn công", "qua spotlighting\nq₁ = 0.05", "qua classifier\nq₂ = 0.2", "qua output rail\nq₃ = 0.3"]
    vals = [1, 0.05, 0.05 * 0.2, 0.05 * 0.2 * 0.3]
    xs = np.arange(4)
    ax.plot(xs, vals, color=t["line"], lw=1.4, zorder=1)
    ax.scatter(xs, vals, s=70, color=[t["c"][RED], t["c"][ORANGE], t["c"][YELLOW], t["c"][VIOLET]], zorder=3)
    ax.set_xticks(xs, labels)
    for x_, v in zip(xs, vals):
        ax.text(x_ + 0.08, v * 1.35, f"{v:g}", ha="left", fontsize=8.8, color=t["fg"])
    ax.set_yscale("log"); ax.set_ylim(1e-3 * 0.5, 4); ax.set_xlim(-0.4, 3.6)
    ax.set_ylabel("tỷ lệ vượt qua (log)")
    ax.tick_params(axis="x", labelsize=8.2)
    ax.set_title("Giả định độc lập: 0.003 → ~1.6 vụ/năm (1/1.000 email là tấn công)", fontsize=9.6)
    ax.text(3.3, 0.06, "tấn công thích nghi phá\ngiả định độc lập → thật\nsự cao hơn", fontsize=8, color=t["fg2"], ha="right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 10.4 phòng thủ nhiều lớp
@figure("defense-layers", size=(8.8, 3.5))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 48))
    prob = [("1", "Tách kênh lệnh / dữ liệu", "instruction hierarchy, StruQ"), ("2", "Spotlighting", "escape + datamark"),
            ("3", "Phát hiện", "classifier trên input & chunk")]
    dmg = [("4", "Quyền tối thiểu", "generator không có tool ghi; rule OR escalate"), ("5", "Xác nhận trước hành động", "người duyệt hoàn tiền, đổi email"),
           ("6", "Xử lý đầu ra an toàn", "sanitize HTML, URL allowlist"), ("7", "Giám sát & red-team", "log, bộ test đa ngữ")]
    ax.text(2, 45, "Giảm XÁC SUẤT tấn công thành công (prompt-level)", fontsize=9, weight="bold", color=t["c"][BLUE] if t["name"] == "light" else t["fg"])
    for i, (k, a, b) in enumerate(prob):
        y = 35 - i * 9.5
        box(ax, 2, y, 54, 7.5, "", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .1))
        ax.text(4, y + 4.9, f"{k}. {a}", fontsize=8.8, weight="bold", color=t["fg"])
        ax.text(4, y + 1.8, b, fontsize=8, color=t["fg2"])
    ax.text(64, 45, "Giới hạn THIỆT HẠI khi thành công (kiến trúc)", fontsize=9, weight="bold", color=t["c"][GREEN] if t["name"] == "light" else t["fg"])
    for i, (k, a, b) in enumerate(dmg):
        y = 35 - i * 9.5
        box(ax, 64, y, 58, 7.5, "", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .1), lw=2 if k == "4" else 1.4)
        ax.text(66, y + 4.9, f"{k}. {a}", fontsize=8.8, weight="bold", color=t["fg"])
        ax.text(66, y + 1.8, b, fontsize=8, color=t["fg2"])
    ax.text(2, 1, "Chỉ nhóm bên phải đứng vững trước tấn công thích nghi.", fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 11 tổng hợp
@figure("hallucination-stack", size=(8.8, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 34))
    steps = [("Dữ liệu sạch\nauthority,\nupdated_at", "04"), ("Retrieval\nrecall, filter", "05–06"), ("Context nhỏ\nhết mâu thuẫn", "06–07"),
             ("Grounding\ncitation, abstain", "07"), ("Decoding\nT thấp, JSON", "01, 07"), ("Verify\nrepair có hạn", "07"),
             ("Ngưỡng theo\nintent", "10"), ("Vận hành\ntheo giai đoạn", "10–12")]
    cols = [AQUA, BLUE, BLUE, VIOLET, VIOLET, ORANGE, YELLOW, GREEN]
    for i, ((a, m), c) in enumerate(zip(steps, cols)):
        x = 2 + i * 15.2
        box(ax, x, 12, 14, 12, a, t, color=t["c"][c], fill=tint(t["c"][c], t, .13), fs=7.6)
        ax.text(x + 7, 8.5, f"Module {m}", ha="center", fontsize=7.6, color=t["fg2"])
        if i < len(steps) - 1:
            arrow(ax, x + 14.1, 18, x + 15.1, 18, t, lw=1.2, ms=8)
    ax.text(2, 29, "Không có biện pháp đơn lẻ: lỗi được giảm ở mọi tầng", fontsize=9.4, weight="bold", color=t["fg"])
    ax.text(2, 2.5, "Mục tiêu: lỗi hiếm, dễ phát hiện, và rẻ khi xảy ra.", fontsize=8.6, color=t["fg2"])


if __name__ == "__main__":
    run("07")
