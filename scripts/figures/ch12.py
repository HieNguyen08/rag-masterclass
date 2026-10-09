"""Hình minh họa cho Module 12 — Capstone: hệ thống AI CS Zendesk."""
import numpy as np
from matplotlib.patches import Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, run, tint, xgrid, ygrid)


def ink(t, c):
    return t["c"][c] if t["name"] == "light" else t["fg"]


# ---------------------------------------------------------------- 2 lợi ích kinh doanh
@figure("business-value", size=(8.4, 2.2))
def _(fig, t):
    ax = fig.subplots()
    parts = [("25% ticket tự trả lời × 8 phút", 1500 * 0.25 * 8, GREEN), ("50% ticket có draft × 3 phút", 1500 * 0.5 * 3, BLUE)]
    left = 0
    for n, v, c in parts:
        ax.barh(0, v, left=left, color=t["c"][c], height=0.5, label=f"{n} = {v:,.0f} phút".replace(",", "."))
        left += v
    ax.text(left + 60, 0, "5.250 phút/ngày ≈ 87,5 giờ\n≈ 11 agent-ngày công", va="center", fontsize=8.4, color=t["fg"], weight="bold")
    ax.set_xlim(0, 7600); ax.set_ylim(-0.45, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("phút agent tiết kiệm mỗi ngày (ước lượng lạc quan, giai đoạn 2)")
    ax.legend(fontsize=7.8, loc="upper left", ncol=2)
    ax.set_title("Lợi ích tiềm năng so với ~1.450 USD/tháng chi phí LLM: ràng buộc chính là an toàn, không phải tiền token", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 8.2 ngưỡng theo chi phí
@figure("cost-threshold", size=(7.8, 2.8))
def _(fig, t):
    ax = fig.subplots()
    cw = np.linspace(1.6, 60, 300)
    for ch, c in [(1.5, BLUE), (3.0, ORANGE)]:
        ax.plot(cw, 1 - ch / cw, color=t["c"][c], lw=1.8, label=f"C_h = {ch:g} USD (chuyển người có draft)".replace(".", ","))
    ax.plot(20, 0.925, "o", color=t["fg"], ms=6)
    ax.text(23, 0.72, "C_w = 20 USD → τ* = 0,925", fontsize=8, color=t["fg"])
    for x, lab in [(5, "how_to\n(sai nhẹ)"), (50, "account_access\n(sai nghiêm trọng)")]:
        ax.axvline(x, color=t["line"], ls=":", lw=1)
        ax.text(x + 0.8, 0.12, lab, fontsize=7.6, color=t["fg2"])
    ax.set_xlabel("C_w = chi phí kỳ vọng khi gửi một câu trả lời sai (USD, giả định)")
    ax.set_ylabel("ngưỡng τ* trên p̂")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 60)
    ax.legend(fontsize=7.6, loc="center right")
    ax.set_title("SEND khi p̂ > τ* = 1 − C_h / C_w: intent càng rủi ro, ngưỡng càng cao", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 10 rollout
@figure("rollout-gates", size=(8.8, 3.1))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    st = [("0. Shadow", "không ghi gì vào\nZendesk", "lỗi < 1%, p95 < 2 phút\nrecall «cần người» ≥ 95%", AQUA),
          ("1. Draft", "internal note +\nescalation tự động", "≥ 60% draft dùng được\n0 vi phạm chính sách", BLUE),
          ("2a. Tự gửi", "how_to, bug_report\nViệt/Anh, A/B 10%", "risk ≤ 2% (cận trên CI)\nCSAT không giảm > 0,1", GREEN),
          ("2b. Mở rộng", "từng intent /\nngôn ngữ (Nhật)", "tiêu chí như 2a,\ntính riêng từng phần", YELLOW),
          ("3. Agentic", "tool chỉ đọc\n(tài khoản, đơn)", "đánh giá tool use\nriêng, audit đầy đủ", VIOLET)]
    for k, (a, b, c, col) in enumerate(st):
        x = 2 + k * 24.4
        box(ax, x, 18, 21, 20, "", t, color=t["c"][col], fill=tint(t["c"][col], t, .14))
        ax.text(x + 1.5, 34.5, a, fontsize=8.8, weight="bold", color=t["fg"])
        ax.text(x + 1.5, 25, b, fontsize=7.4, color=t["fg"], va="center", linespacing=1.35)
        ax.text(x + 10.5, 12.5, c, fontsize=7.0, color=t["fg2"], ha="center", va="center", linespacing=1.35)
        if k < 4:
            arrow(ax, x + 21.3, 28, x + 24.1, 28, t, lw=1.4)
    ax.text(2, 41.5, "Mỗi bước mở thêm quyền tự động chỉ khi tiêu chí đo ≥ 2 tuần đạt (CI 95%)", fontsize=9, weight="bold", color=t["fg"])
    ax.text(2, 2.5, "Kill switch một cú bấm; tự lui về giai đoạn trước nếu risk > 2× ngưỡng hoặc CSAT tuần giảm > 0,3.",
            fontsize=7.8, color=ink(t, RED))


# ---------------------------------------------------------------- 10 cỡ mẫu review
@figure("review-sample-size", size=(7.8, 2.8))
def _(fig, t):
    ax = fig.subplots()
    E = np.linspace(0.004, 0.03, 300)
    for p, c in [(0.01, BLUE), (0.02, ORANGE), (0.05, RED)]:
        ax.plot(100 * E, 1.96 ** 2 * p * (1 - p) / E ** 2, color=t["c"][c], lw=1.8, label=f"risk thật p = {100 * p:g}%")
    n = 1.96 ** 2 * 0.02 * 0.98 / 0.01 ** 2
    ax.plot(1.0, n, "o", color=t["fg"], ms=6)
    ax.text(1.15, n + 250, f"±1% với p = 2% → n ≈ {n:,.0f}\n(~3 tuần ở 40 lượt SEND/ngày)".replace(",", "."), fontsize=7.8, color=t["fg"])
    ax.set_yscale("log")
    ax.set_xlabel("sai số mong muốn E (điểm phần trăm, CI 95%)")
    ax.set_ylabel("số câu trả lời đã gửi cần review")
    ax.legend(fontsize=7.6, loc="upper right")
    ax.set_title("n ≈ 1,96² p(1 − p) / E²: muốn đo risk nhỏ phải review nhiều", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 11 ma trận rủi ro
@figure("risk-matrix", size=(8.4, 3.4))
def _(fig, t):
    ax = fig.subplots()
    lik = ["Thấp", "Trung bình"]
    imp = ["Trung bình", "Cao", "Rất cao"]
    cells = {("Trung bình", "Rất cao"): ["Bịa chính sách giá/hoàn tiền"],
             ("Thấp", "Rất cao"): ["Lộ dữ liệu khách khác"],
             ("Trung bình", "Cao"): ["Prompt injection", "Tri thức cũ", "CS không chấp nhận"],
             ("Trung bình", "Trung bình"): ["Vòng lặp bot", "Gửi trùng email", "Nhà cung cấp LLM đổi/outage"]}
    for i, l in enumerate(lik):
        for j, m in enumerate(imp):
            sev = i + j
            c = [t["c"][GREEN], t["c"][YELLOW], t["c"][ORANGE], t["c"][RED]][min(3, sev)]
            ax.add_patch(Rectangle((j, i), 0.97, 0.95, fc=tint(c, t, 0.25 + 0.12 * sev), ec="none"))
            items = cells.get((l, m), [])
            ax.text(j + 0.485, i + 0.47, "\n".join(items) if items else "—", ha="center", va="center", fontsize=7.8,
                    color=t["fg"], linespacing=1.4)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2)
    ax.set_xticks(np.arange(3) + 0.485, imp, fontsize=8.2)
    ax.set_yticks(np.arange(2) + 0.47, lik, fontsize=8.2)
    ax.set_xlabel("tác động"); ax.set_ylabel("khả năng")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Bảng rủi ro mục 11 (rủi ro pháp lý phụ thuộc kiến trúc nên không đặt vào ô)", fontsize=9)


if __name__ == "__main__":
    run("12")
