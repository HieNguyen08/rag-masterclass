"""Hình minh họa cho Module 11 — Production & quy mô."""
import numpy as np

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, run, tint, xgrid, ygrid)


def vn(x, fmt="{:,.0f}"):
    return fmt.format(x).replace(",", "#").replace(".", ",").replace("#", ".")


# ---------------------------------------------------------------- 1.3 token theo bước
@figure("tokens-per-step", size=(8.4, 2.7))
def _(fig, t):
    ax = fig.subplots()
    steps = ["Phân loại (nhỏ)", "Viết lại query (nhỏ)", "Sinh draft (lớn)", "Kiểm chứng (nhỏ)"]
    tin = np.array([1500, 1500, 6500, 3500]); tout = np.array([100, 150, 400, 200])
    y = np.arange(4)[::-1]
    ax.barh(y, tin, color=t["c"][BLUE], height=0.55, label="token vào")
    ax.barh(y, tout, left=tin, color=t["c"][ORANGE], height=0.55, label="token ra")
    for yi, a, b in zip(y, tin, tout):
        ax.text(a + b + 120, yi, f"{vn(a)} + {vn(b)}", va="center", fontsize=8, color=t["fg"])
    ax.set_yticks(y, steps, fontsize=8.2)
    ax.set_xlim(0, 8600)
    ax.set_xlabel("token mỗi lượt chạy (giả định)")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.set_title("13.000 token vào, 850 token ra mỗi lượt: chi phí bị chi phối bởi input (≈ 15 : 1)", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 1.4–1.5 chi phí & hòa vốn
@figure("cost-options", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.2]))
    names = ["A\nmột model lớn", "B\nrouting", "C\nB + caching", "E\nself-host\n2 GPU"]
    vals = [3105, 1695, 1450, 3600]
    cols = [t["c"][RED], t["c"][ORANGE], t["c"][GREEN], t["c"][VIOLET]]
    a1.bar(range(4), vals, color=cols, width=0.6)
    for k, v in enumerate(vals):
        a1.text(k, v + 60, vn(v), ha="center", fontsize=8.2, color=t["fg"])
    a1.set_xticks(range(4), names, fontsize=7.6)
    a1.set_ylim(0, 4200)
    a1.set_ylabel("USD / tháng")
    a1.set_title("1.500 ticket/ngày (giá tra cứu 10/2026)", fontsize=9)
    ygrid(a1, t)
    n = np.linspace(0, 8000, 200)
    a2.plot(n, n * 2 * 0.016 * 30, color=t["c"][GREEN], lw=1.8, label="API, phương án C (0,016 USD/lượt, r = 2)")
    a2.axhline(3600, color=t["c"][VIOLET], lw=1.8, label="self-host 2 GPU (2,5 USD/giờ, giả định)")
    a2.axvline(1500, color=t["line"], ls=":", lw=1)
    a2.text(1600, 4600, "hiện tại", fontsize=7.8, color=t["fg2"])
    a2.plot(3750, 3600, "o", color=t["fg"], ms=6)
    a2.text(4300, 2300, "hòa vốn ≈ 3.750\nticket/ngày", fontsize=7.8, color=t["fg"])
    a2.set_xlabel("ticket / ngày")
    a2.set_ylabel("USD / tháng")
    a2.set_ylim(0, 8200)
    a2.legend(fontsize=7.2, loc="upper left")
    a2.set_title("Điểm hòa vốn API ↔ self-host", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 1.6 độ nhạy
@figure("cost-sensitivity", size=(7.8, 2.4))
def _(fig, t):
    ax = fig.subplots()
    items = [("r: 2 → 3 lượt/ticket", 50), ("số chunk: 6 → 10", 20), ("lọc 15% ticket trước LLM", -15)]
    y = np.arange(3)[::-1]
    for yi, (n, v) in zip(y, items):
        c = t["c"][RED] if v > 0 else t["c"][GREEN]
        ax.barh(yi, v, color=c, height=0.55)
        ax.text(v + (1.5 if v > 0 else -1.5), yi, f"{v:+d}%", va="center", ha="left" if v > 0 else "right", fontsize=8.6, color=t["fg"])
    ax.axvline(0, color=t["fg2"], lw=1)
    ax.set_yticks(y, [n for n, _ in items], fontsize=8.2)
    ax.set_xlim(-30, 62)
    ax.set_xlabel("thay đổi chi phí phương án C")
    ax.set_title("Thừa số nào đáng tối ưu trước", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 2.2 KV cache theo độ dài
@figure("kv-cache-length", size=(8.0, 2.9))
def _(fig, t):
    ax = fig.subplots()
    L = np.linspace(0, 32000, 200)
    cfg = [("Qwen3-8B (GQA, 8 KV head): 144 KiB/token", 147456, BLUE),
           ("Qwen3-32B (GQA, 8 KV head): 256 KiB/token", 262144, ORANGE),
           ("32B nếu MHA đầy đủ (64 KV head): 2 MiB/token", 262144 * 8, RED)]
    for n, b, c in cfg:
        ax.plot(L, L * b / 1e9, color=t["c"][c], lw=1.8, label=n)
    ax.axvline(8000, color=t["line"], ls=":", lw=1)
    for b, c in [(147456, BLUE), (262144, ORANGE), (262144 * 8, RED)]:
        ax.plot(8000, 8000 * b / 1e9, "o", color=t["c"][c], ms=5)
    ax.text(8300, 33, "8.000 token\n(một lượt sinh draft)", fontsize=7.8, color=t["fg2"])
    ax.text(8300 + 400, 8000 * 262144 * 8 / 1e9 - 0.6, "≈ 16,8 GB", fontsize=7.8, color=t["fg"])
    ax.text(8300 + 400, 2.6, "≈ 2,1 GB", fontsize=7.8, color=t["fg"])
    ax.set_xlabel("độ dài sequence L (token)")
    ax.set_ylabel("KV cache (GB, BF16)")
    ax.set_ylim(0, 40)
    ax.set_xlim(0, 32000)
    ax.legend(fontsize=7.4, loc="upper right")
    ax.set_title("M_KV = 2 · n_layers · n_kv · d_head · b · L — GQA giảm 8 lần", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 2.3 ngân sách VRAM
@figure("gpu-memory-budget", size=(8.8, 3.4))
def _(fig, t):
    ax = fig.subplots()
    cfgs = [("BF16 / KV BF16", 65.6, 2.1), ("FP8 / KV BF16", 32.8, 2.1), ("FP8 / KV FP8", 32.8, 1.05), ("INT4 / KV BF16", 18.0, 2.1)]
    y = np.arange(4)[::-1]
    for yi, (n, w, per) in zip(y, cfgs):
        kv = max(0.0, 73.6 - w - 4)
        ax.barh(yi, w, color=t["c"][VIOLET], height=0.55)
        ax.barh(yi, 4, left=w, color=t["muted"], height=0.55)
        ax.barh(yi, kv, left=w + 4, color=t["c"][GREEN], height=0.55)
        ax.text(74.5, yi, f"~{int(kv // per)} seq 8k", va="center", fontsize=8.2, color=t["fg"], weight="bold")
    ax.axvline(73.6, color=t["fg2"], ls="--", lw=1)
    ax.set_yticks(y, [c[0] for c in cfgs], fontsize=8.2)
    ax.set_xlim(0, 90)
    ax.set_xlabel("GB trên GPU 80 GB (92% dùng được = 73,6 GB)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=t["c"][VIOLET], label="trọng số Qwen3-32B"), Patch(color=t["muted"], label="overhead ~4 GB"),
                       Patch(color=t["c"][GREEN], label="còn cho KV cache")], fontsize=7.4, loc="upper center",
              bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False)
    ax.set_title("Model 30B+ trên một GPU 80 GB: quantization trọng số gần như bắt buộc", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 2.4 PagedAttention
@figure("paged-attention", size=(7.6, 2.4))
def _(fig, t):
    ax = fig.subplots()
    names = ["cấp phát liền mạch\n(max_seq_len 8.192)", "PagedAttention\n(block 16 token)"]
    alloc = [163840, 60160]; used = [60000, 60000]
    y = np.arange(2)[::-1]
    ax.barh(y, alloc, color=t["panel2"], edgecolor=t["line"], height=0.55, label="slot đã cấp phát")
    ax.barh(y, used, color=t["c"][BLUE], height=0.55, label="slot hữu ích (20 seq × 3.000 token)")
    for yi, a, u in zip(y, alloc, used):
        ax.text(a + 2000, yi, f"{vn(a)} slot → hiệu dụng " + f"{100 * u / a:.1f}%".replace(".", ","), va="center", fontsize=8, color=t["fg"])
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlim(0, 230000)
    ax.legend(fontsize=7.4, loc="lower right")
    ax.set_xlabel("slot token trong KV cache")
    xgrid(ax, t)


# ---------------------------------------------------------------- 2.8 speculative decoding
@figure("speculative-decoding", size=(7.6, 2.8))
def _(fig, t):
    ax = fig.subplots()
    a = np.linspace(0, 0.98, 200)
    for k, c in [(2, BLUE), (4, ORANGE), (8, VIOLET)]:
        ax.plot(a, (1 - a ** (k + 1)) / (1 - a), color=t["c"][c], lw=1.8, label=f"k = {k} token nháp")
    ax.plot(0.7, (1 - 0.7 ** 5) / 0.3, "o", color=t["fg"], ms=6)
    ax.text(0.52, 3.2, "α = 0,7; k = 4 → 2,77", fontsize=8, color=t["fg"])
    ax.set_xlabel("α = xác suất một token nháp được chấp nhận")
    ax.set_ylabel("token kỳ vọng / lượt verify")
    ax.set_xlim(0, 1); ax.set_ylim(0, 9.5)
    ax.legend(fontsize=7.8, loc="upper left")
    ax.set_title("(1 − α^(k+1)) / (1 − α): lợi ích lớn khi output chép lại context (α cao)", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.2 latency budget
@figure("latency-budget", size=(8.8, 3.3))
def _(fig, t):
    ax = fig.subplots()
    rows = [("Webhook → enqueue", .03, .15), ("Lấy ticket, làm sạch, che PII", .5, 1.9),
            ("Phân loại + condense (LLM nhỏ)", 1.6, 4), ("Embedding + hybrid retrieval", .15, .5),
            ("Rerank top-50", .3, .9), ("Sinh draft (LLM lớn)", 8, 20), ("Verify (LLM nhỏ)", 1.5, 4),
            ("Ghi vào Zendesk", .4, 2)]
    y = np.arange(len(rows))[::-1]
    for yi, (n, p50, p95) in zip(y, rows):
        c = t["c"][RED] if p50 >= 8 else t["c"][BLUE]
        ax.barh(yi, p50, color=c, height=0.55)
        ax.plot([p50, p95], [yi, yi], color=t["fg2"], lw=1.2)
        ax.plot(p95, yi, "|", color=t["fg2"], ms=9, mew=1.5)
        ax.text(p95 + 0.3, yi, f"{vn(p50, '{:.2f}').rstrip('0').rstrip(',')} / {vn(p95, '{:.2f}').rstrip('0').rstrip(',')} s",
                va="center", fontsize=7.6, color=t["fg"])
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 25)
    ax.set_xlabel("giây (thanh = p50, vạch = p95; không tính chờ queue)")
    ax.set_title("Tổng ~13 s (p50) / ~33 s (tổng p95): sinh draft chiếm hơn 60%", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 3.3 thời gian chờ theo mức sử dụng
@figure("utilization-wait", size=(7.4, 2.7))
def _(fig, t):
    ax = fig.subplots()
    rho = np.linspace(0, 0.97, 300)
    ax.plot(rho, 1 / (1 - rho), color=t["c"][VIOLET], lw=1.8)
    for r, lab in [(0.5, "ρ = 0,5 → 2×"), (0.8, "ρ = 0,8 → 5×"), (0.9, "ρ = 0,9 → 10×")]:
        ax.plot(r, 1 / (1 - r), "o", color=t["c"][ORANGE], ms=5)
        ax.text(r - 0.02, 1 / (1 - r) + 1.2, lab, fontsize=7.8, color=t["fg"], ha="right")
    ax.set_xlabel("mức sử dụng ρ = λ / (c μ)")
    ax.set_ylabel("thời gian lưu trú / thời gian phục vụ")
    ax.set_ylim(0, 34); ax.set_xlim(0, 1)
    ax.set_title("Minh họa hàng đợi M/M/1: W = 1/(μ − λ) bùng nổ khi ρ → 1", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.5 backoff + jitter
@figure("backoff-jitter", size=(7.8, 2.8))
def _(fig, t):
    ax = fig.subplots()
    rng = np.random.default_rng(3)
    n = np.arange(5)
    cap = np.minimum(30, 1.0 * 2 ** n)
    ax.bar(n, cap, color=tint(t["c"][BLUE], t, .25), edgecolor=t["c"][BLUE], width=0.6, label="trần t₀·2ⁿ (khoảng lấy mẫu)")
    ax.bar(n, cap / 2, color=t["c"][BLUE], width=0.25, label="kỳ vọng = trần / 2")
    for k in range(40):
        ax.plot(n + rng.uniform(-0.25, 0.25, 5), rng.uniform(0, cap), ".", color=t["c"][ORANGE], ms=2.5, alpha=0.6)
    ax.plot([], [], ".", color=t["c"][ORANGE], label="40 client lấy mẫu ngẫu nhiên (full jitter)")
    ax.set_xticks(n, [f"lần thử {k + 1}" for k in n], fontsize=8)
    ax.set_ylabel("giây chờ")
    ax.set_ylim(0, 19)
    ax.legend(fontsize=7.4, loc="upper left")
    ax.set_title("Full jitter: client tản ra thay vì cùng thử lại một lúc; tổng kỳ vọng 15,5 s", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 4.3 semantic cache
@figure("semantic-cache", size=(7.8, 2.8))
def _(fig, t):
    ax = fig.subplots()
    e = np.logspace(-4, -1, 300)
    h = 0.3
    for cerr, c in [(1, GREEN), (5, ORANGE), (20, RED)]:
        ax.plot(e, 1000 * h * (0.016 - e * cerr), color=t["c"][c], lw=1.8, label=f"c_err = {cerr} USD (hòa vốn e = {100 * 0.016 / cerr:.2f}%)".replace(".", ","))
    ax.axhline(0, color=t["fg2"], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("e = tỉ lệ câu trả lời từ cache bị sai khi trúng (thang log)")
    ax.set_ylabel("lợi ích ròng (USD / 1.000 query)")
    ax.set_ylim(-35, 7)
    ax.legend(fontsize=7.4, loc="lower left")
    ax.set_title("Δ = h [c_run − e · c_err], h = 0,3: chỉ có lợi khi e rất nhỏ", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 4.4 bộ nhớ index
@figure("index-memory", size=(8.0, 2.5))
def _(fig, t):
    ax = fig.subplots()
    cfgs = [("float32", 819), ("int8", 205), ("binary", 25.6)]
    y = np.arange(3)[::-1]
    for yi, (n, v) in zip(y, cfgs):
        ax.barh(yi, v, color=t["c"][BLUE], height=0.55)
        ax.barh(yi, 25.6, left=v, color=t["c"][ORANGE], height=0.55)
        ax.barh(yi, 400, left=v + 25.6, color=t["muted"], height=0.55)
        ax.text(v + 425.6 + 15, yi, f"{vn(v + 425.6)} MB", va="center", fontsize=8.2, color=t["fg"])
    ax.set_yticks(y, [f"vector {n}" for n, _ in cfgs], fontsize=8.2)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("MB cho 200.000 chunk × 1.024 chiều")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=t["c"][BLUE], label="vector"), Patch(color=t["c"][ORANGE], label="đồ thị HNSW (M = 16)"),
                       Patch(color=t["muted"], label="payload ~2 KB/chunk")], fontsize=7.4, loc="lower right")
    ax.set_title("Dưới 1,5 GB: một node + replica là đủ, chưa cần shard", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 8.2 cascade
@figure("cascade-cost", size=(7.6, 2.8))
def _(fig, t):
    ax = fig.subplots()
    q = np.linspace(0, 1, 200)
    cl, cs, cv = 0.017, 0.0018, 0.001
    ax.plot(q, 1000 * (cs + cv + (1 - q) * cl), color=t["c"][GREEN], lw=1.8, label="cascade: c_s + c_v + (1 − q) c_ℓ")
    ax.axhline(1000 * cl, color=t["c"][RED], lw=1.6, label="luôn dùng model lớn: c_ℓ")
    ax.axvline(0.165, color=t["line"], ls=":", lw=1)
    ax.text(0.18, 3, "hòa vốn q = 0,165", fontsize=7.8, color=t["fg2"])
    ax.plot(0.6, 1000 * (cs + cv + 0.4 * cl), "o", color=t["fg"], ms=6)
    ax.text(0.62, 10.8, "q = 0,6 → 9,6 (−44%)", fontsize=7.8, color=t["fg"])
    ax.set_xlabel("q = tỉ lệ lượt model nhỏ đạt chuẩn (theo verifier)")
    ax.set_ylabel("USD / 1.000 lượt sinh draft")
    ax.set_ylim(0, 22); ax.set_xlim(0, 1)
    ax.legend(fontsize=7.4, loc="upper right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 9.4 tác động lỗ hổng kho
@figure("gap-impact", size=(8.0, 2.5))
def _(fig, t):
    ax = fig.subplots()
    rows = [("Hóa đơn điện tử lỗi MST\n120/tuần · g = 0,7 · 12 phút", 1008), ("SSO với Azure AD\n40/tuần · g = 0,9 · 25 phút", 900),
            ("Đổi mật khẩu\n300/tuần · g = 0,15 · 8 phút", 360)]
    y = np.arange(3)[::-1]
    ax.barh(y, [r[1] for r in rows], color=[t["c"][RED], t["c"][ORANGE], t["muted"]], height=0.55)
    for yi, (_, v) in zip(y, rows):
        ax.text(v + 15, yi, f"{vn(v)} phút/tuần", va="center", fontsize=8.2, color=t["fg"])
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.8)
    ax.set_xlim(0, 1300)
    ax.set_xlabel("Impact = n_c · g_c · h_c (phút agent tiết kiệm được mỗi tuần, giả định)")
    xgrid(ax, t)


if __name__ == "__main__":
    run("11")
