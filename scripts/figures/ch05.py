"""Hình minh họa cho Module 05 — Retrieval: sparse, dense, ANN, hybrid."""
import numpy as np
from matplotlib.patches import Circle, Ellipse, Patch, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


# ---------------------------------------------------------------- 1.1 hai loại recall
@figure("two-recalls", size=(8.6, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    cR, cE, cA = t["c"][GREEN], t["c"][BLUE], t["c"][ORANGE]
    ax.add_patch(Ellipse((30, 22), 40, 30, color=cR, alpha=0.12, lw=0))
    ax.add_patch(Ellipse((30, 22), 40, 30, fill=False, ec=cR, lw=1.8))
    ax.add_patch(Ellipse((52, 22), 40, 30, color=cE, alpha=0.12, lw=0))
    ax.add_patch(Ellipse((52, 22), 40, 30, fill=False, ec=cE, lw=1.8))
    ax.add_patch(Ellipse((58, 20), 30, 22, fill=False, ec=cA, lw=2, ls="--"))
    ax.text(14, 39, "liên quan thật\n(nhãn người)", ha="center", fontsize=8.8, color=t["fg"])
    ax.text(62, 39.5, "top-k chính xác theo s\n(brute force)", ha="center", fontsize=8.8, color=t["fg"])
    ax.text(76, 5.5, "top-k của index ANN", ha="center", fontsize=8.8, color=cA)
    ax.text(84, 36, "Relevance recall@k", fontsize=9.5, weight="bold", color=t["fg"])
    ax.text(84, 31.5, "= |liên quan ∩ top-k| / |liên quan|", fontsize=8.6, color=t["fg2"])
    ax.text(84, 27.5, "→ đo hàm điểm s (embedding, BM25)", fontsize=8.6, color=t["fg2"])
    ax.text(84, 19, "ANN recall@k", fontsize=9.5, weight="bold", color=t["fg"])
    ax.text(84, 14.5, "= |top-k ANN ∩ top-k chính xác| / k", fontsize=8.6, color=t["fg2"])
    ax.text(84, 10.5, "→ đo thuật toán tìm argmax", fontsize=8.6, color=t["fg2"])
    ax.text(2, 1.5, "Hai lỗi độc lập nhau: ANN recall 99% vẫn đi cùng relevance recall 60% nếu embedding kém.",
            fontsize=8.5, color=t["fg2"])


# ---------------------------------------------------------------- 1.3 ba kiểu truy vấn
@figure("query-types", size=(7.6, 2.7))
def _(fig, t):
    ax = canvas(fig, (0, 110), (-4, 36))
    rows = [("Định danh chính xác", "ERR_SYNC_409, POST /v2/invoices"),
            ("Mô tả bằng lời của khách", "«bấm lưu mà nó cứ quay mãi»"),
            ("Xuyên ngôn ngữ", "hỏi tiếng Việt, tài liệu chỉ có tiếng Anh")]
    cols = ["BM25", "Dense"]
    marks = [("mạnh", "yếu"), ("yếu", "mạnh"), ("gần như vô dụng", "lựa chọn duy nhất")]
    for j, c in enumerate(cols):
        ax.text(73 + j * 23, 32.5, c, ha="center", fontsize=9.5, weight="bold", color=t["fg"])
    for i, ((name, ex), mk) in enumerate(zip(rows, marks)):
        y = 22 - i * 10
        ax.text(2, y + 4.5, name, fontsize=9.2, weight="bold", color=t["fg"])
        ax.text(2, y + 0.8, ex, fontsize=8.3, color=t["fg2"])
        for j, m in enumerate(mk):
            good = m in ("mạnh", "lựa chọn duy nhất")
            c = t["c"][GREEN] if good else t["c"][RED]
            box(ax, 62 + j * 23, y, 22, 7.5, m, t, color=c, fill=tint(c, t, .14), fs=8.2)
    ax.text(2, -3.5, "Cả ba nhóm đều xuất hiện mỗi ngày → hybrid là mặc định hợp lý.", fontsize=8.5, color=t["fg2"])


# ---------------------------------------------------------------- 2.1 inverted index
@figure("inverted-index", size=(8.4, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 34))
    ax.text(2, 31, "Posting list: term → [(tài liệu, tf)]", fontsize=9.4, weight="bold", color=t["fg"])
    lists = [("hoàn_tiền", [("d1", 3), ("d2", 1), ("d3", 1)], BLUE),
             ("hóa_đơn", [("d2", 1), ("d3", 2)], ORANGE),
             ("đăng_nhập", [("d7", 1), ("d9", 2), ("d12", 1), ("d40", 1)], t["muted"])]
    for i, (term, posts, c) in enumerate(lists):
        y = 20 - i * 8.5
        col = t["c"][c] if isinstance(c, int) else c
        box(ax, 2, y, 16, 6, term, t, color=col, fill=tint(col, t, .15), fs=8.8)
        arrow(ax, 18.3, y + 3, 22.5, y + 3, t)
        for j, (d, tf) in enumerate(posts):
            box(ax, 23 + j * 11, y, 10, 6, f"{d}, {tf}", t, fs=8.4, color=col if i < 2 else None)
    ax.text(72, 21, "Truy vấn «hoàn_tiền hóa_đơn»:", fontsize=8.8, color=t["fg"])
    ax.text(72, 16.5, "chỉ duyệt 2 posting list", fontsize=8.8, color=t["fg"])
    ax.text(72, 12, "chi phí ≈ Σ df_t, không phải N", fontsize=8.8, color=t["fg"], weight="bold")
    ax.text(72, 6, "Block-Max WAND: bỏ qua cả khối posting\nkhông thể lọt top-k", fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 2.4 ví dụ BM25
@figure("bm25-example", size=(7.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    idf_h, idf_b = np.log(1 + 2.5 / 3.5), np.log(1 + 3.5 / 2.5)
    def tfc(tf, K):
        return tf * 2.2 / (tf + K)
    docs = [("d1  |d|=20\ntf=(3, 0)", 1.2, 3, 0), ("d2  |d|=10\ntf=(1, 1)", 0.75, 1, 1), ("d3  |d|=40\ntf=(1, 2)", 2.1, 1, 2)]
    y = np.arange(3)
    a = np.array([idf_h * tfc(h, K) for _, K, h, _ in docs])
    b = np.array([idf_b * tfc(bb, K) if bb else 0 for _, K, _, bb in docs])
    ax.barh(y, a, color=t["c"][BLUE], height=0.55, label="hoàn_tiền")
    ax.barh(y, b, left=a, color=t["c"][ORANGE], height=0.55, label="hóa_đơn")
    for i in range(3):
        ax.text(a[i] + b[i] + 0.03, i, f"{a[i] + b[i]:.3f}", va="center", fontsize=9.2, color=t["fg"], weight="bold")
    ax.set_yticks(y, [d[0] for d in docs]); ax.invert_yaxis()
    ax.set_xlim(0, 2.15); ax.set_xlabel("điểm BM25 (k₁ = 1.2, b = 0.75)")
    ax.set_title("Đóng góp từng term: d2 thắng nhờ khớp cả hai term")
    ax.legend(fontsize=8.5, loc="lower right")
    xgrid(ax, t)


# ---------------------------------------------------------------- 2.5 k1 và b
@figure("bm25-k1-b", size=(8.8, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
    tf = np.linspace(0, 20, 300)
    for k1, c in [(0.5, VIOLET), (1.2, BLUE), (2.0, AQUA), (3.0, ORANGE)]:
        a1.plot(tf, tf * (k1 + 1) / (tf + k1), color=t["c"][c], label=f"k₁ = {k1} (trần {k1 + 1:.1f})")
    a1.plot(tf, tf, color=t["muted"], ls=":", lw=1.4, label="TF tuyến tính")
    for v in (1, 2, 3, 5, 10):
        a1.scatter([v], [v * 2.2 / (v + 1.2)], color=t["c"][BLUE], s=16, zorder=3)
    a1.set_ylim(0, 5.6); a1.set_xlim(0, 20)
    a1.set_xlabel("tf"); a1.set_ylabel("tf·(k₁+1) / (tf + k₁)")
    a1.set_title("Saturation theo k₁ (|d| = avgdl)")
    a1.legend(fontsize=7.9, loc="upper right", ncol=2)
    ygrid(a1, t)

    r = np.linspace(0.2, 3, 300)
    for b, c in [(0, t["muted"]), (0.3, t["c"][AQUA]), (0.75, t["c"][BLUE]), (1.0, t["c"][ORANGE])]:
        K = 1.2 * (1 - b + b * r)
        a2.plot(r, 2.2 / (1 + K), color=c, label=f"b = {b}")
    for rr in (0.5, 1, 2):
        K = 1.2 * (0.25 + 0.75 * rr)
        a2.scatter([rr], [2.2 / (1 + K)], color=t["c"][BLUE], s=18, zorder=3)
    a2.set_xlabel("|d| / avgdl"); a2.set_ylabel("thành phần TF khi tf = 1")
    a2.set_title("Chuẩn hóa độ dài theo b")
    a2.legend(fontsize=8, loc="upper right")
    ygrid(a2, t)


# ---------------------------------------------------------------- 2.7 & 2.8 tokenization
@figure("tokenization-vi-ja", size=(8.8, 3.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 50))

    def chips(x, y, toks, c, fs=8.4, gap=0.8):
        for s in toks:
            w = 1.6 + len(s) * 1.25 if not any(ord(ch) > 0x3000 for ch in s) else 2.2 + len(s) * 2.4
            box(ax, x, y, w, 4.6, s, t, color=c, fill=tint(c, t, .14), fs=fs, radius=0.6)
            x += w + gap
        return x
    ax.text(2, 47, "Tiếng Việt: «xuất hóa đơn VAT»", fontsize=9.4, weight="bold", color=t["fg"])
    rows = [("(a) âm tiết", ["xuất", "hóa", "đơn", "vat"], t["muted"], "«đơn» khớp cả «đơn giản»"),
            ("(b) tách từ", ["xuất", "hóa_đơn", "VAT"], t["c"][BLUE], "cần cùng bộ tách ở index & query"),
            ("(c) âm tiết + bigram", ["xuất", "hóa", "đơn", "vat", "xuất_hóa", "hóa_đơn", "đơn_vat"], t["c"][GREEN], "khuyến nghị: không lệch tokenizer")]
    for i, (lab, toks, c, cm) in enumerate(rows):
        y = 39 - i * 6.5
        ax.text(2, y + 2.3, lab, fontsize=8.5, color=t["fg2"], va="center")
        end = chips(26, y, toks, c)
        ax.text(end + 1.5, y + 2.3, cm, fontsize=8, color=t["fg2"], va="center")
    ax.text(2, 16.5, "Tiếng Nhật: «請求書をダウンロードできません»", fontsize=9.4, weight="bold", color=t["fg"])
    rows = [("morphological", ["請求書", "を", "ダウンロード", "でき", "ませ", "ん"], t["c"][BLUE], "precision cao"),
            ("char bigram", ["請求", "求書", "書を", "をダ", "ダウ", "…"], t["c"][ORANGE], "không OOV, nhiễu hơn")]
    for i, (lab, toks, c, cm) in enumerate(rows):
        y = 8.5 - i * 6.5
        ax.text(2, y + 2.3, lab, fontsize=8.5, color=t["fg2"], va="center")
        end = chips(26, y, toks, c)
        ax.text(end + 1.5, y + 2.3, cm, fontsize=8, color=t["fg2"], va="center")


# ---------------------------------------------------------------- 3.3 lời nguyền số chiều
@figure("distance-concentration", size=(6.6, 3.0))
def _(fig, t):
    ax = fig.subplots()
    rng = np.random.default_rng(0)
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ratios = []
    for D in dims:
        X = rng.normal(size=(2000, D)); q = rng.normal(size=D)
        d = np.linalg.norm(X - q, axis=1)
        ratios.append(d.max() / d.min())
    ax.plot(dims, ratios, color=t["c"][BLUE], marker="o", ms=5)
    ax.axhline(1, color=t["muted"], ls="--", lw=1)
    for D, r in zip(dims, ratios):
        if D in (2, 16, 128, 1024):
            ax.text(D, r * 1.12, f"{r:.1f}" if r < 100 else f"{r:.0f}", ha="center", fontsize=8.3, color=t["fg"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("số chiều D"); ax.set_ylabel("khoảng cách xa nhất / gần nhất")
    ax.set_title("Mô phỏng 2.000 điểm Gauss: khoảng cách «tập trung» khi D lớn")
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.4 IVF
@figure("ivf", size=(8.8, 3.5))
def _(fig, t):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1.15]))
    rng = np.random.default_rng(4)
    C = rng.uniform(0, 10, (16, 2))
    X = rng.uniform(0, 10, (500, 2))
    q = np.array([5.2, 5.0])
    order = np.argsort(np.linalg.norm(C - q, axis=1))
    probe = set(order[:3])
    lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
    inprobe = np.array([l in probe for l in lab])
    a1.scatter(X[~inprobe, 0], X[~inprobe, 1], s=5, color=t["muted"], alpha=0.5, lw=0)
    a1.scatter(X[inprobe, 0], X[inprobe, 1], s=7, color=t["c"][BLUE], lw=0)
    vor = Voronoi(np.r_[C, [[-50, -50], [-50, 60], [60, -50], [60, 60]]])
    voronoi_plot_2d(vor, ax=a1, show_points=False, show_vertices=False, line_colors=t["line"], line_width=1)
    a1.scatter(C[:, 0], C[:, 1], s=28, color=t["fg"], marker="x", lw=1.5)
    a1.scatter(*q, s=80, color=t["c"][ORANGE], marker="*", zorder=5)
    nn = X[np.argmin(np.linalg.norm(X - q, axis=1))]
    a1.set_xlim(0, 10); a1.set_ylim(0, 10); a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("n_list = 16, n_probe = 3: chỉ quét vùng xanh", fontsize=9.5)
    a1.text(0.2, -0.9, "× tâm cụm  ★ truy vấn — láng giềng thật nằm ngay sau\nranh giới của cụm không được probe sẽ bị bỏ sót",
            fontsize=7.9, color=t["fg2"], va="top")

    N, nprobe, D = 1e6, 16, 1
    nl = np.logspace(1.5, 5.5, 300)
    cost = nl + nprobe * N / nl
    a2.plot(nl, cost, color=t["c"][BLUE])
    opt = np.sqrt(nprobe * N)
    a2.scatter([opt], [opt + nprobe * N / opt], color=t["c"][ORANGE], zorder=4, s=36)
    a2.text(opt * 1.4, (opt + nprobe * N / opt) * 0.62, f"n_list* = √(n_probe·N) = {opt:,.0f}".replace(",", "."), fontsize=8.2, color=t["fg"])
    a2.scatter([1024], [1024 + 16 * N / 1024], color=t["c"][AQUA], zorder=4, s=30)
    a2.text(1024 * 0.8, (1024 + 16 * N / 1024) * 1.05, "ví dụ: n_list = 1024\n≈ 16.650 vector", fontsize=8.2, color=t["fg"], ha="right")
    a2.axhline(N, color=t["muted"], ls="--", lw=1); a2.text(40, N * 1.15, "flat: 10⁶", fontsize=8.2, color=t["fg2"])
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_ylim(3e3, 2e6)
    a2.set_xlabel("n_list"); a2.set_ylabel("số vector phải so")
    a2.set_title("Chi phí theo n_list (N = 10⁶, n_probe = 16)", fontsize=9.5)
    ygrid(a2, t)


# ---------------------------------------------------------------- 3.5 PQ + ADC
@figure("pq-adc", size=(8.8, 3.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 50))
    ax.text(2, 47, "Mã hóa PQ: mỗi đoạn → chỉ số tâm gần nhất", fontsize=9.2,
            weight="bold", color=t["fg"])
    cols = [BLUE, AQUA, ORANGE, VIOLET]
    for j, c in enumerate(cols):
        x = 2 + j * 14
        for k in range(4):
            ax.add_patch(Rectangle((x + k * 3.3, 37), 3.0, 5, color=t["c"][c], alpha=0.85, lw=0))
        ax.text(x + 6.5, 34.5, f"đoạn {j + 1}", ha="center", fontsize=8.6, color=t["fg"])
        arrow(ax, x + 6.5, 33, x + 6.5, 28.5, t, color=t["c"][c])
        box(ax, x + 2.5, 22, 8, 6, ["17", "203", "4", "88"][j], t, color=t["c"][c], fill=tint(t["c"][c], t, .15), fs=9)
    ax.text(2, 18, "D = 1024 fp32 (4096 byte) → m = 64 mã × 1 byte = 64 byte", fontsize=8.6, color=t["fg2"])
    ax.text(2, 14.5, "Mỗi đoạn có codebook K* = 256 tâm riêng (k-means)", fontsize=8.6, color=t["fg2"])
    ax.plot([62, 62], [3, 45], color=t["grid"], lw=1)
    ax.text(66, 47, "ADC: q giữ nguyên, tra bảng", fontsize=9.2, weight="bold", color=t["fg"])
    # bảng tra từ ví dụ số
    T1 = [0.05, 1.45, 0.25, 0.85]; T2 = [0.10, 0.65, 0.40, 0.05]
    for r, (name, T, hl) in enumerate([("T₁", T1, {0: ORANGE, 2: AQUA}), ("T₂", T2, {0: ORANGE, 3: AQUA})]):
        y = 35 - r * 8
        ax.text(66, y + 2.5, name, fontsize=9.2, color=t["fg"], va="center")
        for c_ in range(4):
            col = t["c"][hl[c_]] if c_ in hl else None
            box(ax, 71 + c_ * 10, y, 9, 5.2, f"{T[c_]:.2f}", t, color=col, fill=tint(col, t, .18) if col else None, fs=8.6, radius=0.6)
            if r == 0:
                ax.text(75.5 + c_ * 10, 41.5, f"c={c_}", ha="center", fontsize=7.8, color=t["fg2"])
    ax.text(66, 17, "x ↦ (0, 0):  0.05 + 0.10 = 0.15   (thật 0.10)", fontsize=8.5, color=t["c"][ORANGE] if t["name"] == "light" else t["fg"])
    ax.text(66, 12.5, "y ↦ (2, 3):  0.25 + 0.05 = 0.30   (thật 0.50)", fontsize=8.5, color=t["fg"])
    ax.text(66, 6.5, "m phép tra bảng + cộng cho mỗi vector;\nsai số có, nhưng thứ tự x < y được giữ.", fontsize=8.3, color=t["fg2"])


# ---------------------------------------------------------------- 3.6 HNSW
@figure("hnsw", size=(8.8, 3.6))
def _(fig, t):
    a1 = fig.add_axes([0.0, 0.0, 0.58, 1.0]); a2 = fig.add_axes([0.68, 0.14, 0.31, 0.72])
    rng = np.random.default_rng(11)
    P = rng.uniform(0, 10, (60, 2))
    lvl = np.zeros(60, int); lvl[rng.choice(60, 12, replace=False)] = 1
    top = np.where(lvl == 1)[0][:3]; lvl[top] = 2
    q = np.array([8.6, 2.0])
    a1.set_xlim(-0.5, 16); a1.set_ylim(-1, 25); a1.axis("off")
    offs = {0: 0, 1: 8.2, 2: 16.4}
    path_pts = []
    entry = top[0]
    cur = entry
    for L in (2, 1, 0):
        idx = np.where(lvl >= L)[0]
        oy = offs[L]
        Q = P[idx] * np.array([1, 0.55]) + np.array([L * 0.6, oy])
        a1.add_patch(Rectangle((-0.2 + L * 0.6, oy - 0.4), 10.6, 6.3, color=t["panel"], lw=0, zorder=0))
        # cạnh: mỗi điểm nối 2 láng giềng gần nhất trong tầng
        for i in range(len(idx)):
            dd = np.linalg.norm(P[idx] - P[idx][i], axis=1)
            for j in np.argsort(dd)[1:3]:
                a1.plot([Q[i, 0], Q[j, 0]], [Q[i, 1], Q[j, 1]], color=t["line"], lw=0.7, zorder=1)
        a1.scatter(Q[:, 0], Q[:, 1], s=10 if L == 0 else 18, color=t["c"][BLUE], zorder=2, lw=0)
        a1.text(11.2, oy + 2.5, f"tầng {L}" + ("  (mọi điểm)" if L == 0 else ""), fontsize=8.6, color=t["fg2"])
        # tìm tham lam trong tầng
        steps = [cur]
        while True:
            dd = np.linalg.norm(P[idx] - P[cur], axis=1)
            nb = idx[np.argsort(dd)[1:5]]
            best = min(nb, key=lambda k: np.linalg.norm(P[k] - q))
            if np.linalg.norm(P[best] - q) >= np.linalg.norm(P[cur] - q):
                break
            cur = best; steps.append(cur)
        S = P[steps] * np.array([1, 0.55]) + np.array([L * 0.6, oy])
        a1.plot(S[:, 0], S[:, 1], color=t["c"][ORANGE], lw=2.2, zorder=3)
        a1.scatter(S[0, 0], S[0, 1], s=34, color=t["c"][ORANGE], zorder=4)
        qq = q * np.array([1, 0.55]) + np.array([L * 0.6, oy])
        a1.scatter(*qq, marker="*", s=90, color=t["c"][RED], zorder=5)
    a1.text(0, 24, "Tìm tham lam từ tầng thưa xuống tầng dày (★ = q)", fontsize=9.2, weight="bold", color=t["fg"])

    N, M = 1e6, 16
    L = np.arange(0, 6)
    cnt = N * M ** (-L.astype(float))
    a2.bar(L, cnt, color=t["c"][VIOLET], width=0.6)
    for x, c in zip(L, cnt):
        a2.text(x, c * 1.4, f"{c:,.0f}" if c >= 1 else "~1", ha="center", fontsize=7.8, color=t["fg"])
    a2.set_yscale("log"); a2.set_ylim(0.5, 1e7)
    a2.set_xlabel("tầng L"); a2.set_ylabel("số điểm (log)")
    a2.set_title("N = 10⁶, M = 16: P(ℓ ≥ L) = M⁻ᴸ", fontsize=9.2)
    ygrid(a2, t)


# ---------------------------------------------------------------- 3.7 DiskANN
@figure("diskann", size=(8.0, 2.8))
def _(fig, t):
    ax = canvas(fig, (0, 116), (0, 38))
    box(ax, 2, 6, 44, 28, "", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .1))
    ax.text(4, 30.5, "RAM", fontsize=9.6, weight="bold", color=t["fg"])
    box(ax, 6, 15, 36, 10, "vector nén PQ\n(định hướng duyệt đồ thị)", t, fs=8.8)
    ax.text(24, 9, "nhỏ: ~64 byte/vector", ha="center", fontsize=8.3, color=t["fg2"])
    box(ax, 64, 6, 50, 28, "", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .1))
    ax.text(66, 30.5, "SSD", fontsize=9.6, weight="bold", color=t["fg"])
    for i in range(3):
        box(ax, 67 + i * 15.5, 15, 14, 10, f"block {i + 1}\nvector đầy đủ\n+ láng giềng", t, fs=7.8)
    ax.text(89, 9, "1 lần đọc block = 1 bước nhảy + rescoring", ha="center", fontsize=8.3, color=t["fg2"])
    arrow(ax, 46.5, 22, 63.5, 22, t, lw=1.8)
    ax.text(55, 24.5, "đọc", ha="center", fontsize=8.2, color=t["fg2"])
    ax.text(2, 1, "Đồ thị Vamana một tầng, α > 1 giữ cạnh dài → ít bước → ít lần đọc SSD.", fontsize=8.5, color=t["fg2"])


# ---------------------------------------------------------------- 3.8 bộ nhớ 1M
@figure("memory-1m", size=(8.4, 3.3))
def _(fig, t):
    ax = fig.subplots()
    rows = [("768-d fp32", 3.07, 0.15, 0.2, 0), ("1024-d fp32", 4.10, 0.15, 0.2, 0), ("1024-d fp16", 2.05, 0.15, 0.2, 0),
            ("int8 + fp32 trên đĩa", 1.02, 0.15, 0.2, 4.1), ("binary + fp16 trên đĩa", 0.13, 0.15, 0.2, 2.05),
            ("IVF-PQ m=64", 0.064, 0.004, 0.2, 0)]
    y = np.arange(len(rows))
    v = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows]); p = np.array([r[3] for r in rows])
    ax.barh(y, v, color=t["c"][BLUE], height=0.6, label="vector (RAM)")
    ax.barh(y, g, left=v, color=t["c"][ORANGE], height=0.6, label="đồ thị HNSW / tâm cụm")
    ax.barh(y, p, left=v + g, color=t["c"][AQUA], height=0.6, label="payload")
    for i, r in enumerate(rows):
        tot = v[i] + g[i] + p[i]
        s = f"~{tot:.1f} GB RAM" + (f"  + {r[4]} GB đĩa" if r[4] else "")
        ax.text(tot + 0.06, i, s, va="center", fontsize=8.3, color=t["fg"])
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 6.2); ax.set_xlabel("GB (ước lượng, N = 10⁶, HNSW M = 16)")
    ax.set_title("Bộ nhớ cho 1 triệu vector theo cấu hình")
    ax.legend(fontsize=8, loc="lower right")
    xgrid(ax, t)


# ---------------------------------------------------------------- 4.1 hợp hai danh sách
@figure("hybrid-union", size=(8.4, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.25]))
    rS, rD = 0.70, 0.80
    vals = [rS, rD, 1 - (1 - rS) * (1 - rD)]
    b = a1.bar(["sparse", "dense", "hợp\n(độc lập)"], vals, color=[t["c"][ORANGE], t["c"][BLUE], t["c"][GREEN]], width=0.6)
    for bb, v in zip(b, vals):
        a1.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=t["fg"])
    a1.set_ylim(0, 1.1); a1.set_ylabel("P(tài liệu đúng trong top-k)")
    a1.set_title("Lỗi độc lập: 1 − (1−r_S)(1−r_D)", fontsize=9.6)
    ygrid(a1, t)
    rho = np.linspace(0, 1, 200)
    sS, sD = np.sqrt(rS * (1 - rS)), np.sqrt(rD * (1 - rD))
    # P(cả hai trượt) = (1-rS)(1-rD) + ρ·σS·σD, chặn trên bởi min(1-rS, 1-rD)
    both_miss = np.minimum((1 - rS) * (1 - rD) + rho * sS * sD, min(1 - rS, 1 - rD))
    a2.plot(rho, 1 - both_miss, color=t["c"][GREEN], label="hợp hai danh sách")
    a2.axhline(rD, color=t["c"][BLUE], ls="--", lw=1.2, label="dense đơn lẻ")
    a2.set_xlabel("tương quan ρ giữa hai biến «trượt»"); a2.set_ylim(0.75, 0.97)
    a2.set_title("Lỗi tương quan dương → lợi ích giảm", fontsize=9.6)
    a2.legend(fontsize=8.3, loc="upper right")
    ygrid(a2, t)


# ---------------------------------------------------------------- 4.2 RRF
@figure("rrf-k", size=(8.8, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32))
    r = np.arange(1, 101)
    for k, c in [(0, ORANGE), (2, VIOLET), (60, BLUE)]:
        w = 1 / (k + r)
        a1.plot(r, w / w[0], color=t["c"][c], label=f"k = {k}")
    a1.set_xscale("log"); a1.set_xlabel("thứ hạng (log)"); a1.set_ylabel("1/(k+rank), chuẩn hóa về rank 1")
    a1.set_title("k nhỏ: «đứng đầu là thắng»; k lớn: đếm phiếu", fontsize=9.6)
    a1.legend(fontsize=8.3)
    ygrid(a1, t)
    docs = ["A", "C", "B", "E", "D", "F"]
    bm = {"A": 1, "B": 2, "C": 3, "D": 4}; de = {"C": 1, "A": 2, "E": 3, "F": 4}
    sb = np.array([1 / (60 + bm[d]) if d in bm else 0 for d in docs])
    sd = np.array([1 / (60 + de[d]) if d in de else 0 for d in docs])
    x = np.arange(len(docs))
    a2.bar(x, sb * 1000, color=t["c"][ORANGE], width=0.6, label="từ BM25")
    a2.bar(x, sd * 1000, bottom=sb * 1000, color=t["c"][BLUE], width=0.6, label="từ dense")
    for i in range(len(docs)):
        a2.text(i, (sb[i] + sd[i]) * 1000 + 0.6, f"{(sb[i] + sd[i]) * 1000:.2f}", ha="center", fontsize=8, color=t["fg"])
    a2.set_xticks(x, docs); a2.set_ylim(0, 38)
    a2.set_ylabel("RRF × 1000 (k = 60)")
    a2.set_title("Ví dụ «ERR_SYNC_409…»: đồng thuận thắng", fontsize=9.6)
    a2.legend(fontsize=8.3, loc="upper right")
    ygrid(a2, t)


# ---------------------------------------------------------------- 4.3 convex
@figure("convex-alpha", size=(6.8, 3.2))
def _(fig, t):
    ax = fig.subplots()
    sp = {"A": 1, "B": 0.667, "C": 0.111, "D": 0}
    de = {"C": 1, "A": 0.667, "E": 0.417, "F": 0}
    al = np.linspace(0, 1, 200)
    for d, c in [("A", BLUE), ("C", ORANGE), ("B", AQUA), ("E", VIOLET)]:
        y = al * de.get(d, 0) + (1 - al) * sp.get(d, 0)
        ax.plot(al, y, color=t["c"][c], label=d)
        ax.text(1.01, y[-1], d, fontsize=9, color=t["c"][c], va="center", weight="bold")
    for a in (0.3, 0.5, 0.8):
        ax.axvline(a, color=t["muted"], ls=":", lw=1)
        ax.text(a, 1.05, f"α={a}", ha="center", fontsize=8.2, color=t["fg2"])
    xc = (1 - 0.111) / ((1 - 0.111) + (1 - 0.667))
    ax.scatter([xc], [xc * 0.667 + (1 - xc)], color=t["fg"], s=22, zorder=4)
    ax.text(xc - 0.05, 0.86, f"A = C tại α ≈ {xc:.2f}", fontsize=8.3, color=t["fg"], ha="right")
    ax.set_xlabel("α (trọng số dense)"); ax.set_ylabel("điểm hybrid (min-max)")
    ax.set_ylim(-0.03, 1.12); ax.set_xlim(0, 1.06)
    ax.set_title("Convex combination: thứ hạng đổi theo α")
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.2 filter selectivity
@figure("filter-selectivity", size=(8.8, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32))
    s = np.logspace(-3, 0, 300)
    a1.plot(s, 50 / s, color=t["c"][BLUE])
    a1.scatter([0.01], [5000], color=t["c"][ORANGE], zorder=4, s=36)
    a1.text(0.012, 7000, "s = 0.01 → k′ = 5.000", fontsize=8.4, color=t["fg"])
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("selectivity s"); a1.set_ylabel("k′ cần lấy để còn k = 50")
    a1.set_title("Post-filter: k′ ≈ k / s", fontsize=9.6)
    ygrid(a1, t)
    for M0, c in [(16, ORANGE), (32, BLUE)]:
        a2.plot(s, (1 - s) ** M0, color=t["c"][c], label=f"M₀ = {M0}")
    for sv in (0.01, 0.1):
        for M0, c in [(16, ORANGE), (32, BLUE)]:
            v = (1 - sv) ** M0
            a2.scatter([sv], [v], color=t["c"][c], s=24, zorder=4)
            a2.text(sv * 1.15, v + (0.03 if M0 == 16 else -0.07), f"{v:.3f}", fontsize=8, color=t["fg"])
    a2.set_xscale("log"); a2.set_ylim(-0.03, 1.05)
    a2.set_xlabel("selectivity s"); a2.set_ylabel("P(nút không có láng giềng hợp lệ)")
    a2.set_title("In-filter HNSW: đồ thị con tan rã khi s nhỏ", fontsize=9.6)
    a2.legend(fontsize=8.3)
    ygrid(a2, t)


# ---------------------------------------------------------------- 7 multi-tenant
@figure("tenant-isolation", size=(8.8, 3.0))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    titles = [("Logic", "một collection + tenant_id"), ("Phân vùng", "partition / shard / partial index"), ("Vật lý", "collection / DB riêng")]
    cols = [t["c"][BLUE], t["c"][ORANGE], t["c"][AQUA]]
    rng = np.random.default_rng(5)
    for i, ((a, b), ) in enumerate(zip(titles)):
        x = 2 + i * 41
        ax.text(x, 37, a, fontsize=9.6, weight="bold", color=t["fg"])
        ax.text(x, 33.5, b, fontsize=8.2, color=t["fg2"])
        if i == 0:
            box(ax, x, 6, 38, 24, "", t)
            pts = rng.uniform([x + 2, 8], [x + 36, 28], (45, 2))
            for j, (px, py) in enumerate(pts):
                ax.add_patch(Circle((px, py), 0.8, color=cols[j % 3], lw=0))
        elif i == 1:
            box(ax, x, 6, 38, 24, "", t)
            for j in range(3):
                box(ax, x + 1.5 + j * 12.2, 8, 11, 20, "", t, color=cols[j], fill=tint(cols[j], t, .12), ls="--")
                pts = rng.uniform([x + 3 + j * 12.2, 10], [x + 11 + j * 12.2, 26], (12, 2))
                for px, py in pts:
                    ax.add_patch(Circle((px, py), 0.8, color=cols[j], lw=0))
        else:
            for j in range(3):
                box(ax, x + j * 13, 6, 12, 24, "", t, color=cols[j], fill=tint(cols[j], t, .12))
                pts = rng.uniform([x + 1.5 + j * 13, 8], [x + 10.5 + j * 13, 28], (12, 2))
                for px, py in pts:
                    ax.add_patch(Circle((px, py), 0.8, color=cols[j], lw=0))
    ax.text(2, 1.5, "← rẻ, đơn giản, dựa vào việc không bao giờ quên filter", fontsize=8.3, color=t["fg2"])
    ax.text(122, 1.5, "cô lập mạnh, dễ audit, đắt theo số tenant →", fontsize=8.3, color=t["fg2"], ha="right")


if __name__ == "__main__":
    run("05")
