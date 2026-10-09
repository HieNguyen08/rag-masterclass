"""Hình minh họa cho Module 04 — Ingestion, làm sạch dữ liệu & chunking."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


# ---------------------------------------------------------------- 2. thẩm quyền nguồn
@figure("authority-tiers", size=(8.6, 3.0))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 41))
    tiers = [("1", "Chính sách\nchính thức", "giá · hoàn tiền · SLA", VIOLET),
             ("2", "Tài liệu\nsản phẩm / API", "release notes", BLUE),
             ("3", "Help Center", "~800 bài × 3 locale", AQUA),
             ("4", "Macro", "~300 mẫu trả lời", YELLOW),
             ("5", "Q/A trích\ntừ ticket", "~200K ticket", ORANGE)]
    for i, (k, name, sub, c) in enumerate(tiers):
        x = 2 + i * 23.4; h = 32 - i * 4
        box(ax, x, 2, 22, h, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .16))
        ax.text(x + 11, 2 + h - 3.5, f"tier {k}", ha="center", fontsize=8.4, color=t["fg2"])
        ax.text(x + 11, (2 + h - 3.5 + 5) / 2 + 0.8, name, ha="center", va="center", fontsize=9, color=t["fg"], weight="bold")
        ax.text(x + 11, 5, sub, ha="center", fontsize=7.8, color=t["fg2"])
    arrow(ax, 4, 37.5, 115, 37.5, t, color=t["fg2"])
    ax.text(4, 38.6, "thẩm quyền cao", fontsize=8.5, color=t["fg2"])
    ax.text(115, 38.6, "khối lượng lớn, thẩm quyền thấp", fontsize=8.5, color=t["fg2"], ha="right")


# ---------------------------------------------------------------- 3.2 Unicode
@figure("unicode-forms", size=(8.6, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 38))
    ax.text(2, 35, "Cùng hiển thị «ệ» nhưng khác chuỗi byte", fontsize=9.6, weight="bold", color=t["fg"])
    box(ax, 2, 20, 18, 10, "ệ\nU+1EC7", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .16), fs=10)
    ax.text(11, 16.5, "NFC: 1 code point", ha="center", fontsize=8.6, color=t["fg2"])
    for i, (g, cp) in enumerate([("e", "U+0065"), ("◌̣", "U+0323"), ("◌̂", "U+0302")]):
        box(ax, 28 + i * 11, 20, 10, 10, f"{g}\n{cp}", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .16), fs=9)
    ax.text(44, 16.5, "NFD: 3 code point", ha="center", fontsize=8.6, color=t["fg2"])
    ax.text(31, 7, "«Tiếng Việt»: 10 code point (NFC) vs 14 (NFD)\n→ == sai, hash khác, token khác",
            ha="center", fontsize=8.6, color=t["fg"])
    ax.plot([64, 64], [3, 34], color=t["grid"], lw=1)
    ax.text(68, 35, "NFKC gộp biến thể trình bày", fontsize=9.6, weight="bold", color=t["fg"])
    rows = [("ＡＢＣ１２３", "ABC123", "full-width"), ("ｶﾀｶﾅ", "カタカナ", "katakana\nhalf-width"), ("①  ²", "1  2", "khoanh tròn, số mũ\n(cẩn thận!)")]
    for i, (a, b, lab) in enumerate(rows):
        y = 26 - i * 9
        box(ax, 67, y, 16, 6.5, a, t, fs=9)
        arrow(ax, 83.5, y + 3.2, 88, y + 3.2, t)
        box(ax, 88.5, y, 12, 6.5, b, t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .16), fs=9)
        ax.text(101.5, y + 3.2, lab, fontsize=7.6, color=t["fg2"], va="center")


# ---------------------------------------------------------------- 4.1 giải phẫu email
@figure("email-anatomy", size=(8.6, 3.6))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 50))
    segs = [
        ("Dạ em vẫn chưa xuất được hóa đơn ạ, nút Export bị mờ.", "nội dung mới → GIỮ", GREEN, 1),
        ("Trân trọng,\nNguyễn Văn A — Phòng Kế toán\nĐT: 09xx xxx xxx", "chữ ký + PII → cắt", ORANGE, 3),
        ("Vào Th 3, 14 thg 10, 2026 lúc 09:12, Support đã viết:\n> Chào anh A, anh vui lòng vào Settings > Billing…\n>> On Mon, Oct 13, 2026 … wrote: Hi, I can't export…",
         "quoted reply (đã có ở comment trước) → cắt", BLUE, 3),
        ("CONFIDENTIAL: This email and any attachments are…", "disclaimer lặp ở hàng nghìn email → cắt", VIOLET, 1),
    ]
    y = 49.5
    for text, lab, c, nl in segs:
        h = 3.2 + 3.0 * nl
        y -= h + 1.6
        box(ax, 2, y, 72, h, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .13), radius=0.8)
        ax.text(4, y + h / 2, text, fontsize=8.4, color=t["fg"], va="center", linespacing=1.4)
        ax.text(77, y + h / 2, lab, fontsize=8.8, color=t["fg"], va="center",
                weight="bold" if c == GREEN else "normal")
        ax.add_patch(Rectangle((74.3, y + h / 2 - 0.15), 2, 0.3, color=t["c"][c]))
    ax.text(2, 1.2, "Embed nguyên comment → vector bị kéo về câu trả lời cũ, chữ ký thành hub, PII lan khắp index.",
            fontsize=8.5, color=t["fg2"])


# ---------------------------------------------------------------- 5.2 Jaccard
@figure("jaccard-shingles", size=(8.4, 3.0))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 43))
    ca, cb = t["c"][BLUE], t["c"][ORANGE]
    ax.add_patch(Circle((42, 20), 18, color=ca, alpha=0.18, lw=0))
    ax.add_patch(Circle((60, 20), 18, color=cb, alpha=0.18, lw=0))
    ax.add_patch(Circle((42, 20), 18, fill=False, ec=ca, lw=1.6))
    ax.add_patch(Circle((60, 20), 18, fill=False, ec=cb, lw=1.6))
    ax.text(30, 39, "A: «…vào tài khoản»", ha="center", fontsize=9, color=t["fg"], weight="bold")
    ax.text(72, 39, "B: «…vào hệ thống»", ha="center", fontsize=9, color=t["fg"], weight="bold")
    for i, s in enumerate(["vào tài", "tài khoản"]):
        ax.text(32, 23 - i * 6, s, ha="center", fontsize=8.6, color=t["fg"])
    for i, s in enumerate(["vào hệ", "hệ thống"]):
        ax.text(70, 23 - i * 6, s, ha="center", fontsize=8.6, color=t["fg"])
    for i, s in enumerate(["tôi không", "không đăng", "đăng nhập", "nhập được", "được vào"]):
        ax.text(51, 31 - i * 5, s, ha="center", fontsize=8.6, color=t["fg"])
    ax.text(86, 26, "|A ∩ B| = 5", fontsize=9.5, color=t["fg"])
    ax.text(86, 20, "|A ∪ B| = 9", fontsize=9.5, color=t["fg"])
    ax.text(86, 13, "J = 5/9 ≈ 0.556", fontsize=10.5, color=t["fg"], weight="bold")
    ax.text(2, 1, "Shingle 2 từ của hai câu; Jaccard = phần chung / phần hợp", fontsize=8.5, color=t["fg2"])


# ---------------------------------------------------------------- 5.3 MinHash
@figure("minhash-variance", size=(8.6, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
    J = 0.7
    k = np.arange(16, 513)
    sd = np.sqrt(J * (1 - J) / k)
    a1.fill_between(k, J - 2 * sd, J + 2 * sd, color=t["c"][BLUE], alpha=0.18, lw=0, label="J ± 2σ")
    a1.plot(k, np.full_like(k, J, dtype=float), color=t["c"][BLUE], lw=2, label="J thật = 0.7")
    for kk in (64, 128, 256):
        s = np.sqrt(J * (1 - J) / kk)
        a1.errorbar([kk], [J], yerr=[2 * s], color=t["c"][ORANGE], capsize=4, lw=2, fmt="o", ms=5)

    a1.text(505, 0.97, "k=64: σ=0.057\nk=128: σ=0.041\nk=256: σ=0.029", ha="right", va="top", fontsize=8.3, color=t["fg"])
    a1.set_xlabel("số hàm băm k"); a1.set_ylabel("Ĵ")
    a1.set_ylim(0.4, 1.0); a1.set_xlim(0, 520)
    a1.set_title("Var(Ĵ) = J(1−J)/k")
    a1.legend(fontsize=8.2, loc="lower right")
    ygrid(a1, t)

    rng = np.random.default_rng(2)
    for J_, c in [(0.5, VIOLET), (0.7, BLUE), (0.85, AQUA)]:
        est = rng.binomial(128, J_, 20000) / 128
        a2.hist(est, bins=(np.arange(-0.5, 129.5, 2)) / 128, color=t["c"][c], alpha=0.75, label=f"J = {J_}")
    a2.set_xlabel("Ĵ với k = 128 (mô phỏng 20.000 lần)")
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_title("Đủ tách «gần trùng» khỏi «cùng chủ đề»")
    a2.legend(fontsize=8.2, loc="upper left")


# ---------------------------------------------------------------- 5.4 LSH
@figure("lsh-s-curve", size=(6.8, 3.4))
def _(fig, t):
    ax = fig.subplots()
    s = np.linspace(0, 1, 400)
    for (b, r), c in [((32, 4), BLUE), ((16, 8), ORANGE), ((8, 16), AQUA)]:
        P = 1 - (1 - s ** r) ** b
        ax.plot(s, P, color=t["c"][c], label=f"(b, r) = ({b}, {r}),  s* ≈ {(1 / b) ** (1 / r):.2f}")
        ss = (1 / b) ** (1 / r)
        ax.axvline(ss, color=t["c"][c], lw=0.8, ls=":")
    for sv in (0.5, 0.8):
        p = 1 - (1 - sv ** 8) ** 16
        ax.scatter([sv], [p], color=t["c"][ORANGE], zorder=4, s=34)
        ax.text(sv + 0.015, p - 0.07 if sv == 0.8 else p + 0.04, f"P({sv}) = {p:.3f}", fontsize=8.4, color=t["fg"])
    ax.set_xlabel("độ tương đồng Jaccard s của một cặp")
    ax.set_ylabel("P(thành ứng viên)")
    ax.set_title("LSH banding với k = 128: đường cong chữ S")
    ax.legend(fontsize=8.3, loc="upper left")
    ygrid(ax, t)


# ---------------------------------------------------------------- 6 PII
@figure("pii-redaction", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    box(ax, 2, 6, 52, 28, "", t)
    ax.text(4, 31, "Trước", fontsize=9, color=t["fg2"], weight="bold")
    raw = ["Chào CS, em là Nguyễn Văn A.", "Email cũ a.nguyen@abc.vn không nhận", "được mã, em đổi sang an.nv@gmail.com.",
           "SĐT: 0912 345 678. Thẻ 4111 1111 1111 1111", "Mã đơn 202610140001."]
    for i, s in enumerate(raw):
        ax.text(4, 27 - i * 4.4, s, fontsize=8.3, color=t["fg"])
    arrow(ax, 55, 20, 64, 20, t, lw=2)
    ax.text(59.5, 22.5, "redact", ha="center", fontsize=8.2, color=t["fg2"])
    box(ax, 65, 6, 57, 28, "", t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .12))
    ax.text(67, 31, "Sau (placeholder có kiểu, nhất quán trong tài liệu)", fontsize=9, color=t["fg2"], weight="bold")
    red = ["Chào CS, em là <PERSON_1>.", "Email cũ <EMAIL_1> không nhận", "được mã, em đổi sang <EMAIL_2>.",
           "SĐT: <PHONE_1>. Thẻ <CARD_1>", "Mã đơn 202610140001.   ← không qua Luhn: giữ"]
    for i, s in enumerate(red):
        ax.text(67, 27 - i * 4.4, s, fontsize=8.3, color=t["fg"])
    ax.text(2, 1.5, "Mapping ngược (placeholder → giá trị) chỉ nằm trong vault riêng — không bao giờ ở index, prompt hay log.",
            fontsize=8.4, color=t["fg2"])


@figure("luhn-check", size=(8.4, 2.4))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 32))
    digits = [int(c) for c in "79927398713"][::-1]
    xs = [6 + i * 9.6 for i in range(len(digits))]
    total = 0
    ax.text(2, 29, "Số 79927398713, đọc từ phải sang trái", fontsize=9.4, weight="bold", color=t["fg"])
    for i, (x, d) in enumerate(zip(xs, digits)):
        even = i % 2 == 1
        v = d * 2 if even else d
        v2 = v - 9 if v > 9 else v
        total += v2
        c = t["c"][ORANGE] if even else t["line"]
        box(ax, x - 3.8, 17, 7.6, 7, str(d), t, color=c, fill=tint(t["c"][ORANGE], t, .15) if even else t["panel"], fs=11)
        ax.text(x, 12.5, (f"×2={v}" + (f"→{v2}" if v > 9 else "")) if even else "", ha="center", fontsize=7.6, color=t["fg2"])
        ax.text(x, 8, str(v2), ha="center", fontsize=10, color=t["fg"], weight="bold" if even else "normal")
    ax.text(2, 2, f"Tổng = {total} chia hết cho 10 → hợp lệ.  Chuỗi số ngẫu nhiên chỉ qua Luhn với xác suất 1/10.",
            fontsize=8.8, color=t["fg"])


# ---------------------------------------------------------------- 7.2 trọng số chất lượng × độ mới
@figure("ticket-weight-decay", size=(6.8, 3.2))
def _(fig, t):
    ax = fig.subplots()
    dt = np.linspace(0, 730, 300); T = 365
    ax.plot(dt, np.exp(-dt / T), color=t["c"][BLUE], label="CSAT good (q = 1)")
    ax.plot(dt, 0.5 * np.exp(-dt / T), color=t["c"][ORANGE], label="không đánh giá (q = 0.5)")
    ax.plot(dt, 0 * dt, color=t["c"][RED], label="CSAT bad (q = 0)")
    for d, q, c in [(30, 1, BLUE), (400, 1, BLUE), (30, 0.5, ORANGE)]:
        w = q * np.exp(-d / T)
        ax.scatter([d], [w], color=t["c"][c], zorder=4, s=34, edgecolor=t["bg"])
        ax.text(d + 15, w + 0.03, f"{d} ngày: w ≈ {w:.2f}", fontsize=8.4, color=t["fg"])
    ax.set_xlabel("Δt — số ngày từ khi giải quyết (T = 365)")
    ax.set_ylabel("w = q_csat · exp(−Δt/T)")
    ax.set_ylim(-0.04, 1.08)
    ax.set_title("Trọng số Q/A từ ticket theo chất lượng và độ mới")
    ax.legend(fontsize=8.3)
    ygrid(ax, t)


# ---------------------------------------------------------------- 8.2 mô hình kích thước chunk
@figure("chunk-size-model", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
    l, u, o, B = 60, 200, 40, 4000
    s = np.linspace(100, 1000, 400)
    P = np.minimum(1, (s - l) / (s - o))
    cosd = np.where(s >= u, np.sqrt(u / s), 1.0)
    a1.plot(s, P, color=t["c"][BLUE], label="P_trọn: span không bị cắt đôi")
    a1.plot(s, cosd, color=t["c"][ORANGE], label="cos pha loãng √(u/s)")
    a1.axvline(u, color=t["muted"], ls="--", lw=1)
    a1.text(u + 12, 1.06, "s ≈ u = 200 (một đơn vị ngữ nghĩa)", fontsize=8.2, color=t["fg2"])
    for sv in (100, 200, 400, 800):
        a1.scatter([sv], [min(1, (sv - l) / (sv - o))], color=t["c"][BLUE], s=22, zorder=3)
        a1.scatter([sv], [np.sqrt(u / sv) if sv >= u else 1], color=t["c"][ORANGE], s=22, zorder=3)
    a1.set_ylim(0.25, 1.12); a1.set_xlabel("kích thước chunk s (token)")
    a1.set_title("Chunk lớn: ít bị cắt đôi, nhưng loãng hơn")
    a1.legend(fontsize=8.1, loc="lower right")
    ygrid(a1, t)

    a2.plot(s, l / s, color=t["c"][AQUA], label="độ chính xác context ℓ/s")
    a2b = a2  # một trục: số chunk chuẩn hóa về [0,1] để tránh hai thang đo
    a2b.plot(s, (B / s) / (B / 100), color=t["c"][VIOLET], label="số chunk k = B/s (chia cho 40)")
    for sv in (100, 200, 400, 800):
        a2.scatter([sv], [l / sv], color=t["c"][AQUA], s=22, zorder=3)
        a2.scatter([sv], [B / sv / 40], color=t["c"][VIOLET], s=22, zorder=3)
    a2.set_xlabel("kích thước chunk s (token)")
    a2.set_ylim(0, 1.15)
    a2.set_title("Chunk lớn: nhiều token thừa, ít chunk vừa ngân sách")
    a2.legend(fontsize=8.1, loc="upper right")
    ygrid(a2, t)
    fig.text(0.5, -0.1, "ℓ = 60, u = 200, o = 40, B = 4.000 token — đúng các giả định của bảng ví dụ mục 8.2.",
             ha="center", fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 8.3 fixed + overlap
@figure("fixed-overlap", size=(8.6, 2.6))
def _(fig, t):
    ax = fig.subplots()
    L, s, o = 2600, 400, 80; g = s - o
    n = int(np.ceil((L - o) / g))
    ax.add_patch(Rectangle((0, n + 0.3), L, 0.6, color=t["panel2"], lw=0))
    ax.text(L / 2, n + 0.6, f"tài liệu L = {L} token", ha="center", va="center", fontsize=8.6, color=t["fg"])
    for i in range(n):
        a = i * g; b = min(a + s, a + s)
        ax.add_patch(Rectangle((a, n - 1 - i + 0.15), s, 0.7, color=t["c"][BLUE], alpha=0.85, lw=0))
        if i > 0:
            ax.add_patch(Rectangle((a, n - 1 - i + 0.15), o, 0.7, color=t["c"][ORANGE], lw=0))
        ax.text(a + s + 20, n - 1 - i + 0.5, f"#{i + 1}", va="center", fontsize=7.8, color=t["fg2"])
    ax.axvline(L, color=t["muted"], ls="--", lw=1)
    ax.set_xlim(0, L + 450); ax.set_ylim(-0.2, n + 1.1)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("vị trí token")
    ax.set_title(f"s = {s}, o = {o} → bước g = {g}, {n} chunk, lưu {n * s} token (thừa {n * s / L - 1:.0%})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=t["c"][BLUE], label="chunk"), Patch(color=t["c"][ORANGE], label="phần chồng lấn o")],
              fontsize=8.2, loc="lower left")


# ---------------------------------------------------------------- 8.4 heading-aware
@figure("heading-aware", size=(8.8, 3.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 50))
    box(ax, 2, 40, 32, 6, "Xuất hóa đơn PDF (H1)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14), fs=9, weight="bold")
    hs = [("## Điều kiện", 30), ("## Các bước", 20), ("## Lỗi thường gặp", 10)]
    for name, y in hs:
        box(ax, 10, y, 24, 6, name, t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .14), fs=8.8)
        ax.plot([5, 5, 10], [40, y + 3, y + 3], color=t["line"], lw=1.2)
    ax.text(10, 4, "ranh giới chunk = ranh giới heading;\nkhông cắt giữa bảng, code, danh sách bước", fontsize=8.3, color=t["fg2"])
    chunks = [(37, "[Help Center > Thanh toán > Xuất hóa đơn PDF > Điều kiện]", "Chỉ gói Pro trở lên; quyền Billing admin…"),
              (24, "[… > Xuất hóa đơn PDF > Các bước]", "1. Vào Settings > Billing  2. Chọn kỳ…"),
              (11, "[… > Xuất hóa đơn PDF > Lỗi thường gặp]", "Nếu nút Export bị mờ, kiểm tra quyền…")]
    for (y, path, body), (_, hy) in zip(chunks, hs):
        box(ax, 46, y, 76, 10, "", t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .1))
        ax.text(48, y + 6.8, path, fontsize=8.1, color=t["c"][AQUA] if t["name"] == "light" else t["fg"])
        ax.text(48, y + 2.6, body, fontsize=8.4, color=t["fg"])
        arrow(ax, 34.5, hy + 3, 45.5, y + 5, t, color=t["c"][AQUA])
    ax.text(46, 48, "Chunk mang heading path: ngữ cảnh rẻ, xác định, dùng cả cho embed lẫn citation",
            fontsize=8.6, color=t["fg2"])


# ---------------------------------------------------------------- 8.5 semantic chunking
@figure("semantic-chunking", size=(7.2, 3.0))
def _(fig, t):
    ax = fig.subplots()
    cos = np.array([0.82, 0.78, 0.41, 0.80, 0.76, 0.35, 0.84])
    d = 1 - cos
    thr = np.percentile(d, 80)
    x = np.arange(1, 8)
    cols = [t["c"][RED] if v > thr else t["c"][BLUE] for v in d]
    ax.bar(x, d, color=cols, width=0.6)
    ax.axhline(thr, color=t["c"][ORANGE], ls="--", lw=1.4)
    ax.text(0.55, thr + 0.02, f"ngưỡng = phân vị 80 ≈ {thr:.2f}", fontsize=8.4, color=t["fg"], ha="left")
    for xi, v in zip(x, d):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.2, color=t["fg2"])
    ax.set_xticks(x, [f"{i}|{i + 1}" for i in x])
    ax.set_xlabel("ranh giới giữa câu i và câu i+1")
    ax.set_ylabel("δᵢ = 1 − cos(eᵢ, eᵢ₊₁)")
    ax.set_ylim(0, 0.8)
    ax.set_title("Cắt sau câu 3 và câu 6 → chunk: câu 1–3 · 4–6 · 7–8")
    ygrid(ax, t)


# ---------------------------------------------------------------- 8.6 parent-child
@figure("parent-child", size=(8.6, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 44))
    ax.text(2, 41, "Index child nhỏ (truy hồi chính xác) → trả về parent lớn (đủ ngữ cảnh cho LLM)", fontsize=9.2,
            color=t["fg"], weight="bold")
    for p, (x, name) in enumerate([(2, "Parent A: mục «Hoàn tiền»"), (62, "Parent B: mục «Gia hạn»")]):
        box(ax, x, 6, 56, 30, "", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .08), ls="--")
        ax.text(x + 2, 32.5, name, fontsize=8.8, color=t["fg"], weight="bold")
        scores = [0.81, 0.77, 0.42] if p == 0 else [0.55, 0.38, 0.30]
        for i, sc in enumerate(scores):
            hit = sc > 0.7
            box(ax, x + 3 + i * 17.5, 14, 16, 12, f"child {i + 1}\nscore {sc:.2f}", t,
                color=t["c"][GREEN] if hit else t["line"], fill=tint(t["c"][GREEN], t, .2) if hit else t["panel"], fs=8.4)
        ax.text(x + 3, 9, f"score(parent) = max = {max(scores):.2f}" + ("  → trả về cả mục" if p == 0 else ""),
                fontsize=8.4, color=t["fg"] if p == 0 else t["fg2"])
    ax.text(2, 1.5, "Hai child cùng parent lọt top-k → gộp thành một parent, không lặp context.", fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 8.7 late chunking
@figure("late-chunking", size=(8.8, 3.5))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 48))
    cols = [BLUE, AQUA, ORANGE]
    # Cách thường
    ax.text(2, 45, "Cắt rồi encode (thường)", fontsize=9.4, color=t["fg"], weight="bold")
    for i, c in enumerate(cols):
        x = 2 + i * 19
        box(ax, x, 33, 17, 7, f"chunk {i + 1}", t, color=t["c"][c], fill=tint(t["c"][c], t, .16), fs=8.6)
        box(ax, x, 22, 17, 6, "Encoder", t, fs=8.4)
        arrow(ax, x + 8.5, 33, x + 8.5, 28.2, t)
        box(ax, x + 4, 11, 9, 6, f"e{i + 1}", t, color=t["c"][c], fs=8.6)
        arrow(ax, x + 8.5, 22, x + 8.5, 17.2, t)
    ax.text(2, 5, "«tính năng này» trong chunk 2\nkhông biết là tính năng nào", fontsize=8.4, color=t["fg2"])
    ax.plot([62, 62], [2, 46], color=t["grid"], lw=1)
    # Late chunking
    ax.text(66, 45, "Encode cả tài liệu rồi mới cắt (late chunking)", fontsize=9.4, color=t["fg"], weight="bold")
    box(ax, 66, 33, 55, 7, "toàn bộ tài liệu (ngữ cảnh dài)", t, fs=8.6)
    box(ax, 66, 22, 55, 6, "Encoder — mỗi token attend cả tài liệu", t, fs=8.4)
    arrow(ax, 93.5, 33, 93.5, 28.2, t)
    for j in range(15):
        c = cols[j // 5]
        ax.add_patch(Rectangle((67 + j * 3.6, 17.5), 3.0, 3.0, color=t["c"][c], lw=0))
    arrow(ax, 93.5, 22, 93.5, 20.8, t)
    for i, c in enumerate(cols):
        x = 67 + i * 18
        box(ax, x + 4, 7, 9, 6, f"e{i + 1}", t, color=t["c"][c], fs=8.6)
        arrow(ax, x + 8.5, 17.2, x + 8.5, 13.2, t, color=t["c"][c])
    ax.text(66, 2.5, "mean-pool token vector trong từng đoạn [aⱼ, bⱼ)", fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 8.8 Contextual Retrieval
@figure("contextual-retrieval", size=(8.8, 3.1))
def _(fig, t):
    a0 = fig.add_axes([0.0, 0.05, 0.45, 0.9]); a1 = fig.add_axes([0.56, 0.17, 0.43, 0.68])
    a0.set_xlim(0, 60); a0.set_ylim(0, 40); a0.axis("off")
    box(a0, 1, 26, 56, 10, "", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14))
    a0.text(3, 33, "context (LLM viết, 50–100 token):", fontsize=8.2, color=t["fg2"])
    a0.text(3, 28.5, "Đoạn này thuộc bài «Xuất hóa đơn PDF», mục Lỗi\nthường gặp; áp dụng gói Pro, bản 4.x.", fontsize=8.2, color=t["fg"])
    box(a0, 1, 13, 56, 11, "", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .1))
    a0.text(3, 21, "chunk gốc:", fontsize=8.2, color=t["fg2"])
    a0.text(3, 15.5, "Nếu nút này bị mờ, kiểm tra quyền\nBilling admin của tài khoản…", fontsize=8.2, color=t["fg"])
    a0.text(29, 7.5, "→ ghép rồi tạo cả embedding lẫn chỉ mục BM25", ha="center", fontsize=8.4, color=t["fg"])
    a0.text(29, 2.5, "LLM đọc toàn bộ tài liệu (prompt caching)", ha="center", fontsize=8, color=t["fg2"])

    labels = ["baseline", "+ contextual\nembeddings", "+ contextual\nBM25", "+ rerank"]
    vals = [5.7, 3.7, 2.9, 1.9]
    cols = [t["muted"], t["c"][BLUE], t["c"][AQUA], t["c"][VIOLET]]
    b = a1.bar(labels, vals, color=cols, width=0.6)
    for bb, v, r in zip(b, vals, ["", "−35%", "−49%", "−67%"]):
        a1.text(bb.get_x() + bb.get_width() / 2, v + 0.15, f"{v}%" + (f"\n{r}" if r else ""), ha="center", fontsize=8.4, color=t["fg"])
    a1.set_ylim(0, 7.4); a1.set_ylabel("tỷ lệ truy hồi thất bại top-20")
    a1.tick_params(axis="x", labelsize=7.8)
    a1.set_title("Số Anthropic báo cáo (9/2024)", fontsize=9.6)
    ygrid(a1, t)


# ---------------------------------------------------------------- 8.9 proposition
@figure("proposition", size=(8.6, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 34))
    box(ax, 2, 9, 40, 18, "", t)
    ax.text(4, 23.5, "Đoạn gốc", fontsize=8.6, color=t["fg2"], weight="bold")
    ax.text(4, 15, "Tính năng này chỉ có ở gói Pro.\nNó cho phép xuất tối đa 500\nhóa đơn mỗi lần.", fontsize=8.6, color=t["fg"])
    arrow(ax, 43, 18, 52, 18, t, lw=2)
    ax.text(47.5, 20.5, "LLM", ha="center", fontsize=8.2, color=t["fg2"])
    props = ["Tính năng xuất hóa đơn hàng loạt chỉ có ở gói Pro.",
             "Tính năng xuất hóa đơn hàng loạt cho phép xuất tối đa\n500 hóa đơn mỗi lần."]
    for i, p in enumerate(props):
        box(ax, 53, 19 - i * 11, 65, 9 if i else 8, p, t, color=t["c"][AQUA], fill=tint(t["c"][AQUA], t, .14), fs=8.5, ha="left")
    ax.text(2, 2.5, "Mỗi mệnh đề: một sự kiện, tự đủ nghĩa, đại từ được thay bằng danh từ đầy đủ.", fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 9.2 diff theo hash
@figure("hash-diff", size=(8.4, 2.3))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 30))
    n = 20; w = 5.2
    for row, (label, changed) in enumerate([("bản cũ", set()), ("bản mới", {7})]):
        y = 18 - row * 10
        ax.text(2, y + 3, label, fontsize=8.6, color=t["fg2"], va="center")
        for i in range(n):
            ch = i in changed
            ax.add_patch(Rectangle((14 + i * w, y), w - 0.6, 6,
                                   color=t["c"][ORANGE] if ch else t["c"][BLUE], alpha=1 if ch else 0.35, lw=0))
    ax.text(14 + 7 * w + 2.3, 5.5, "↑", ha="center", fontsize=10, color=t["c"][ORANGE])
    ax.text(14 + 7 * w + 2.3, 1.2, "1 hash mới → chỉ embed lại chunk này, xóa hash cũ", ha="center", fontsize=8.5, color=t["fg"])
    ax.text(2, 27.5, "So content_hash từng chunk: bài 20 chunk sửa một đoạn = 1 lần embed, không phải 20", fontsize=8.8,
            color=t["fg"], weight="bold")


# ---------------------------------------------------------------- 9.4 lineage xóa dữ liệu
@figure("deletion-lineage", size=(8.8, 3.4))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    box(ax, 2, 18, 24, 10, "Yêu cầu xóa /\nticket bị xóa", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .15), fs=9, weight="bold")
    items = [("1. Raw ticket + comment", "xóa"), ("2. Chunk, vector theo source_id", "tombstone → xóa vật lý"),
             ("3. Q/A trích & Q/A gộp cụm (lineage)", "tái tạo đại diện cụm"),
             ("4. BM25, cache embedding, semantic cache", "xóa"), ("5. Tập đánh giá, dữ liệu fine-tune", "xóa / tạo lại"),
             ("6. Log, trace · backup", "che PII sớm; backup hết hạn")]
    for i, (a, b) in enumerate(items):
        y = 40 - i * 7.2
        box(ax, 40, y, 50, 5.6, a, t, fs=8.4, ha="left", color=t["c"][BLUE] if i == 2 else None,
            fill=tint(t["c"][BLUE], t, .12) if i == 2 else None)
        ax.text(92, y + 2.8, b, fontsize=8.2, color=t["fg2"], va="center")
        arrow(ax, 26.3, 23, 39.6, y + 2.8, t, lw=1.1)
    ax.text(2, 4, "Bảng lineage cho biết\nmọi dẫn xuất của một nguồn", fontsize=8.4, color=t["fg2"])


if __name__ == "__main__":
    run("04")
