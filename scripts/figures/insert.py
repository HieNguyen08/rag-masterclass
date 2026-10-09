"""
Chèn khối hình vào file Markdown của module.

Dùng:  python insert.py <spec.py>
spec.py định nghĩa:
    MD = "docs/03-embedding-va-bieu-dien.md"; MODULE = "03"; PREFIX = "3"
    FIGS = [ (after_substring, "ten-hinh" | {"mermaid": "..."}, "Chú thích"), ... ]
Mỗi hình được chèn ngay sau dòng (duy nhất) chứa after_substring, theo thứ tự xuất hiện
trong file; số hình (Hình 3.1, 3.2, …) đánh theo vị trí. Chạy lại an toàn: các khối
cũ (đánh dấu <!-- fig:... -->) bị gỡ trước khi chèn.
"""
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def block(module, name, num, cap):
    if isinstance(name, dict):
        key = name.get("id", "mermaid")
        return (f"<!-- fig:{key} -->\n```mermaid\n{name['mermaid'].strip()}\n```\n\n"
                f'<p class="fig-caption">Hình {num} — {cap}</p>\n<!-- /fig -->')
    src = f"assets/figures/{module}/{name}"
    alt = re.sub(r"[\[\]*_`$]", "", cap.split(".")[0])
    return (f"<!-- fig:{name} -->\n<figure markdown=\"span\">\n"
            f"  ![{alt}]({src}.light.svg#only-light){{ loading=lazy }}\n"
            f"  ![{alt}]({src}.dark.svg#only-dark){{ loading=lazy }}\n"
            f"  <figcaption>Hình {num} — {cap}</figcaption>\n</figure>\n<!-- /fig -->")


def main(spec_path):
    spec = runpy.run_path(spec_path)
    md = ROOT / spec["MD"]
    text = md.read_text(encoding="utf-8")
    text = re.sub(r"\n\n<!-- fig:[^>]*-->.*?<!-- /fig -->\n", "\n", text, flags=re.S)
    lines = text.split("\n")
    places = []
    for after, name, cap in spec["FIGS"]:
        hits = [i for i, l in enumerate(lines) if after in l]
        if len(hits) != 1:
            sys.exit(f"Neo không duy nhất ({len(hits)} kết quả): {after!r}")
        if hits[0] + 1 < len(lines) and lines[hits[0] + 1].strip():
            sys.exit(f"Dòng sau neo phải trống: {after!r}")
        places.append((hits[0], name, cap))
    places.sort(key=lambda p: p[0])
    out, k = [], 0
    for i, line in enumerate(lines):
        out.append(line)
        for j, (pos, name, cap) in enumerate(places):
            if pos == i:
                num = f"{spec['PREFIX']}.{j + 1}"
                out += ["", block(spec["MODULE"], name, num, cap)]
                k += 1
    md.write_text("\n".join(out), encoding="utf-8")
    print(f"{md.name}: chèn {k} hình")


if __name__ == "__main__":
    main(sys.argv[1])
