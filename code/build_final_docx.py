"""期末報告：以 期中報告_訂房取消預測_引用修正.docx 為底，套入 drafts_final/*.md，輸出 期末報告_訂房取消預測.docx
（期中 docx 是 build_docx.py 輸出後手改的，故不重建，只做定點插入與替換）"""
import copy
import re
import sys
from pathlib import Path

import docx
from docx.shared import Cm
from docx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "期中報告_訂房取消預測_引用修正.docx"
OUT = ROOT / "期末報告_訂房取消預測.docx"
DF = ROOT / "drafts_final"
TEXT_W = 9026
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

doc = docx.Document(SRC)
body = doc.element.body
P = docx.text.paragraph.Paragraph
tpl_tbl = copy.deepcopy(doc.tables[12]._tbl)


def ptext(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def find_p(pred, start=None):
    for el in body.iterchildren():
        if el.tag == qn("w:p") and pred(ptext(el).strip(), el):
            return el
    raise KeyError


def heading_el(text):
    return find_p(lambda t, el: t == text and el.find(".//" + qn("w:pStyle")) is not None)


def replace_in_para(p, old_new_fn):
    runs = p.runs
    full = "".join(r.text for r in runs)
    new = old_new_fn(full)
    if new == full:
        return False
    # 找出第一個與最後一個不同的位置，只改動中間那段，保留其他 run 格式
    a = 0
    while a < min(len(full), len(new)) and full[a] == new[a]:
        a += 1
    b = 0
    while b < min(len(full), len(new)) - a and full[-1 - b] == new[-1 - b]:
        b += 1
    old_end = len(full) - b
    pos, si, ei = 0, None, None
    bounds = []
    for k, r in enumerate(runs):
        bounds.append((pos, pos + len(r.text)))
        pos += len(r.text)
    for k, (s, e) in enumerate(bounds):
        if si is None and s <= a < e or (si is None and a == e and k == len(runs) - 1):
            si = k
        if s < old_end <= e or (old_end == s == e):
            ei = k
    if ei is None or ei < si:
        ei = si
    s0 = bounds[si][0]; e1 = bounds[ei][1]
    runs[si].text = full[s0:a] + new[a:len(new) - b] + full[old_end:e1]
    for k in range(si + 1, ei + 1):
        runs[k].text = ""
    assert "".join(r.text for r in p.runs) == new, (full[:40], new[:40])
    return True


# ---------- 建構元素（同 build_docx.py 的格式）----------
def E(tag, **attrs):
    e = etree.Element(qn(tag))
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def sub(parent, tag, **attrs):
    e = E(tag, **attrs); parent.append(e); return e


def add_runs(p, text, bold=False, size=None, italic=False):
    text = text.replace("`", "")
    for part in re.split(r"(\*[^*]+\*)", text):
        if not part:
            continue
        it = italic
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, it = part[1:-1], True
        r = sub(p, "w:r"); rpr = sub(r, "w:rPr")
        if bold:
            sub(rpr, "w:b"); sub(rpr, "w:bCs")
        if it:
            sub(rpr, "w:i"); sub(rpr, "w:iCs")
        if size:
            sub(rpr, "w:sz", **{"w:val": size}); sub(rpr, "w:szCs", **{"w:val": size})
        if len(rpr) == 0:
            r.remove(rpr)
        t = sub(r, "w:t"); t.text = part
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def para(text="", style=None, jc=None, ind=None, spacing=None, bold=False, size=None, keep_next=False):
    p = E("w:p"); ppr = sub(p, "w:pPr")
    if style:
        sub(ppr, "w:pStyle", **{"w:val": style})
    if keep_next:
        sub(ppr, "w:keepNext")
    if spacing:
        sub(ppr, "w:spacing", **spacing)
    if ind:
        sub(ppr, "w:ind", **ind)
    if jc:
        sub(ppr, "w:jc", **{"w:val": jc})
    if text:
        add_runs(p, text, bold=bold, size=size)
    return p


_bm = [5000]


def heading(text, level):
    p = para(style=str(level))
    _bm[0] += 1
    sub(p, "w:bookmarkStart", **{"w:id": _bm[0], "w:name": f"_Toc8{_bm[0]}"})
    add_runs(p, text)
    sub(p, "w:bookmarkEnd", **{"w:id": _bm[0]})
    return p


def body_p(text):
    return para(text, ind={"w:firstLine": 480}, jc="both")


def list_p(text):
    return para(text, ind={"w:left": 480, "w:hanging": 360}, jc="both")


def label_p(label, rest):
    p = para(ind={"w:left": 480}, jc="both"); add_runs(p, label, bold=True); add_runs(p, rest); return p


def tcap(text):
    return para(text, jc="center", spacing={"w:before": 200, "w:after": 80}, bold=True, size=22, keep_next=True)


def fcap(text):
    return para(text, jc="center", spacing={"w:before": 80, "w:after": 200}, bold=True, size=22)


def note(text):
    return para(text, spacing={"w:before": 60, "w:after": 160}, size=20)


def picture(path, width_cm):
    tmp = doc.add_paragraph(); tmp.alignment = 1; tmp.paragraph_format.keep_with_next = True
    tmp.add_run().add_picture(str(path), width=Cm(width_cm))
    p = tmp._p; p.getparent().remove(p); return p


NUMLIKE = re.compile(r"^[\s−\-+＋0-9.,±%\[\]（）()／/：:~～至＝<＞]*$|^(採用|－)$")


def make_table(rows):
    ncol = len(rows[0])
    L = lambda s: sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)
    lens = [max(min(L(r[c]), 60) for r in rows) + 4 for c in range(ncol)]
    widths = [max(int(TEXT_W * l / sum(lens)), 700) for l in lens]
    widths[-1] += TEXT_W - sum(widths)
    tbl = copy.deepcopy(tpl_tbl)
    for tr in tbl.findall("w:tr", NS):
        tbl.remove(tr)
    grid = tbl.find("w:tblGrid", NS)
    for g in list(grid):
        grid.remove(g)
    for w in widths:
        sub(grid, "w:gridCol", **{"w:w": w})
    for ri, row in enumerate(rows):
        tr = sub(tbl, "w:tr"); trpr = sub(tr, "w:trPr"); sub(trpr, "w:cantSplit")
        if ri == 0:
            sub(trpr, "w:tblHeader")
        for ci, cell in enumerate(row):
            hdr = ri == 0
            tc = sub(tr, "w:tc"); tcpr = sub(tc, "w:tcPr")
            sub(tcpr, "w:tcW", **{"w:w": widths[ci], "w:type": "dxa"})
            b = sub(tcpr, "w:tcBorders")
            for side in ("top", "left", "bottom", "right"):
                sub(b, f"w:{side}", **{"w:val": "single", "w:sz": 4, "w:space": 0, "w:color": "000000"})
            if hdr:
                sub(tcpr, "w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": "E7E6E6"})
            m = sub(tcpr, "w:tcMar")
            for side, v in (("top", 60), ("left", 100), ("bottom", 60), ("right", 100)):
                sub(m, f"w:{side}", **{"w:w": v, "w:type": "dxa"})
            sub(tcpr, "w:vAlign", **{"w:val": "center"})
            seg = cell.strip()
            center = hdr or (NUMLIKE.match(seg) is not None and seg != "") or L(seg) <= 8
            p = para(jc="center" if center else "left", spacing={"w:after": 0, "w:line": 276, "w:lineRule": "auto"})
            add_runs(p, seg, bold=hdr, size=21)
            tc.append(p)
    return tbl


def md_to_elems(md):
    out = []
    lines = re.sub(r"<!--.*?-->", "", md, flags=re.S).split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1; continue
        if ln.startswith("|"):
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i]); i += 1
            rows = [[c.strip() for c in b.strip().strip("|").split("|")] for b in blk]
            rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r if c)]
            out.append(make_table(rows))
            if not (i + 1 < len(lines) and lines[i + 1].startswith("註：")):
                out.append(para(spacing={"w:after": 0, "w:line": 120, "w:lineRule": "exact"}))
            continue
        if ln.startswith("# "):
            out.append(heading(ln[2:].strip(), 1))
        elif ln.startswith("## "):
            out.append(heading(ln[3:].strip(), 2))
        elif ln.startswith("### "):
            out.append(heading(ln[4:].strip(), 3))
        elif re.match(r"^表 [0-9A-Z]-\d+　", ln):
            out.append(tcap(ln))
        elif re.match(r"^圖 \d-\d+　", ln):
            m = re.search(r"（(results/[^）]+\.png)）", ln)
            path = ROOT / m.group(1)
            w = 15.5 if "retrain" in path.name else (14 if "demo" in path.name else 12)
            out.append(picture(path, w))
            out.append(fcap(re.sub(r"（results/[^）]+）", "", ln)))
        elif ln.startswith("註："):
            out.append(note(ln))
        elif re.match(r"^- RQ\d：", ln):
            out.append(label_p(ln[2:6], ln[6:]))
        elif re.match(r"^\d+\. ", ln):
            out.append(list_p(ln))
        else:
            out.append(body_p(ln))
        i += 1
    return out


def insert_before(anchor, elems):
    for e in elems:
        anchor.addprevious(e)


def insert_after(anchor, elems):
    for e in reversed(elems):
        anchor.addnext(e)


def set_text(el, text):
    """保留第一個 run 的格式，換掉整段文字"""
    rs = el.findall(qn("w:r"))
    for r in rs[1:]:
        el.remove(r)
    ts = rs[0].findall(qn("w:t"))
    for t in ts[1:]:
        rs[0].remove(t)
    ts[0].text = text
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def replace_all(old, new):
    n = 0
    containers = [doc.element.body] + [s.header._element for s in doc.sections] + [s.footer._element for s in doc.sections]
    for c in containers:
        for el in c.iter(qn("w:p")):
            if old in ptext(el):
                n += replace_in_para(P(el, doc), lambda t: t.replace(old, new))
    assert n, old
    return n


# ---------- 1. 封面、頁首 ----------
replace_all("期中專題報告", "期末專題報告")
replace_all("Midterm Project Report", "Final Project Report")

# ---------- 2. 摘要 ----------
f0 = (DF / "F0_摘要與RQ.md").read_text(encoding="utf-8")
sec = lambda name: f0.split(f"# {name}")[1].split("\n# ")[0].strip().split("\n\n")
zh, en = sec("摘要"), sec("Abstract")
zh_h, en_h = heading_el("摘要"), heading_el("Abstract")
p_zh = zh_h.getnext(); set_text(p_zh, zh[0])
kw = p_zh.getnext(); rs = kw.findall(qn("w:r")); [kw.remove(r) for r in rs[2:]]
rs[1].find(qn("w:t")).text = zh[1].split("：", 1)[1]
p_en = en_h.getnext(); set_text(p_en, en[0])
kw = p_en.getnext(); rs = kw.findall(qn("w:r")); [kw.remove(r) for r in rs[2:]]
rs[1].find(qn("w:t")).text = " " + en[1].split(":", 1)[1].strip()

# ---------- 3. RQ4、RQ5 ----------
rq3 = find_p(lambda t, el: t.startswith("RQ3："))
rq_md = f0.split("# RQ 補充（接在 RQ3 之後）")[1]
insert_after(rq3, md_to_elems(rq_md))

# ---------- 4. 參之八（插在 肆 之前）、肆 改名與期末結果 ----------
h4 = heading_el("肆、初步結果")
insert_before(h4, md_to_elems((DF / "F3_期末方法.md").read_text(encoding="utf-8")))
replace_in_para(P(h4, doc), lambda t: t.replace("肆、初步結果", "肆、研究結果"))
h5 = heading_el("伍、專題規劃分析")
insert_before(h5, md_to_elems((DF / "F4_期末結果.md").read_text(encoding="utf-8")))

# ---------- 5. 以 陸、柒 取代期中 陸 ----------
h6 = heading_el("陸、結論與後續規劃")
href = heading_el("參考文獻")
el = h6
while el is not href:
    nxt = el.getnext(); body.remove(el); el = nxt
insert_before(href, md_to_elems((DF / "F6_系統與結論.md").read_text(encoding="utf-8")))

# ---------- 6. 參考文獻：holidays 套件 ----------
yoo = find_p(lambda t, el: t.startswith("Yoo, M."))
newref = copy.deepcopy(yoo)
runs = newref.findall(qn("w:r"))
plain = copy.deepcopy(next(r.find(qn("w:rPr")) for r in runs if r.find(qn("w:rPr")) is not None and r.find(qn("w:rPr")).find(qn("w:i")) is None)) if any(
    r.find(qn("w:rPr")) is not None and r.find(qn("w:rPr")).find(qn("w:i")) is None for r in runs) else None
ital = copy.deepcopy(next(r.find(qn("w:rPr")) for r in runs if r.find(qn("w:rPr")) is not None and r.find(qn("w:rPr")).find(qn("w:i")) is not None))
for r in runs:
    newref.remove(r)
for seg, it in [("Vacanza Team. (n.d.). ", False), ("holidays", True),
                (" (Version 0.108) [Computer software]. Retrieved November 11, 2026, from https://github.com/vacanza/holidays", False)]:
    r = E("w:r")
    rp = ital if it else plain
    if rp is not None:
        r.append(copy.deepcopy(rp))
    t = sub(r, "w:t"); t.text = seg; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    newref.append(r)
yoo.addprevious(newref)

# ---------- 7. 交互參照、表 3-5、附錄 ----------
replace_all("（見陸之二）", "（見參之八）")
replace_all("依表 6-2 時程", "依表 7-2")
replace_all("pandas 3.0.6（McKinney, 2010）", "pandas 3.0.6（McKinney, 2010）、holidays 0.108（Vacanza Team, n.d.）")
replace_all("42（E4、bootstrap、錯誤分析）", "42（E4、bootstrap、錯誤分析）；期末 LightGBM 42、43、44")
replace_all("摘要、壹至陸章、附錄 C 程式", "摘要、壹至柒章、附錄 C 程式（含期末 code/final_01～04、booking_eval.py 與網頁展示）")
replace_all("組員重新執行 code/run_experiments.py、code/run_extra.py 與 code/run_overlap.py",
            "組員重新執行 code/run_experiments.py、code/run_extra.py、code/run_overlap.py 與期末 code/final_01～04_*.py、code/booking_eval.py；網頁單筆預測以 300 筆與 Python 結果比對")
replace_all("依序執行即重建 results/，CPU 約 30 至 60 分鐘", "依序執行即重建 results/，CPU 約 30 至 60 分鐘；期末第 6 至 8 節約 20 分鐘")
replace_all("期中未製作", "未製作；改以網頁展示（見下列）")
tc = doc.tables[-2]
base_row = tc.rows[-1]._tr
for item, link, desc in [
    ("評估工具", "https://github.com/qawszx992016910/hotel-cancellation-prediction/blob/main/code/booking_eval.py",
     "python code/booking_eval.py --data <CSV> --out <資料夾>；輸出 report.html（圖內嵌單一檔案）與各項 csv"),
    ("網頁展示", "https://github.com/qawszx992016910/hotel-cancellation-prediction/blob/main/demo/index.html",
     "單一 HTML 檔，下載後以瀏覽器開啟，不需伺服器；由 python code/build_demo.py 產生"),
]:
    tr = copy.deepcopy(base_row)
    for cell_tc, val in zip(tr.findall(qn("w:tc")), (item, link, desc)):
        ps = cell_tc.findall(qn("w:p"))
        for extra in ps[1:]:
            cell_tc.remove(extra)
        rs = ps[0].findall(qn("w:r"))
        for r in rs[1:]:
            ps[0].remove(r)
        rs[0].find(qn("w:t")).text = val
    base_row.addnext(tr); base_row = tr

doc.save(OUT)
print("saved", OUT)
