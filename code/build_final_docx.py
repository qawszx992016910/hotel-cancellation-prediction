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
            w = 13.5 if "retrain" in path.name else (11.5 if "demo" in path.name else 11)
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
replace_all("摘要、壹至陸章、附錄 C 程式", "摘要、壹至柒章、附錄 C 程式（含期末 code/final_01～06、booking_eval.py 與網頁展示）")
replace_all("組員重新執行 code/run_experiments.py、code/run_extra.py 與 code/run_overlap.py",
            "組員重新執行 code/run_experiments.py、code/run_extra.py、code/run_overlap.py 與期末 code/final_01～06_*.py、code/booking_eval.py；網頁單筆預測以 300 筆與 Python 結果比對")
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


# ---------- 8. 審查 F1 修正（期中留下的文字；見 drafts_final/審查紀錄_F1.md）----------
FIX = [
    ("以 E1 至 E4 四組實驗回答三個研究問題（第五、六節）",
     "以 E1 至 E4 四組期中實驗（第五、六節）與期末 rolling-origin 實驗（第八節）回答五個研究問題"),
    ("本研究的流程分為五個模組，名稱與本章各節對應。", "如圖 3-1 所示，本研究的流程分為五個模組，名稱與本章各節對應。"),
    ("所有步驟寫在 code/run_experiments.py（E1 至 E4 主實驗）", "期中步驟寫在 code/run_experiments.py（E1 至 E4 主實驗）"),
    ("執行方式與檔案連結列於附錄 C。",
     "期末步驟寫在 code/final_01_rolling_origin.py 至 code/final_06_contact_time.py，評估工具與網頁展示為 code/booking_eval.py 與 code/build_demo.py；執行方式與檔案連結列於附錄 C。"),
    ("如表 3-1 所示，取消訂單約佔四成", "如表 3-1 與圖 3-2 所示，取消訂單約佔四成"),
    ("（含 3 個衍生特徵）", "（含 3 個衍生特徵：抵達月份數值、總住宿晚數、總人數）"),
    ("三種模型使用相同的特徵與前處理（Logistic Regression 另加標準化），差異只在模型本身。本階段的超參數",
     "表 3-2 為期中的模型設定，期末選定值見表 4-6。三種模型使用相同的特徵與前處理（Logistic Regression 另加標準化），差異只在模型本身。期中階段的超參數"),
    ("E1 至 E3 皆採隨機分層切分", "表 3-3 列出期中四組實驗的設定。E1 至 E3 皆採隨機分層切分"),
    ("本研究以「取消」為正類（positive class）", "表 3-4 列出評估指標的定義與選用理由。本研究以「取消」為正類（positive class）"),
    ("本研究以驗證集提前停止控制樹的數量", "期中以驗證集提前停止控制樹的數量，期末改以 rolling-origin 驗證從網格選擇樹數（表 3-7）"),
    ("方案 B 在實際上線與定期重新訓練上都比本專題完整",
     "方案 B 在實際上線與自動化重新訓練上都比本專題完整（本專題只以離線方式模擬每季重新訓練）"),
    ("並量化洩漏欄位、重複資料與切分方式對效能估計的影響。",
     "並量化洩漏欄位、重複資料與切分方式對效能估計的影響；期末再以 rolling-origin 時間驗證與事先定義的選模規則檢驗模型對未來期間的表現，並將評估流程做成工具與網頁展示。"),
    ("五、預期貢獻", "五、研究貢獻"),
    ("說明本專題的需求、定位與風險。", "說明本專題的需求、定位與風險：需求優先序見表 5-1，SWOT 分析與策略見表 5-2、表 5-3，風險評估見表 5-4。"),
    ("逐筆辨識無訂金訂單的重要性高，因為取消率達 12.2% 至 40.0%",
     "逐筆辨識無訂金訂單的重要性高，因為各飯店的整體取消率達 12.2% 至 40.0%"),
    ("期末加入「以相同特徵分組」的切分方式，確保相同的列不會同時出現在訓練集與測試集",
     "期末以時間切分為主要評估（相同列抵達日相同，不跨期），bootstrap 以重複列群組為單位"),
    ("調參與門檻只用驗證集；測試集只在最後評估一次",
     "調參與門檻只用驗證集；正式方案在看期末測試結果前定案（此測試期期中已使用，見參之八）"),
    ("依表 7-2，W13 前完成模型定版；", "期末先完成模型定版（表 7-2）；"),
]
for old, new in FIX:
    replace_all(old, new)

# 預期貢獻補期末貢獻段
anchor = find_p(lambda t, el: t.startswith("在實務面，本研究比較類別權重"))
insert_after(anchor, [body_p(
    "期末的貢獻對應 RQ4、RQ5 與第陸章：（1）在公開資料上示範事先寫下的選模規則，並報告規則選出的模型在下一年被推翻，以及規則判定對驗證期組成的敏感度；"
    "（2）指出每日聯繫名單若不排除聯繫時已取消的訂單，命中率會由 36.1% 被高估為 84.3%，並以聯繫時點仍有效的訂單評估；"
    "（3）公開評估工具，並示範統計式洩漏檢查會漏掉影響最大的 country，必須搭配欄位確定時點清單（對應 RQ4、RQ5）。")])

# 表 2-3 前加引用句
cap = find_p(lambda t, el: t.startswith("表 2-3　"))
cap.addprevious(body_p("表 2-3 整理前述文獻顯示的四項研究缺口，以及本研究對應的作法。"))

# 表 2-2：本專題「每日或定期自動重新訓練」○ → ◐
hit = 0
for tb in doc.tables:
    for row in tb.rows:
        if row.cells[0].text.strip() == "每日或定期自動重新訓練":
            c = row.cells[2].paragraphs[0]
            assert c.text.strip() == "○"
            replace_in_para(c, lambda s: s.replace("○", "◐")); hit += 1
    if tb.rows[0].cells[0].text.strip() == "功能項目":
        for p in tb._tbl.itersiblings():
            if p.tag == qn("w:p") and ptext(p).startswith("註："):
                replace_in_para(P(p, doc), lambda s: s.replace("－ 公開資料未說明。", "－ 公開資料未說明。本專題的重新訓練為離線模擬每季重新訓練（第肆章第三節），未自動化。", 1))
                break
assert hit == 1

doc.save(OUT)
print("saved", OUT)
