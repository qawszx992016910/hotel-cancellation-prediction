"""以 期中Proposal口試範本_前三章.pptx 為底，填入本專題內容。內容皆取自 drafts/ 與 results/。"""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.oxml.ns import qn
from PIL import Image

SRC = "期中Proposal口試範本_前三章.pptx"
OUT = "AI期中Proposal_第X組_訂房取消預測.pptx"
DARK = "1C2833"

prs = Presentation(SRC)

# ---------- 刪除第 1 頁（使用說明） ----------
sldIdLst = prs.slides._sldIdLst
first = sldIdLst[0]
prs.part.drop_rel(first.rId)
sldIdLst.remove(first)
S = list(prs.slides)          # S[0] = 原第 2 頁（封面）


def slide(n):                 # n = 範本原頁碼
    return S[n - 2]


def shp(sl, name):
    for s in sl.shapes:
        if s.name == name:
            return s
    raise KeyError(name)


def _style_run(r, text, bold=None, size=None, color=None, italic=False):
    r.find(qn("a:t")).text = text
    rPr = r.find(qn("a:rPr"))
    if rPr is None:
        rPr = r.makeelement(qn("a:rPr"), {})
        r.insert(0, rPr)
    if not italic and "i" in rPr.attrib:
        del rPr.attrib["i"]
    if italic:
        rPr.set("i", "1")
    if bold is not None:
        if bold:
            rPr.set("b", "1")
        elif "b" in rPr.attrib:
            del rPr.attrib["b"]
    if size:
        rPr.set("sz", str(int(size * 100)))
    if color:
        sf = rPr.find(qn("a:solidFill"))
        if sf is not None:
            for c in list(sf):
                sf.remove(c)
        else:
            sf = rPr.makeelement(qn("a:solidFill"), {})
            rPr.insert(0, sf)
        clr = sf.makeelement(qn("a:srgbClr"), {"val": color})
        sf.append(clr)


def fill(sh, paras, size=None, color=DARK, txBody=None):
    """paras: list；每項為 str 或 [(text, {bold,size,color}), ...]。
    第 i 段沿用範本第 min(i, n-1) 段的格式。"""
    tb = txBody if txBody is not None else sh.text_frame._txBody
    ps = tb.findall(qn("a:p"))
    protos = [copy.deepcopy(p) for p in ps]
    for p in ps:
        tb.remove(p)
    for i, para in enumerate(paras):
        proto = protos[min(i, len(protos) - 1)]
        p = copy.deepcopy(proto)
        runs = p.findall(qn("a:r"))
        rproto = runs[-1] if runs else None
        if runs and len(runs) > 1 and isinstance(para, list) and len(para) > 1:
            rprotos = runs
        else:
            rprotos = [rproto]
        for r in runs:
            p.remove(r)
        for br in p.findall(qn("a:br")):
            p.remove(br)
        for extra in p.findall(qn("a:pPr"))[1:]:
            p.remove(extra)
        if isinstance(para, str):
            para = [(para, {})]
        end = p.find(qn("a:endParaRPr"))
        for j, (text, opt) in enumerate(para):
            base = rprotos[min(j, len(rprotos) - 1)]
            if base is None:
                base = p.makeelement(qn("a:r"), {})
                base.append(base.makeelement(qn("a:rPr"), {"lang": "zh-TW"}))
                base.append(base.makeelement(qn("a:t"), {}))
            r = copy.deepcopy(base)
            _style_run(r, text, bold=opt.get("bold"), size=opt.get("size", size),
                       color=opt.get("color", color), italic=opt.get("italic", False))
            if end is not None:
                end.addprevious(r)
            else:
                p.append(r)
        tb.append(p)


def remove(sh):
    sh._element.getparent().remove(sh._element)


def picture(sl, box_name, img, extra_remove=(), pad=0.1):
    box = shp(sl, box_name)
    x, y, w, h = box.left, box.top, box.width, box.height
    remove(box)
    for n in extra_remove:
        remove(shp(sl, n))
    iw, ih = Image.open(img).size
    pw, ph = w - Inches(2 * pad), h - Inches(2 * pad)
    sc = min(pw / iw, ph / ih)
    W, H = int(iw * sc), int(ih * sc)
    sl.shapes.add_picture(img, x + (w - W) // 2, y + (h - H) // 2, W, H)


def solid_border(sh):
    ln = sh._element.spPr.find(qn("a:ln"))
    if ln is not None:
        d = ln.find(qn("a:prstDash"))
        if d is not None:
            d.set("val", "solid")


def set_rows(tbl, n):
    trs = tbl._tbl.findall(qn("a:tr"))
    while len(trs) > n:
        tbl._tbl.remove(trs[-1]); trs = trs[:-1]
    while len(trs) < n:
        new = copy.deepcopy(trs[-1]); trs[-1].addnext(new); trs.append(new)


def table(sh, rows, size=12, row_h=None, col_w=None, header=None):
    t = sh.table
    set_rows(t, len(rows) + 1)
    ncol = len(header) if header else len(rows[0])
    grid = t._tbl.find(qn("a:tblGrid"))
    while len(grid.findall(qn("a:gridCol"))) < ncol:
        grid.append(copy.deepcopy(grid.findall(qn("a:gridCol"))[-1]))
        for tr in t._tbl.findall(qn("a:tr")):
            tr.append(copy.deepcopy(tr.findall(qn("a:tc"))[-1]))
    if col_w:
        for c, w in zip(t.columns, col_w):
            c.width = Inches(w)
        sh.width = Inches(sum(col_w))
    if header:
        for j, v in enumerate(header):
            fill(None, [v], txBody=t.cell(0, j).text_frame._txBody, color=None)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            fill(None, v if isinstance(v, list) else [v], size=size,
                 txBody=t.cell(i, j).text_frame._txBody)
    if row_h:
        hs = row_h if isinstance(row_h, list) else [row_h] * (len(rows) + 1)
        for r, hh in zip(t.rows, hs):
            r.height = Inches(hh)
        sh.height = Inches(sum(hs))


def add_text(sl, x, y, w, h, paras, like):
    """複製 like 文字框到新位置並填入內容。"""
    el = copy.deepcopy(like._element)
    sl.shapes._spTree.append(el)
    new = sl.shapes[-1]
    new.left, new.top, new.width, new.height = Inches(x), Inches(y), Inches(w), Inches(h)
    fill(new, paras)
    return new


def notes(sl, text):
    sl.notes_slide.notes_text_frame.text = text


B = {"bold": True, "color": None}
TITLE_ZH = "飯店訂房取消預測：排除資料洩漏與時間外推之評估"
TITLE_EN = "Hotel Booking Cancellation Prediction: A Leakage-Aware Evaluation with Temporal Validation"

# ---------- 2 封面 ----------
s = slide(2)
fill(shp(s, "title-zh"), [TITLE_ZH], color=None)
fill(shp(s, "title-en"), [TITLE_EN], color=None, size=22)

# ---------- 5 研究背景 ----------
s = slide(5)
fill(shp(s, "Text 2"), [
    "客房是無法儲存的易逝性商品；訂房臨近入住才取消，空房常來不及再售出（António et al., 2017a）",
    "葡萄牙飯店的取消率介於 11.8% 至 40.0%（António et al., 2017a, 2019a）；本研究資料 119,390 筆訂單中有 44,224 筆取消（37.0%）",
    "以 PMS 訂單資料訓練 XGBoost，已部署於兩家飯店每日預測可能取消的訂單（António et al., 2017b）",
])
picture(s, "placeholder-box", "results/fig_cancel_rates.png", ["Text 4"])
fill(shp(s, "source"), ["來源：António et al. (2017a, 2017b, 2019a, 2019b)；圖為本組依上述文獻數值與本研究資料繪製"], color=None)

# ---------- 6 問題陳述 ----------
s = slide(6)
fill(shp(s, "Text 5"), ["傳統依歷史取消率估總量再超賣，無法指出哪一筆會取消（António et al., 2017a）"])
fill(shp(s, "Text 9"), ["隨機切分 Accuracy 最高 98.6%；部署後對未來訂單 Precision 只有 0.248、0.319（António et al., 2017a, 2017b）"])
fill(shp(s, "Text 13"), ["公開資料仍保留訂單狀態、國籍等事後欄位；照原樣建模會高估效能，且缺少可重現的比較"])
fill(shp(s, "Text 15"), [[("核心問題　", B), ("在公開訂房資料上排除洩漏欄位、改以時間切分評估時，模型效能與優劣排序會如何改變？", {})]])

# ---------- 7 研究目的與 RQ ----------
s = slide(7)
fill(shp(s, "Text 3"), [[("研究目的　", B), ("以公開訂房資料建立排除洩漏欄位的取消預測流程，比較 LR、RF 與 LightGBM，並量化洩漏欄位、重複列與切分方式對效能估計的影響", {"size": 18, "color": None})]])
fill(shp(s, "Text 7"), ["LightGBM 的 F1 與 AUC 在隨機切分與時間切分下，是否都優於 LR 與 RF 兩個 baseline？"])
fill(shp(s, "Text 11"), ["取消約佔 37% 時，類別權重對取消類別的 Recall 與 F1 有何影響？與調整門檻有何差異？"])
fill(shp(s, "Text 15"), ["加回入住時才確定的欄位或保留重複列，效能會高估多少？錯誤案例有何共同特徵？"])

# ---------- 8 範圍與限制 ----------
s = slide(8)
fill(shp(s, "Text 4"), [
    [("資料範圍：", B), ("Kaggle Hotel Booking Demand，葡萄牙度假與城市飯店各一家，抵達日 2015-07-01 至 2017-08-31", {})],
    [("任務範圍：", B), ("二元分類（取消含 no-show），預測時點在入住前", {})],
    [("應用對象：", B), ("使用此資料或相同欄位結構的研究者與學生；導入模型前需驗證的飯店", {})],
])
fill(shp(s, "Text 7"), [
    [("資料面：", B), ("只有兩家飯店且早於疫情；沒有訂單編號，31,980 列完全重複無法判斷是否為團體訂單", {})],
    [("方法面：", B), ("欄位取自抵達前一天的紀錄，booking_changes 可能含事後資訊（E3 檢驗）；三種模型未系統性調參", {})],
    [("不處理：", B), ("營收損失、超賣最佳化與動態定價，需要本資料沒有的成本與價格資料", {})],
])

# ---------- 9 預期貢獻 ----------
s = slide(9)
fill(shp(s, "Text 4"), [
    "主要貢獻是評估流程：列出每個排除欄位與依據，前處理參數只以訓練集擬合，程式公開於 GitHub",
    "在同一資料上同時報告隨機切分、時間切分與去重結果，分辨模型本身與評估設計造成的差異",
])
fill(shp(s, "Text 7"), [
    "比較類別權重與門檻調整對取消 Recall 的影響",
    "依訂金類型與市場區隔分析錯誤，指出現有欄位難以辨識的訂單",
    "說明導入時應以時間切分驗證，並依成本調整門檻",
])
fill(shp(s, "Text 8"), ["對應研究問題：RQ1、RQ3"], color=None)
fill(shp(s, "Text 9"), ["對應研究問題：RQ2、RQ3"], color=None)

# ---------- 11 文獻架構 ----------
s = slide(11)
fill(shp(s, "Text 3"), ["飯店訂房取消預測"], color=None)
fill(shp(s, "Text 8"), ["模型方法"], color=None)
fill(shp(s, "Text 13"), ["評估方法"], color=None)
fill(shp(s, "Text 18"), ["飯店應用研究"], color=None)
fill(shp(s, "Text 10"), ["Logistic Regression、Random Forest", "梯度提升：XGBoost、LightGBM",
                         [("文獻：Breiman, 2001；Chen & Guestrin, 2016；Ke et al., 2017", {"size": 13})]])
fill(shp(s, "Text 15"), ["資料洩漏", "時間外推評估", "類別權重與門檻",
                         [("文獻：Kapoor & Narayanan, 2023；António et al., 2019a", {"size": 13})]])
fill(shp(s, "Text 20"), ["葡萄牙飯店 PMS 取消預測", "疫情前後的模型更新",
                         [("文獻：António et al., 2017a, 2017b, 2019a；Silvestre et al., 2026", {"size": 13})]])

# ---------- 12 理論基礎 ----------
s = slide(12)
fill(shp(s, "Text 2"), [
    [("原理：", B), ("梯度提升樹依序加入新樹，每棵擬合前一輪損失的負梯度", {})],
    [("LightGBM：", B), ("直方圖分箱、GOSS、EFB、逐葉生長，訓練速度快（Ke et al., 2017）", {})],
    [("適用理由：", B), ("欄位以類別與計數為主，樹模型在表格資料上普遍優於深度學習（Grinsztajn et al., 2022）", {})],
    [("限制與因應：", B), ("逐葉生長易過度擬合 → 驗證集提前停止；可解釋性低 → 特徵重要度與分群錯誤分析", {})],
])
picture(s, "placeholder-box", "results/fig_methods.png", ["Text 4"])
fill(shp(s, "source"), ["來源：本組依 Breiman (2001)、Chen & Guestrin (2016)、Ke et al. (2017) 自行繪製"], color=None)

# ---------- 13 相關研究比較 ----------
s = slide(13)
tb = [x for x in s.shapes if x.has_table][0]
table(tb, [
    ["António et al. (2017a)", "Algarve 4 家度假飯店 PMS", "BDT、Decision Forest 等 5 種", "分層隨機切分＋k-fold", "Accuracy 皆 > 90%，最高 98.6%", "只用隨機切分；未納入 LR"],
    ["António et al. (2017b)", "度假、城市飯店各 1 家", "XGBoost，每日重新訓練", "月份分塊後分層切分；刪 Country 等", "AUC 0.877／0.922；未來訂單 Precision 0.248／0.319", "A/B 測試無顯著差異；未公開"],
    ["António et al. (2019a)", "8 家飯店 PMS＋天氣、假日等", "XGBoost", "訓練期之後的訂單測試", "AUC 最高 0.9307；R1 Precision 0.754→0.461", "需外部資料；未報告 Recall"],
    ["Silvestre et al. (2026)", "4 家飯店，疫情前後", "XGBoost，靜態 vs 滑動視窗", "依時間切分；刪 Country、DepositType", "R2 AUC 0.70→0.85（滑動視窗）", "部分飯店測試集比例偏低"],
], size=12, header=["作者（年份）", "資料集", "方法", "切分與洩漏處理", "主要結果", "限制"],
   row_h=[0.45, 0.8, 0.8, 0.75, 0.75])
fill(shp(s, "Text 3"), [[("歸納　", B), ("使用時間評估的研究只用 XGBoost，比較多種演算法的研究只用隨機切分；各研究刪欄理由不一且未量化影響；資料與程式皆未公開", {})]])

# ---------- 14 現有方案比較 ----------
s = slide(14)
tb = [x for x in s.shapes if x.has_table][0]
table(tb, [
    ["逐筆訂單取消預測", "5", "●", "－", "●", "○"],
    ["排除入住後才確定的欄位", "5", "●", "－", "●", "○"],
    ["量化洩漏欄位與評估方式影響", "4", "●", "－", "○", "○"],
    ["以時間順序評估模型", "5", "●", "－", "◐", "◐"],
    ["每日或定期自動重新訓練", "4", "○", "－", "●", "○"],
    ["於飯店上線並驗證成效", "5", "○", "◐", "◐", "●"],
    ["將取消納入超賣與需求決策", "4", "○", "◐", "◐", "●"],
    ["資料與程式公開、可重現", "3", "●", "○", "○", "○"],
], size=13, header=["功能項目", "重要性", "本專題", "方案 A", "方案 B", "方案 C"], row_h=0.52)
fill(shp(s, "Text 4"), ["● 完整支援", "◐ 部分支援", "○ 不支援", "－ 公開資料未說明", "",
                        [("方案 A：", B), ("IDeaS G3 RMS（商用營收管理系統）", {})],
                        [("方案 B：", B), ("António et al. (2017b) 部署於兩家飯店之學術原型", {})],
                        [("方案 C：", B), ("依歷史取消率超賣＋嚴格取消政策", {})]], size=14)
fill(shp(s, "source"), ["來源：IDeaS 官網 https://ideas.com/revenue-management；SoftwareOne IDeaS RMS 產品頁（2026 年 11 月 3 日查詢）；António et al. (2017a, 2017b)；重要性為本組依文獻判斷"], color=None)

# ---------- 15 研究缺口（4 項） ----------
s = slide(15)
gaps = [
    ("G1", "時間評估的研究只用 XGBoost，多演算法比較只用隨機切分", "依據：表 2-1；António et al. (2017a, 2017b, 2019a)"),
    ("G2", "國籍、分配房型等欄位被刪除的理由不一，影響未分別量化", "依據：António et al. (2017b)；Silvestre et al. (2026)"),
    ("G3", "時間外推評估只見於未公開資料，此公開資料缺少隨機與時間切分對照", "依據：António et al. (2019a)；表 2-2"),
    ("G4", "取消 Recall 偏低（0.603）時，未討論類別權重或門檻的取捨", "依據：António et al. (2017b)"),
]
grp = [("Shape 2", "Shape 3", "Text 4", "Text 5"), ("Shape 6", "Shape 7", "Text 8", "Text 9"),
       ("Shape 10", "Shape 11", "Text 12", "Text 13")]
# 複製第 3 組做第 4 組
spTree = s.shapes._spTree
for n in grp[2]:
    e = copy.deepcopy(shp(s, n)._element)
    cn = e.find(".//" + qn("p:cNvPr")); cn.set("name", n + "_g4"); cn.set("id", str(900 + len(spTree)))
    spTree.append(e)
grp.append(tuple(n + "_g4" for n in grp[2]))
H = 1.1
for k, (card, circ, lab, txt) in enumerate(grp):
    y = 1.55 + k * (H + 0.13)
    c = shp(s, card); c.top, c.height = Inches(y), Inches(H)
    for n in (circ, lab):
        o = shp(s, n); o.top = Inches(y + (H - 0.7) / 2)
    t = shp(s, txt); t.top, t.height = Inches(y + 0.05), Inches(H - 0.1)
    fill(shp(s, lab), [gaps[k][0]], color=None)
    fill(t, [gaps[k][1], gaps[k][2]])
arrow = shp(s, "Shape 14"); arrow.top = Inches(3.6)
fill(shp(s, "Text 17"), [
    [("G1 → E1：", B), ("同一前處理、5 個種子比較 LR、RF、LightGBM", {})],
    [("G2 → E3：", B), ("逐欄加回三個洩漏欄位，並檢驗重複列", {})],
    [("G3 → E4：", B), ("以 2017 年訂單測試，bootstrap 估計差異", {})],
    [("G4 → E2：", B), ("比較類別權重與門檻調整", {})],
], size=15, color=None)
p15 = shp(s, "Shape 15"); p15.top, p15.height = Inches(1.55), Inches(4.82)

# ---------- 17 研究架構 ----------
s = slide(17)
for n, v in [("Text 3", "資料集"), ("Text 8", "資料前處理"), ("Text 13", "模型設計"),
             ("Text 18", "實驗設計與評估指標"), ("Text 23", "錯誤分析與應用展示")]:
    fill(shp(s, n), [v], color=None, size=15 if len(v) > 5 else None)
for n, v in [("Text 5", "Kaggle Hotel Booking Demand，119,390 筆"),
             ("Text 10", "刪除洩漏欄位、異常值、缺失值、編碼"),
             ("Text 15", "LR、RF（baseline）＋ LightGBM（提出方法）"),
             ("Text 20", "E1–E4；F1、AUC、bootstrap"),
             ("Text 25", "分群錯誤分析 → 期末評估工具")]:
    fill(shp(s, n), [v])
solid_border(shp(s, "placeholder-box"))
fill(shp(s, "Text 27"), [
    [("流程原則", {"bold": True, "size": 17, "color": "9A7D0A"})],
    [("需要統計量的步驟（中位數、截尾門檻、標準化、類別對照）只以訓練集擬合；測試集只在最後評估一次", {"size": 16, "color": DARK})],
    [("模組名稱與第三章各節對應；本研究的主要貢獻是評估流程，LightGBM 為在此流程下檢驗的提出模型", {"size": 16, "color": DARK})],
])

# ---------- 18 資料集描述 ----------
s = slide(18)
tb = [x for x in s.shapes if x.has_table][0]
table(tb, [
    ["來源與授權", "Kaggle jessemostipak/hotel-booking-demand 版本 1；CC BY 4.0（António et al., 2019b）"],
    ["資料型態", "結構化表格，每列一筆訂單"],
    ["樣本數", "原始 119,390 筆（城市 79,330、度假 40,060）；清理後 119,209 筆"],
    ["特徵或類別", "原始 32 欄；使用 9 個類別＋18 個數值特徵，One-hot 後約 145 維"],
    ["類別分布", "未取消 75,166 筆（63.0%）、取消 44,224 筆（37.0%）"],
    ["資料分割", "70／15／15 分層抽樣，5 個種子；E4 依抵達日期切分"],
    ["倫理與個資", "原作者已刪除可識別飯店與顧客的欄位；本研究未蒐集個資"],
], size=13, row_h=[0.45] + [0.62] * 7)
picture(s, "placeholder-box", "results/fig_class_distribution.png", ["Text 3"])
fill(shp(s, "source"), ["來源：Mostipak (2020), https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand（2026 年 11 月 3 日下載）；António et al. (2019b)；圖為本研究繪製"], color=None)

# ---------- 19 前處理 ----------
s = slide(19)
for n, v in [("Text 5", "刪除洩漏欄位"), ("Text 10", "異常值處理"), ("Text 15", "缺失值處理"),
             ("Text 20", "編碼與標準化"), ("Text 25", "類別不平衡處理")]:
    fill(shp(s, n), [v], color=None)
for n, v in [("Text 7", "刪 6 欄：訂單狀態 2 欄、國籍、分配房型、停車需求、年份"),
             ("Text 12", "刪 0 人訂單 180 筆、負房價 1 筆；adr 以訓練集 99.9 百分位截尾"),
             ("Text 17", "agent、company 空值＝不適用，轉為 None；children 以中位數補值"),
             ("Text 22", "One-hot，比例 < 0.1% 併為 infrequent；僅 LR 標準化"),
             ("Text 27", "主實驗不重抽樣；E2 比較類別權重與門檻調整")]:
    fill(shp(s, n), [v])
fill(shp(s, "Text 29"), [[("資料洩漏防範　", B), ("逐列固定規則（刪欄、空值轉類別、衍生特徵）在切分前執行；中位數、截尾門檻、標準化參數與類別對照只以訓練集擬合，再套用至驗證集與測試集", {})]])

# ---------- 20 模型設計 ----------
s = slide(20)
fill(shp(s, "Text 3"), ["Baseline 1：Logistic Regression"], color=None)
fill(shp(s, "Text 4"), ["主要超參數：C＝1.0，L2，max_iter＝3000，數值特徵標準化", "選用理由：線性且可解釋，作為最低比較基準"])
fill(shp(s, "Text 6"), ["Baseline 2：Random Forest"], color=None)
fill(shp(s, "Text 7"), ["主要超參數：n_estimators＝300，min_samples_leaf＝2", "選用理由：可處理非線性與交互作用（Breiman, 2001）"])
fill(shp(s, "Text 8"), ["提出方法：LightGBM（以驗證集 log-loss 提前停止）"], color=None)
picture(s, "placeholder-box", "results/fig_gbdt_schematic.png", ["Text 10"], pad=0.05)

# ---------- 21 實驗設計 ----------
s = slide(21)
tb = [x for x in s.shapes if x.has_table][0]
table(tb, [
    ["E1", "比較模型效能", "模型種類（無類別權重）", "LR、RF、LightGBM；同種子成對差異", "RQ1"],
    ["E2", "類別權重的影響", "class_weight 無／balanced；同 Recall 之門檻", "三種模型各自有無權重", "RQ2"],
    ["E3", "洩漏欄位與重複列（消融）", "逐欄加回 3 欄；刪除完全重複列", "主實驗設定", "RQ3"],
    ["E4", "時間外推能力", "依抵達日期切分；0.5 與驗證集門檻", "E1 隨機切分、E3 去重結果", "RQ1"],
], size=12, row_h=[0.42, 0.5, 0.58, 0.5, 0.5])
fill(shp(s, "Text 4"), ["70／15／15 分層抽樣；E4 訓練 2015-07～2016-09、驗證 2016 Q4、測試 2017-01～08"], size=14)
fill(shp(s, "Text 7"), ["獨立驗證集（提前停止、選門檻）；E4 以 bootstrap B＝2,000 估計差異"], size=14)
fill(shp(s, "Text 10"), ["42–46，每次重新切分與擬合，報告平均 ± 標準差；E4 為 42"], size=14)

# ---------- 22 評估指標 ----------
s = slide(22)
for n, v in [("Text 5", "選用理由：取消僅 37%，全猜未取消也有 63%，不作主要指標"),
             ("Text 9", "選用理由：據預測超賣或聯繫顧客時，誤判的成本"),
             ("Text 13", "選用理由：漏掉取消則空房無法提前補售；RQ2 主要指標"),
             ("Text 17", "選用理由：兼顧 Precision 與 Recall；RQ1 主要比較指標")]:
    fill(shp(s, n), [v], color="566573")
add_text(s, 0.6, 6.3, 12.1, 0.4, [[("另報告 AUC-ROC（不受門檻影響，可與文獻比較）與 PR-AUC；正類為「取消」，門檻預設 0.5", {"size": 16, "color": DARK})]],
         shp(s, "Text 5"))

# ---------- 23 初步結果 ----------
s = slide(23)
tb = [x for x in s.shapes if x.has_table][0]
table(tb, [
    ["Logistic Regression", "隨機", "0.832", "0.626", "0.715", "0.858"],
    ["Random Forest", "隨機", "0.886", "0.697", "0.780", "0.921"],
    ["LightGBM", "隨機", "0.858", "0.744", "0.797", "0.923"],
    ["Logistic Regression", "時間", "0.804", "0.558", "0.659", "0.827"],
    ["Random Forest", "時間", "0.823", "0.515", "0.634", "0.840"],
    ["LightGBM", "時間", "0.815", "0.526", "0.640", "0.841"],
], size=12, header=["模型", "切分", "Precision", "Recall", "F1", "AUC"],
   col_w=[2.0, 0.7, 1.0, 0.95, 0.95, 1.0], row_h=0.36)
tb.table.cell(0, 0)  # keep
fill(shp(s, "Text 2"), ["註：測試集結果，取至小數點後第三位；門檻 0.5；隨機切分為 5 個種子平均，時間切分測試期為 2017-01～08"], size=12, color=None)
fill(shp(s, "Text 4"), [
    [("初步觀察", {"bold": True})],
    "隨機切分 LightGBM F1 最高；時間切分降至 0.640，低於 LR 的 0.659",
    "改用驗證集門檻後三者 F1 皆約 0.69，差距 0.005 以內",
    "類別權重 ≈ 降低門檻；三個洩漏欄位至多高估 F1 約 0.056",
], size=14)
picture(s, "placeholder-box", "results/fig_confusion_lgbm.png", ["Text 6"])
add_text(s, 7.6, 6.45, 5.1, 0.3, [[("圖：LightGBM 混淆矩陣（隨機切分測試集，seed 42）", {"size": 11, "color": "566573"})]],
         shp(s, "Text 2"))

# ---------- 24 時程 ----------
s = slide(24)
cols = {}
for x in s.shapes:
    if x.has_text_frame and x.text_frame.text.strip().startswith("W") and len(x.text_frame.text.strip()) == 3:
        cols[x.text_frame.text.strip()] = (x.left, x.width)
plan = [("Text 14", "Shape 15", "模型優化與調參", 10, 12),
        ("Text 17", "Shape 18", "比較與消融實驗", 11, 12),
        ("Text 19", "Shape 20", "系統整合與介面", 13, 15),
        ("Text 22", "Shape 23", "期末報告撰寫", 14, 16),
        ("Text 24", "Shape 25", "成果發表準備", 16, 18)]
for tname, bname, label, a, b in plan:
    fill(shp(s, tname), [label])
    bar = shp(s, bname)
    la, wa = cols[f"W{a:02d}"]; lb, wb = cols[f"W{b:02d}"]
    bar.left = la + Emu(Inches(0.05)); bar.width = (lb + wb) - la - Emu(Inches(0.1))
remove(shp(s, "Text 27")); remove(shp(s, "Shape 28")); remove(shp(s, "Shape 26"))
fill(shp(s, "Text 34"), ["註：依期中報告表 6-2；模型於 W12 結束時定版；負責人待組內分工後加註"], color=None)

# ---------- 25 預期成果與風險 ----------
s = slide(25)
fill(shp(s, "Text 4"), [
    "以 rolling-origin 時間驗證重新選模型：LightGBM 對 LR 的 F1 差異 95% CI 下界 > 0.01 才採用，否則採用 LR",
    "評估工具：Colab／命令列輸出隨機、時間、去重效能與兩層洩漏檢查",
    "網頁展示預先計算結果與單筆預測示範",
    "程式碼與實驗結果已公開於 GitHub",
], size=15)
tb = [x for x in s.shapes if x.has_table][0]
table(tb, [
    ["仍有未察覺的洩漏欄位", "中／高", "逐欄確認確定時點；統計異常欄位標為需人工確認，加回與刪除比較"],
    ["時間切分下無模型明顯優於 LR", "高／中", "事先定 CI 下界 > 0.01 的規則；未達則採用 LR 並如實說明"],
    ["重複列性質無法確認", "高／中", "同時報告保留與刪除重複列；期末加入依相同列分組的切分"],
    ["調參或門檻用到測試集", "低／高", "只用驗證集；測試集只在最後評估一次"],
    ["期末系統整合時間不足", "高／中", "W12 模型定版；網頁只展示預先計算結果，上傳功能延後"],
], size=12, header=["風險項目", "等級", "因應措施"], row_h=[0.42] + [0.72] * 5)
fill(shp(s, "Text 5"), ["註：等級為「發生機率／影響程度」，以低、中、高標示"], color=None)

# ---------- 26 參考文獻 ----------
s = slide(26)
refs = [
    "António, N., de Almeida, A., & Nunes, L. (2017a). Predicting hotel booking cancellations to decrease uncertainty and increase revenue. Tourism & Management Studies, 13(2), 25–39. https://doi.org/10.18089/tms.2017.13203",
    "António, N., de Almeida, A., & Nunes, L. (2017b). Predicting hotel bookings cancellation with a machine learning classification model. In Proceedings of the 16th IEEE ICMLA (pp. 1049–1054). IEEE. https://doi.org/10.1109/ICMLA.2017.00-11",
    "António, N., de Almeida, A., & Nunes, L. (2019a). Big data in hotel revenue management: Exploring cancellation drivers to gain insights into booking cancellation behavior. Cornell Hospitality Quarterly, 60(4), 298–319. https://doi.org/10.1177/1938965519851466",
    "António, N., de Almeida, A., & Nunes, L. (2019b). Hotel booking demand datasets. Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD (pp. 785–794). https://doi.org/10.1145/2939672.2939785",
    "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In NeurIPS 35 Datasets and Benchmarks Track. https://arxiv.org/abs/2207.08815",
    "Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30 (pp. 3146–3154).",
    "Mostipak, J. (2020). Hotel booking demand (Version 1) [Data set]. Kaggle. https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand",
    "Silvestre, P., Antonio, N., & Carrasco, P. (2026). Navigating uncertainty: Enhancing hotel cancellation predictions with adaptive machine learning. Information Technology & Tourism, 28, Article 9. https://doi.org/10.1007/s40558-025-00349-9",
]
r = shp(s, "Text 2"); r.top, r.height = Inches(1.45), Inches(4.9)
fill(r, refs, size=11)
t3 = shp(s, "Text 3"); t3.top = Inches(6.45)
fill(t3, ["註：僅列簡報中引用之文獻，APA 第 7 版；完整清單見期中報告參考文獻"], color=None)

# ---------- 27 結尾 ----------
s = slide(27)
fill(shp(s, "Text 2"), ["第　組｜" + TITLE_ZH], color=None)

# ---------- 講稿寫入備忘稿 ----------
import re as _re
_md = open("講稿_與預期提問.md", encoding="utf8").read().split("# 預期提問")[0]
_pat = r"### p(\d+) ([^\n]+)\n(.*?)(?=\n### p|\n---|\Z)"
for _m in _re.finditer(_pat, _md, _re.S):
    _n = int(_m.group(1)); _body = _m.group(3).strip()
    S[_n - 1].notes_slide.notes_text_frame.text = _m.group(2).strip() + chr(10) + _body

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
