"""網頁功能測試（表 5-2）：複製 demo/index.html 到暫存資料夾，在 </body> 前注入測試腳本，
以 Chrome 無頭模式載入並取回 DOM，解析每項測試結果。不修改 demo/index.html 本身。
執行：python code/demo_functional_test.py [chrome 路徑]
輸出：results/final/demo_functional_test.csv
"""
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\Google\Chrome\Application\chrome.exe"

TEST_JS = r"""
<script>
(function () {
  const R = [];
  const ok = (id, name, pass, detail) => R.push({id, name, pass: !!pass, detail: String(detail)});
  const fire = (el, ev) => el.dispatchEvent(new Event(ev, {bubbles: true}));
  const rows = sel => [...document.querySelectorAll(sel + " tbody tr")];
  try {
    // F1 分頁切換
    const btns = [...document.querySelectorAll("#tabs button")];
    let good = btns.length === 3;
    for (const b of btns) { b.click(); good = good && document.getElementById(b.dataset.p).classList.contains("on")
      && [...document.querySelectorAll(".panel.on")].length === 1; }
    ok("F1", "三個分頁可切換且一次只顯示一頁", good, btns.length + " 個分頁");
    // F2 成果摘要
    const kp = document.querySelector("#kpis").innerText;
    ok("F2", "成果摘要顯示 2017 測試期 F1 0.680", kp.includes("0.680"), kp.replace(/\s+/g, " ").slice(0, 80));
    // F3 每日名單：指定日期、飯店、聯繫時點
    $("#d").value = "2017-08-15"; $("#dh").value = "City Hotel"; $("#dc").value = "7"; renderDaily();
    const g = rows("#dgen");
    const hotels = g.map(r => r.cells[2].innerText.trim());
    ok("F3", "篩選飯店後名單只含該飯店", g.length > 0 && hotels.every(h => h === "City Hotel"), g.length + " 筆");
    const ps = g.map(r => parseFloat(r.cells[1].innerText));
    ok("F4", "一般訂單依取消機率由高到低排序", ps.every((v, i) => i === 0 || ps[i - 1] >= v), "第一筆 " + ps[0] + "%");
    const m = $("#dstat").innerText.match(/一般訂單 (\d+) 筆/), n = m ? +m[1] : -1;
    const tops = g.filter(r => r.classList.contains("top")).length;
    ok("F5", "前 10% 標示筆數＝ceil(一般訂單數×10%)", n > 0 && tops === Math.ceil(n * 0.1), `一般 ${n} 筆、標示 ${tops} 筆`);
    // F6 找一天同時有不退款與團體訂單（全部飯店、抵達當天）再檢查分列
    $("#dh").value = ""; $("#dc").value = "0";
    let day6 = "";
    for (const dday of [...new Set(Lz.arrival_date)].sort()) {
      $("#d").value = dday; renderDaily();
      if (rows("#dnr").length && rows("#dgrp").length) { day6 = dday; break; }
    }
    const g6 = rows("#dgen");
    const nrOk = day6 && rows("#dnr").every(r => r.cells[5].innerText.trim().startsWith("Non Refund"))
      && rows("#dgrp").every(r => r.cells[4].innerText.trim().startsWith("Groups"))
      && g6.every(r => !r.cells[5].innerText.trim().startsWith("Non Refund") && !r.cells[4].innerText.trim().startsWith("Groups"));
    ok("F6", "不退款與團體訂單分列、不混入一般名單", nrOk, `${day6}：一般 ${g6.length}、不退款 ${rows("#dnr").length}、團體 ${rows("#dgrp").length} 筆`);
    $("#d").value = "2017-08-15"; $("#dh").value = "City Hotel";
    // F7 聯繫時點：越早聯繫仍有效的訂單越少或相等（含已取消的「當天」最多）
    const cnt = c => { $("#dc").value = c; renderDaily(); const mm = $("#dstat").innerText.match(/一般訂單 (\d+) 筆/); return mm ? +mm[1] : 0; };
    const c30 = cnt("30"), c0 = cnt("0");
    ok("F7", "聯繫時點改變時名單隨之更新（抵達前 30 天 ≤ 當天）", c30 <= c0 && c0 > 0, `30 天 ${c30} 筆、當天 ${c0} 筆`);
    // F8 無資料的日期
    $("#d").value = "2016-06-01"; renderDaily();
    ok("F8", "超出資料期間的日期顯示提示而不出錯", $("#dstat").innerText.includes("沒有符合條件"), $("#dstat").innerText.slice(0, 30));
    // F9 單筆預測預設值
    const out = () => $("#out").innerText;
    runOne();
    ok("F9", "單筆預測預設輸入顯示取消機率與說明", /\d+\.\d%/.test(out()) && out().includes("說明"), (out().match(/\d+\.\d%/) || [""])[0]);
    // F10 修改欄位即時更新
    const before = (out().match(/\d+\.\d%/) || [""])[0];
    const f = $("#f");
    f.elements.hotel.value = "Resort Hotel"; fire(f.elements.hotel, "input");
    const after = (out().match(/\d+\.\d%/) || [""])[0];
    ok("F10", "修改欄位後機率即時重算", before !== after, `${before} → ${after}`);
    // F11 不退款提示
    f.elements.deposit_type.value = "Non Refund"; fire(f.elements.deposit_type, "input");
    ok("F11", "不退款訂單顯示分開看待的提示", out().includes("不退款訂單已收款"), "");
    f.elements.deposit_type.value = "No Deposit"; f.elements.market_segment.value = "Groups"; fire(f.elements.market_segment, "input");
    ok("F12", "團體訂單顯示合約提示", out().includes("團體訂單通常有合約"), "");
    // F13 空白日期不出錯
    const keep = out(); f.elements.arrival.value = ""; fire(f.elements.arrival, "input");
    ok("F13", "抵達日空白時不出錯、保留上次結果", out() === keep, "");
    // JS 推論時間
    f.elements.arrival.value = "2017-08-15"; f.elements.market_segment.value = "Online TA"; fire(f.elements.market_segment, "input");
    const t0 = performance.now(); for (let i = 0; i < 1000; i++) runOne(); const ms = (performance.now() - t0) / 1000;
    ok("T1", "單筆預測含畫面更新之平均時間（毫秒）", ms < 100, ms.toFixed(3));
  } catch (e) { ok("ERR", "測試腳本例外", false, e && e.stack); }
  const pre = document.createElement("pre"); pre.id = "TESTOUT"; pre.textContent = JSON.stringify(R); document.body.appendChild(pre);
})();
</script>
"""


def main():
    html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src=["\']https?:', html), "網頁不應載入外部腳本"
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "index.html"
        page.write_text(html.replace("</body>", TEST_JS + "</body>"), encoding="utf-8")
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--virtual-time-budget=20000",
                            f"--user-data-dir={td}\\prof", "--dump-dom", page.as_uri()],
                           capture_output=True, text=True, encoding="utf-8", timeout=180)
    m = re.search(r'<pre id="TESTOUT">(.*?)</pre>', r.stdout, re.S)
    assert m, r.stderr[-2000:]
    res = json.loads(m.group(1).replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    res.insert(0, dict(id="F0", name="網頁不載入任何外部腳本（可離線使用）", passed=True, detail=""))
    for x in res:
        x["passed"] = x.pop("pass", x.get("passed"))
        print(("PASS" if x["passed"] else "FAIL"), x["id"], x["name"], x["detail"])
    with open(ROOT / "results" / "final" / "demo_functional_test.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "name", "passed", "detail"]); w.writeheader(); w.writerows(res)
    f = [x for x in res if x["id"].startswith("F")]
    print(f"功能測試 {sum(x['passed'] for x in f)}/{len(f)} 通過")


if __name__ == "__main__":
    main()
