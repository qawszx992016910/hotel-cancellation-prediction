"""報告圖 5-2、5-3：以 Chrome 無頭模式截取網頁畫面（淺色主題）。
執行：python code/demo_screenshots.py → results/final/fig_demo_daily.png、fig_demo_one.png
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PAGE = (ROOT / "demo" / "index.html").as_uri()

for tab, name, h in (("daily", "fig_demo_daily", 1100), ("one", "fig_demo_one", 1000)):
    out = ROOT / "results" / "final" / f"{name}.png"
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={td}",
                        f"--window-size=1150,{h}", "--virtual-time-budget=8000", f"--screenshot={out}",
                        f"{PAGE}?tab={tab}&theme=light"], check=True, capture_output=True, timeout=120)
    print("saved", out)
