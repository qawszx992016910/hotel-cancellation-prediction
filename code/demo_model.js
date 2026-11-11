// 瀏覽器端 LightGBM 推論：重現 sklearn 前處理（adr 截尾 → 中位數補值 → one-hot 含 infrequent）＋樹加總＋sigmoid
function featureVector(M, row) {
  const x = [];
  M.num.forEach((c, i) => {
    let v = row[c];
    v = (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) ? NaN : Number(v);
    if (c === "adr" && !Number.isNaN(v)) v = Math.min(v, M.adr_cap);
    x.push(Number.isNaN(v) ? M.medians[i] : v);
  });
  for (const c of M.cats) {
    const v = String(row[c.name]);
    const hasInf = c.infrequent.length > 0;
    const idx = c.frequent.indexOf(v);
    for (let k = 0; k < c.frequent.length; k++) x.push(k === idx ? 1 : 0);
    if (hasInf) x.push(idx === -1 ? 1 : 0);   // 罕見或未見過的類別 → infrequent 欄
  }
  return x;
}
function predict(M, row) {
  const x = featureVector(M, row);
  let s = 0;
  for (const t of M.trees) {
    let i = 0;
    while (t[i].length === 4) i = x[t[i][0]] <= t[i][1] ? t[i][2] : t[i][3];
    s += t[i][0];
  }
  return 1 / (1 + Math.exp(-s));
}
