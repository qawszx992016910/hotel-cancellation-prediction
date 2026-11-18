# 以 Word 開啟 docx：更新目錄與功能變數、存檔、匯出 PDF
# 用法：powershell -ExecutionPolicy Bypass -File code/word_update.ps1 <docx 完整路徑>
param([string]$Path)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Path)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    $pdf = [System.IO.Path]::ChangeExtension($Path, ".pdf")
    $doc.ExportAsFixedFormat($pdf, 17)
    $pages = $doc.ComputeStatistics(2)
    Write-Output "pages=$pages pdf=$pdf"
    $doc.Close()
} finally {
    $word.Quit()
}
