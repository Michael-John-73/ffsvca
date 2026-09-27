$env:PYTHONIOENCODING = "utf-8"
Set-Location F:\RCE\PAPER3
& F:\RCE\venv_p3\Scripts\python.exe -u scripts_p3\16_make_paper_figures.py --metrics_dir sd21_results\outputs_p3\metrics --splits_dir sd21_results\outputs_p3\splits --out_dir outputs_p3\figures
Get-ChildItem outputs_p3\figures | Select-Object Name, LastWriteTime | Format-Table -AutoSize
