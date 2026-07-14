$py = "F:\Projects\.amd_venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "F:\Projects\igmcg-llm"
Add-Content full_cmp_run.log "=== START ENH FULL ==="
& $py -u scripts/train.py --config configs/config_cmp_enh_full.yaml >> full_cmp_run.log 2>&1
Add-Content full_cmp_run.log "=== START SEL FULL (old 8-seg) ==="
& $py -u scripts/train.py --config configs/config_cmp_sel_full.yaml >> full_cmp_run.log 2>&1
Add-Content full_cmp_run.log "=== START SELv2 FULL ==="
& $py -u scripts/train.py --config configs/config_cmp_selv2_full.yaml >> full_cmp_run.log 2>&1
Add-Content full_cmp_run.log "ALL_DONE"
