# R04 review reproduction (read-only; CPU only; writes ./outputs/*.json)
# Usage (PowerShell):  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
$ErrorActionPreference = 'Stop'
$python = 'C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $PSScriptRoot
foreach ($script in 'r01_hash_bindings.py', 'r02_optimizer_schedule.py', 'r03_split_reproduction.py',
                    'r04_independent_ap_matching.py', 'r05_ignore_geometry_check.py',
                    'r06_exposure_and_anchor_coverage.py', 'r07_eval_caps.py', 'r08_excluded_objects.py') {
    & $python -B $script
    if ($LASTEXITCODE -ne 0) { throw "$script failed with exit code $LASTEXITCODE" }
}
