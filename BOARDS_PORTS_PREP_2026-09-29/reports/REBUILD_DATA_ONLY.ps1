# Data preparation only. Does not run training or test evaluation.
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\boards_ports'
$taskPython = 'C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskRoot)) { throw "Dataset workspace is missing: $taskRoot" }
Push-Location -LiteralPath $taskRoot
try {
    & $taskPython import_roboflow.py
    if ($LASTEXITCODE -ne 0) { throw 'Import failed' }
    & $taskPython assemble.py
    if ($LASTEXITCODE -ne 0) { throw 'Assembly failed' }
    & $taskPython verify_datasets.py
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
    Write-Host 'Read datasets/verification.json: structure_pass is separate from training_ready.'
} finally { Pop-Location }
