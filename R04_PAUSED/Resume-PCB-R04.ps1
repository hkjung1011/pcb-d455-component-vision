param([string]$PythonPath='C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe')
$ErrorActionPreference='Stop'
$TaskWorkspace='C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Recorded Python environment is missing.' }
Set-Location -LiteralPath $TaskWorkspace
Write-Host 'R04: reuse completed training and validation; run remaining evaluation only.' -ForegroundColor Cyan
& $PythonPath -B 'work/r04/scripts/run_r04_evaluation.py' --root 'work/r04'
if ($LASTEXITCODE -ne 0) { throw 'Evaluation stopped. Preserve the records and inspect the log before retrying.' }
Write-Host 'Main evaluation complete. Return to this Codex chat for stress checks, audit, figures, packaging and GitHub publication.' -ForegroundColor Green
