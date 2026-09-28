param(
 [Parameter(Mandatory=$true)][string]$InputPath,
 [Parameter(Mandatory=$true)][string]$OutputPath,
 [string]$PythonPath='C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe',
 [string]$Device='0',
 [ValidateSet('recommended','baseline','improved')][string]$Arm='recommended'
)
$ErrorActionPreference='Stop'
& $PythonPath -B (Join-Path $PSScriptRoot 'scripts/infer_r04.py') --input $InputPath --output $OutputPath --selection (Join-Path $PSScriptRoot 'selected_models.json') --device $Device --arm $Arm
if ($LASTEXITCODE -ne 0) { throw "Inference exited $LASTEXITCODE" }
