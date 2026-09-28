param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$Device = '0',
    [string]$PythonPath = 'C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe',
    [switch]$BoardRoi
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Python environment not found. Pass -PythonPath with an environment containing the recorded dependencies.'
}
$sourcePath = (Resolve-Path -LiteralPath $InputPath).Path
$entryPoint = Join-Path $PSScriptRoot 'scripts\infer_rpi.py'
$pythonArguments = @('-B', $entryPoint, '--input', $sourcePath, '--output', $OutputPath, '--device', $Device)
if ($BoardRoi) { $pythonArguments += '--board-roi' }
& $PythonPath @pythonArguments
exit $LASTEXITCODE
