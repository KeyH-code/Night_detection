$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:surrogateescape'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$trainingCache = Join-Path (Get-Location).Path '.cache\training'
foreach ($sub in @('tmp','pip','ultralytics','torch','matplotlib')) {
    New-Item -ItemType Directory -Path (Join-Path $trainingCache $sub) -Force | Out-Null
}
$env:TEMP = Join-Path $trainingCache 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $trainingCache 'pip'
$env:YOLO_CONFIG_DIR = Join-Path $trainingCache 'ultralytics'
$env:TORCH_HOME = Join-Path $trainingCache 'torch'
$env:MPLCONFIGDIR = Join-Path $trainingCache 'matplotlib'
python -m venv .venv-train
if ($LASTEXITCODE) { throw 'venv creation failed; use Python 3.12' }
& .\.venv-train\Scripts\python.exe -m pip install --require-hashes -r environment/requirements-windows.lock --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE) { throw 'dependency installation failed' }
& .\.venv-train\Scripts\python.exe -m pip check
if ($LASTEXITCODE) { throw 'dependency check failed' }
