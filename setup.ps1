$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Khong tao duoc moi truong Python' }
}
python -m pip --python .venv install pip
if ($LASTEXITCODE -ne 0) { throw 'Khong cai duoc pip' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Khong cai duoc thu vien' }
Write-Host 'Da san sang. Chay run.bat de mo backend va giao dien.'
