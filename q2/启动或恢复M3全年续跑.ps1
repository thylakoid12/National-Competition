$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $taskPython -X utf8 -u (Join-Path $PSScriptRoot 'q2_m3_resume.py') --workers 6
if ($LASTEXITCODE -ne 0) { throw "M3 run exited with code $LASTEXITCODE. Re-run this script to resume." }

