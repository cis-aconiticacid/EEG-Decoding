$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Report = Join-Path $Root 'reports/a077_pair_tasks'
$Review = Join-Path $Report 'PRE_RUN_REVIEW.md'
if (-not (Test-Path -LiteralPath $Review)) { throw "Sol review is missing: $Review" }
if (-not (Select-String -LiteralPath $Review -Pattern '^Decision:\s*PASS\s*$' -Quiet)) {
    throw "Model fitting is blocked until PRE_RUN_REVIEW.md contains the exact line 'Decision: PASS'."
}
$Ready = Join-Path $Report 'PRE_RUN_READY.md'
if (-not (Test-Path -LiteralPath $Ready)) { throw "PRE_RUN_READY.md is missing: $Ready" }
$Python = Join-Path $Root '.venv/Scripts/python.exe'
$Runner = Join-Path $PSScriptRoot 'a077_run_models.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Repository venv Python is missing: $Python" }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout = Join-Path $Report "training_$Stamp.stdout.log"
$Stderr = Join-Path $Report "training_$Stamp.stderr.log"
$Process = Start-Process -FilePath $Python -ArgumentList @($Runner, '--resume') -WorkingDirectory $Root `
    -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
$State = [ordered]@{
    status = 'running'
    pid = $Process.Id
    started_local = (Get-Date).ToString('o')
    stdout_log = $Stdout
    stderr_log = $Stderr
    command = "$Python $Runner --resume"
    max_cpu_threads = 4
    local_only = $true
}
$StatePath = Join-Path $Report 'background_process.json'
$State | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatePath -Encoding utf8
Write-Output "Started hidden A077 runner PID $($Process.Id)."
Write-Output "stdout: $Stdout"
Write-Output "stderr: $Stderr"
