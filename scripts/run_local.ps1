<#
Run the public-test submission on a local GPU: shards one after another, resumable,
then merge + validate. Safe to stop (Ctrl+C) and start again with the same arguments:
questions that already have a row are skipped.

Examples (from the repo root, with the Python 3.12 venv):
  # 1) 10-question smoke test first: measures speed and memory
  .\scripts\run_local.ps1 -Adapter "D:\Zalo AIC 2026\final_adapter" -Limit 10
  # 2) the real run, all four shards, then merge
  .\scripts\run_local.ps1 -Adapter "D:\Zalo AIC 2026\final_adapter"
  # 3) only some shards (e.g. the small one first)
  .\scripts\run_local.ps1 -Adapter "D:\Zalo AIC 2026\final_adapter" -Shards 1
#>
param(
    [Parameter(Mandatory = $true)][string]$Adapter,
    [string]$Python = ".\venv312\Scripts\python.exe",
    [string]$OutDir = "outputs\local_run1",
    [int[]]$Shards = @(1, 2, 3, 4),
    [int]$Limit = 0,
    [int]$MaxInputTokens = 3072,
    [int]$FallbackInputTokens = 2048,
    [int]$MaxAttempts = 2,
    [int]$RulesChars = 1600,
    # Where the ~15 GB Qwen model is downloaded. Used only if HF_HOME is not already set.
    [string]$HfHome = "D:\hf_cache"
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns a native program's stderr lines
# (progress bars, logging) into terminating errors under "Stop". Exit codes are checked below.
$ErrorActionPreference = "Continue"
$questions = "public_test\questions.jsonl"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
if (-not $env:HF_HOME) { $env:HF_HOME = $HfHome }
# hf-xet (the newer downloader) can stall on 5 GB files and buffers them in RAM. Plain HTTP
# downloads go straight to disk and resume after an interruption.
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = "1" }
if (-not $env:HF_HUB_DOWNLOAD_TIMEOUT) { $env:HF_HUB_DOWNLOAD_TIMEOUT = "60" }
Write-Host "Hugging Face cache (model download location): $env:HF_HOME"

if (-not (Test-Path $Python)) { throw "Python not found: $Python (create it: py -3.12 -m venv venv312)" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

Write-Host "== Pre-flight check =="
& $Python -m code_member_d.check_env --adapter $Adapter --questions $questions
if ($LASTEXITCODE -ne 0) { throw "Environment is not ready; fix the [FAIL] lines above first." }

$smoke = $Limit -gt 0
$outFiles = @()
foreach ($shard in $Shards) {
    $name = if ($smoke) { "smoke_shard$shard.jsonl" } else { "shard$shard.jsonl" }
    $out = Join-Path $OutDir $name
    $outFiles += $out
    Write-Host "== Shard $shard -> $out =="
    $runArgs = @("-m", "code_member_d.run_all",
        "--questions", $questions, "--cache-root", "cache", "--split", "public_test",
        "--out", $out, "--adapter", $Adapter, "--shard", $shard,
        "--max-input-tokens", $MaxInputTokens, "--fallback-input-tokens", $FallbackInputTokens,
        "--max-attempts", $MaxAttempts, "--rules-chars", $RulesChars)
    if ($smoke) { $runArgs += @("--limit", $Limit) }
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) { Write-Warning "Shard $shard exited with code $LASTEXITCODE; re-run the same command to resume." }
    if ($smoke) { break }   # a smoke test only needs one shard
}

if (-not $smoke) {
    $final = Join-Path $OutDir "final.jsonl"
    Write-Host "== Merge + validate -> $final =="
    # @(...) forces an array: with ONE shard file PowerShell returns a bare string, and
    # splatting a string passes it character by character ("o", "u", "t", ...).
    $existing = @($outFiles | Where-Object { Test-Path $_ })
    if ($existing.Count -eq 0) { throw "No shard output files found in $OutDir" }
    & $Python -m code_member_d.make_baseline_submission --questions $questions --merge-from $existing --out $final
    if ($LASTEXITCODE -ne 0) { throw "Merge failed" }
    & $Python -m code_member_d.make_baseline_submission --questions $questions --validate $final
    if ($LASTEXITCODE -ne 0) { throw "Validation failed: do not upload $final" }
    Write-Host "Upload: $final"
}
