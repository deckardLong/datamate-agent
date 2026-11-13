<#
Run inference on the 30-business train split using code_member_d and score with code_member_c.
Example:
  .\code_member_c\run_eval_train.ps1 -Adapter "D:\Zalo AIC 2026\final_adapter"
#>
param(
    [Parameter(Mandatory = $true)][string]$Adapter,
    [string]$Python = ".\venv312\Scripts\python.exe",
    [string]$TrainDir = "train",
    [string]$OutDir = "outputs\eval_train",
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$questions = Join-Path $TrainDir "questions.jsonl"
$answers = Join-Path $TrainDir "answers.jsonl"
$submission = Join-Path $OutDir "submission_train.jsonl"

New-Item -ItemType Directory -Force $OutDir | Out-Null

$runArgs = @(
    "-m", "code_member_d.run_all",
    "--questions", $questions,
    "--cache-root", "cache",
    "--split", "train",
    "--out", $submission,
    "--adapter", $Adapter
)
if ($Limit -gt 0) { $runArgs += @("--limit", $Limit) }

Write-Host "== Running inference on train split =="
& $Python @runArgs

Write-Host "== Scoring with code_member_c/evaluate.py =="
& $Python code_member_c\evaluate.py --answers $answers --submission $submission --questions $questions

