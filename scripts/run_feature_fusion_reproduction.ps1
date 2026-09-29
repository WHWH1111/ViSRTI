[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "python",
    [string]$RunName = "feature_fusion_random_seed1234",
    [switch]$Smoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

$runRoot = Join-Path $RepoRoot ("outputs\" + $RunName)
$dataRoot = Join-Path $runRoot "input_data"
$experimentRoot = Join-Path $runRoot "experiments"
$taskLogRoot = Join-Path $runRoot "task_logs"
$manifestPath = Join-Path $runRoot "run_manifest.json"
if (Test-Path -LiteralPath $manifestPath) {
    throw "Run manifest already exists; refusing to overwrite an experiment: $manifestPath"
}
New-Item -ItemType Directory -Path $dataRoot, $experimentRoot, $taskLogRoot -Force | Out-Null

foreach ($mode in @("pos", "neg")) {
    $name = "MMF_GNN_${mode}.csv"
    $source = Join-Path $RepoRoot ("data\MMF-3\" + $name)
    $target = Join-Path $dataRoot $name
    Copy-Item -LiteralPath $source -Destination $target
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash -ne
        (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash) {
        throw "Copied source-data hash does not match: $name"
    }
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONHASHSEED = "1234"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$iteration = if ($Smoke) { "1" } else { "150" }
$common = @(
    "--split-strategy", "random",
    "--split-seed", "1234",
    "--dev-split-seed", "42",
    "--data-root", $dataRoot,
    "--iteration", $iteration,
    "--batch-train", "64",
    "--batch-test", "64",
    "--standardize",
    "--standardize-features",
    "--loss-function", "l1",
    "--early-stopping-patience", "20"
)
if ($Smoke) {
    $common += @("--debug-size", "128")
}

$levels = @(
    [ordered]@{ code = "G"; level = "graph" },
    [ordered]@{ code = "GP"; level = "graph_physchem" },
    [ordered]@{ code = "GPT"; level = "graph_physchem_toxicity" },
    [ordered]@{ code = "GPTC"; level = "all" }
)
$tasks = @()
foreach ($level in $levels) {
    foreach ($mode in @("pos", "neg")) {
        $outputDir = Join-Path $experimentRoot ($level.code + "_" + $mode)
        $toxicityDim = if ($mode -eq "pos") { "64" } else { "128" }
        $tasks += [ordered]@{
            id = ($level.code + "_" + $mode)
            mode = $mode
            feature_code = $level.code
            feature_level = $level.level
            toxicity_hidden_dim = [int]$toxicityDim
            output_dir = $outputDir
            arguments = @(
                "--dataname", "MMF_GNN_${mode}",
                "--visnet-v2-feature-level", $level.level,
                "--visnet-v2-toxicity-hidden-dim", $toxicityDim,
                "--output-dir", $outputDir
            ) + $common
            status = "queued"
            started_at = $null
            completed_at = $null
            duration_seconds = $null
            exit_code = $null
            command_log = $null
        }
    }
}

$gpu = (& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits) -join " | "
$state = [ordered]@{
    run_id = $RunName
    repo_root = $RepoRoot
    python = $Python
    gpu = $gpu
    smoke = [bool]$Smoke
    split_strategy = "random"
    split_seed = 1234
    dev_split_seed = 42
    model_revision = "eager_nested_head_before_optimizer"
    output_head = "nested_mlp"
    feature_order = @("G", "GP", "GPT", "GPTC")
    supervisor_pid = $PID
    status = "running"
    started_at = (Get-Date).ToString("o")
    updated_at = (Get-Date).ToString("o")
    completed_at = $null
    total_tasks = $tasks.Count
    completed_tasks = 0
    current_task = $null
    failure = $null
    tasks = $tasks
}

function Save-State {
    $state.updated_at = (Get-Date).ToString("o")
    $json = $state | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText(
        $manifestPath, $json, [System.Text.UTF8Encoding]::new($true)
    )
}

try {
    $cudaCheck = & $Python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed." }
    $state.cuda_preflight = ($cudaCheck -join " ")
    Save-State

    foreach ($task in $tasks) {
        if (Test-Path -LiteralPath $task.output_dir) {
            throw "Experiment directory already exists: $($task.output_dir)"
        }
        $started = Get-Date
        $task.status = "running"
        $task.started_at = $started.ToString("o")
        $task.command_log = Join-Path $taskLogRoot ($task.id + ".log")
        $state.current_task = $task.id
        Save-State

        & $Python -u "train_visnet_v2.py" @($task.arguments) 2>&1 |
            Tee-Object -FilePath $task.command_log
        $task.exit_code = $LASTEXITCODE
        if ($task.exit_code -ne 0) {
            throw "Task $($task.id) failed with exit code $($task.exit_code)."
        }
        foreach ($required in @(
            "experiment_config.json", "split_manifest.json", "partition_assignments.csv",
            "best.pt", "model.pt", "predictions_test.csv", "metrics.json"
        )) {
            if (-not (Test-Path -LiteralPath (Join-Path $task.output_dir $required))) {
                throw "Task $($task.id) is missing $required."
            }
        }
        $task.status = "completed"
        $task.completed_at = (Get-Date).ToString("o")
        $task.duration_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
        $state.completed_tasks = @($tasks | Where-Object { $_.status -eq "completed" }).Count
        Save-State
    }

    $state.current_task = "aggregate_results"
    Save-State
    & $Python -u "scripts\data-val\aggregate_feature_fusion_results.py" --run-root $runRoot 2>&1 |
        Tee-Object -FilePath (Join-Path $taskLogRoot "aggregate_results.log")
    if ($LASTEXITCODE -ne 0) { throw "Feature-fusion aggregation failed." }

    $state.status = "completed"
    $state.current_task = $null
    $state.completed_at = (Get-Date).ToString("o")
    Save-State
}
catch {
    $state.status = "failed"
    $state.failure = $_.Exception.Message
    $state.completed_at = (Get-Date).ToString("o")
    if ($null -ne $state.current_task) {
        $failedTask = $tasks | Where-Object { $_.id -eq $state.current_task } | Select-Object -First 1
        if ($null -ne $failedTask) { $failedTask.status = "failed" }
    }
    Save-State
    throw
}
