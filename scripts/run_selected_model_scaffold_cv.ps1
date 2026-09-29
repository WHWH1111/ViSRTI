[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "python",
    [string]$RunName = "selected_models_scaffold_seed1234",
    [string]$SelectionManifest = "",
    [switch]$Smoke,
    [switch]$DryRun,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

if (-not $SelectionManifest) {
    $SelectionManifest = Join-Path $RepoRoot 'results/final_masked_models_seed1234/selection_manifest.json'
}

if ($SelectionManifest) {
    if ($PSBoundParameters.ContainsKey('RunName') -and $RunName -ne 'final_masked_scaffold_seed1234') {
        throw 'A selected-mask run must use RunName final_masked_scaffold_seed1234.'
    }
    $RunName = 'final_masked_scaffold_seed1234'
}

$runRoot = Join-Path $RepoRoot ("outputs\" + $RunName)
$taskLogRoot = Join-Path $runRoot "task_logs"
$manifestPath = Join-Path $runRoot "run_manifest.json"
$iteration = if ($Smoke) { "1" } else { "150" }

$configs = @()

$common = @(
    "--split-strategy", "scaffold",
    "--split-seed", "1234",
    "--iteration", $iteration,
    "--batch-train", "64",
    "--batch-test", "128",
    "--standardize",
    "--standardize-features",
    "--loss-function", "l1",
    "--early-stopping-patience", "20"
)
if ($Smoke) { $common += @("--debug-size", "128") }

$selectionProvenance = $null
if ($SelectionManifest) {
    $SelectionManifest = (Resolve-Path -LiteralPath $SelectionManifest).Path
    $selection = Get-Content -LiteralPath $SelectionManifest -Raw | ConvertFrom-Json -AsHashtable
    $selectionProvenance = [ordered]@{
        manifest = $SelectionManifest
        manifest_sha256 = (Get-FileHash -LiteralPath $SelectionManifest -Algorithm SHA256).Hash
        training_initialization = 'fresh_from_seed_not_selected_checkpoint_weights'
        selection_partition = 'development'
        selection_metric = 'MAE_dev'
    }
    $scalarParameters = @(
        'batch_train', 'batch_test', 'learning_rate', 'iteration', 'weight_decay',
        'scheduler_patience', 'scheduler_factor', 'min_lr', 'early_stopping_patience',
        'loss_function', 'visnet_v2_node_feature_dim', 'visnet_v2_physchem_feature_dim',
        'visnet_v2_toxicity_feature_dim', 'visnet_v2_chromato_feature_dim',
        'visnet_v2_graph_hidden_dim', 'visnet_v2_physchem_hidden_dim',
        'visnet_v2_toxicity_hidden_dim', 'visnet_v2_chromato_hidden_dim',
        'visnet_v2_toxicity_intermediate_dim', 'visnet_v2_dropout_rate',
        'visnet_v2_feature_level', 'visnet_v2_output_head'
    )
    $booleanParameters = @(
        'standardize', 'standardize_features', 'visnet_v2_use_attention', 'visnet_v2_use_gating'
    )
    $featureLevels = @{ G = 'graph'; GP = 'graph_physchem'; GPT = 'graph_physchem_toxicity'; GPTC = 'all' }
    $configs = @()
    foreach ($mode in @('pos', 'neg')) {
        $selected = $selection.models[$mode]
        foreach ($key in @('feature_code', 'model_label', 'checkpoint', 'training_params', 'removed_features', 'physchem_mask', 'toxicity_mask')) {
            if (-not $selected.Contains($key)) { throw "Selection for $mode is missing $key." }
        }
        $paramsPath = [string]$selected.training_params
        $checkpointPath = [string]$selected.checkpoint
        if (-not [System.IO.Path]::IsPathRooted($paramsPath)) { $paramsPath = Join-Path (Split-Path $SelectionManifest) $paramsPath }
        if (-not [System.IO.Path]::IsPathRooted($checkpointPath)) { $checkpointPath = Join-Path (Split-Path $SelectionManifest) $checkpointPath }
        $paramsPath = (Resolve-Path -LiteralPath $paramsPath).Path
        $checkpointPath = (Resolve-Path -LiteralPath $checkpointPath).Path
        $selectedParams = Get-Content -LiteralPath $paramsPath -Raw | ConvertFrom-Json -AsHashtable
        if ($selectedParams.debug_size) { throw "Selected $mode parameters refer to a debug experiment." }
        if ($selectedParams.split_seed -ne 1234) { throw "Selected $mode split seed must be 1234." }
        if ($selectedParams.visnet_v2_feature_level -ne $featureLevels[$selected.feature_code]) {
            throw "Selected $mode feature code and training feature level disagree."
        }
        $expected = [ordered]@{ split_strategy = 'scaffold'; split_seed = 1234 }
        $selectedArguments = @('--split-strategy', 'scaffold', '--split-seed', '1234')
        foreach ($key in $scalarParameters) {
            if (-not $selectedParams.Contains($key)) { throw "Selected $mode parameters are missing $key." }
            $value = $selectedParams[$key]
            if ($Smoke -and $key -eq 'iteration') { $value = 1 }
            $expected[$key] = $value
            $selectedArguments += @(('--' + $key.Replace('_', '-')), [string]$value)
        }
        foreach ($key in $booleanParameters) {
            if (-not $selectedParams.Contains($key)) { throw "Selected $mode parameters are missing $key." }
            $expected[$key] = [bool]$selectedParams[$key]
            if ($selectedParams[$key]) { $selectedArguments += '--' + $key.Replace('_', '-') }
        }
        $expected.visnet_v2_fusion_hidden_dims = @($selectedParams.visnet_v2_fusion_hidden_dims)
        if ($expected.visnet_v2_fusion_hidden_dims.Count -eq 0) { throw "Missing fusion dimensions for $mode." }
        $selectedArguments += '--visnet-v2-fusion-hidden-dims'
        $selectedArguments += @($expected.visnet_v2_fusion_hidden_dims | ForEach-Object { [string]$_ })
        foreach ($group in @('physchem', 'toxicity', 'chromato')) {
            $key = "visnet_v2_${group}_mask"
            $mask = $selectedParams[$key]
            if ($group -ne 'chromato') {
                $requestedMask = $selected["${group}_mask"]
                $width = [int]$selectedParams["visnet_v2_${group}_feature_dim"]
                $effectiveSource = if ($null -eq $mask) { @(1) * $width } else { @($mask) }
                $effectiveRequested = if ($null -eq $requestedMask) { @(1) * $width } else { @($requestedMask) }
                if ($effectiveRequested.Count -ne $width -or
                    @($effectiveRequested | Where-Object { $_ -notin @(0, 1) }).Count -gt 0 -or
                    ($effectiveSource -join ',') -ne ($effectiveRequested -join ',')) {
                    throw "Selected $mode $group mask differs from its checkpoint training parameters."
                }
                $mask = $requestedMask
            }
            $expected[$key] = $mask
            if ($null -ne $mask) {
                $selectedArguments += '--' + $key.Replace('_', '-')
                $selectedArguments += @($mask | ForEach-Object { [string]$_ })
            }
        }
        $expected.debug_size = if ($Smoke) { 128 } else { $null }
        if ($Smoke) { $selectedArguments += @('--debug-size', '128') }
        $splitRoot = Join-Path $RepoRoot "data\MMF-3\MMF_GNN_${mode}_5"
        $splitPath = Join-Path $splitRoot 'split_manifest.json'
        $split = Get-Content -LiteralPath $splitPath -Raw | ConvertFrom-Json
        if ($split.split_strategy -ne 'scaffold' -or $split.random_seed -ne 1234 -or $split.n_folds -ne 5 -or
            @($split.folds | Where-Object { $_.scaffold_overlap_count -ne 0 }).Count -gt 0) {
            throw "The existing $mode scaffold split does not satisfy the frozen five-fold plan."
        }
        $configs += [ordered]@{
            mode = $mode
            feature_code = $selected.feature_code
            model = $selected.model_label
            feature_level = $selectedParams.visnet_v2_feature_level
            toxicity_hidden_dim = [string]$selectedParams.visnet_v2_toxicity_hidden_dim
            arguments = $selectedArguments
            expected_parameters = $expected
            selection = [ordered]@{
                training_params = $paramsPath
                training_params_sha256 = (Get-FileHash -LiteralPath $paramsPath -Algorithm SHA256).Hash
                checkpoint = $checkpointPath
                checkpoint_sha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash
                removed_features = @($selected.removed_features)
                physchem_mask = $selected.physchem_mask
                toxicity_mask = $selected.toxicity_mask
                split_manifest = $splitPath
                split_manifest_sha256 = (Get-FileHash -LiteralPath $splitPath -Algorithm SHA256).Hash
            }
        }
    }
}

$tasks = @()
foreach ($config in $configs) {
    foreach ($fold in 1..5) {
        $runLabel = "MMF_GNN_$($config.mode)_$($config.feature_code)_fold_${fold}_scaffold_seed1234"
        $task = [ordered]@{
            id = "$($config.mode)_$($config.feature_code)_cv_fold${fold}"
            kind = "cv"
            mode = $config.mode
            fold = $fold
            feature_code = $config.feature_code
            model = $config.model
            feature_level = $config.feature_level
            script = "train_visnet_v2_5.py"
            run_label = $runLabel
            arguments = @(
                "--cross-validation", "--cv-dataname", "MMF_GNN_$($config.mode)",
                "--fold", "$fold",
                "--visnet-v2-feature-level", $config.feature_level,
                "--visnet-v2-toxicity-hidden-dim", $config.toxicity_hidden_dim
            ) + $common
            status = "queued"
            started_at = $null
            completed_at = $null
            duration_seconds = $null
            exit_code = $null
            log_dir = $null
            command_log = $null
        }
        if ($SelectionManifest) {
            $task.arguments = @(
                '--cross-validation', '--cv-path', (Join-Path $RepoRoot 'data\MMF-3'),
                '--cv-dataname', "MMF_GNN_$($config.mode)", '--fold', [string]$fold
            ) + $config.arguments
            $task.expected_parameters = $config.expected_parameters
            $task.selection = $config.selection
            $foldRoot = Join-Path $RepoRoot "data\MMF-3\MMF_GNN_$($config.mode)_5\fold_$fold"
            $task.frozen_split_files = @('train', 'test') | ForEach-Object {
                $foldFile = Join-Path $foldRoot "fold_${fold}_$($_)_set.txt"
                [ordered]@{ path = $foldFile; sha256 = (Get-FileHash -LiteralPath $foldFile -Algorithm SHA256).Hash }
            }
        }
        $tasks += $task
    }
}

if ($DryRun) {
    $tasks | ConvertTo-Json -Depth 10
    exit 0
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONHASHSEED = "1234"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$gpu = (& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits) -join " | "
if (Test-Path -LiteralPath $manifestPath) {
    if (-not $Resume) {
        throw "Run manifest already exists; use -Resume only after checking the recorded failure: $manifestPath"
    }
    $state = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if (-not $SelectionManifest -and $state.PSObject.Properties['selection_provenance']) {
        throw 'Resume of a final-mask run requires its SelectionManifest.'
    }
    if ($SelectionManifest) {
        if (-not $state.PSObject.Properties['selection_provenance'] -or
            $state.selection_provenance.manifest_sha256 -ne $selectionProvenance.manifest_sha256) {
            throw 'Resume selection differs from the recorded final-mask selection.'
        }
        if (@($state.tasks).Count -ne $tasks.Count) { throw 'Resume task count differs from the selected plan.' }
        foreach ($planned in $tasks) {
            $recorded = @($state.tasks | Where-Object { $_.id -eq $planned.id })
            if ($recorded.Count -ne 1 -or
                (@($recorded[0].arguments) -join "`0") -ne (@($planned.arguments) -join "`0") -or
                $recorded[0].selection.training_params_sha256 -ne $planned.selection.training_params_sha256 -or
                $recorded[0].selection.checkpoint_sha256 -ne $planned.selection.checkpoint_sha256 -or
                $recorded[0].selection.split_manifest_sha256 -ne $planned.selection.split_manifest_sha256 -or
                (@($recorded[0].frozen_split_files.sha256) -join ',') -ne (@($planned.frozen_split_files.sha256) -join ',')) {
                throw "Resume task $($planned.id) differs from the selected parameters or frozen split."
            }
        }
    }
    $tasks = @($state.tasks)
    if ([bool]$state.smoke -ne [bool]$Smoke) {
        throw "Resume mode does not match the recorded smoke setting."
    }
    $state.status = "running"
    $state.failure = $null
    $state.completed_at = $null
}
else {
    New-Item -ItemType Directory -Path $taskLogRoot -Force | Out-Null
    $state = [ordered]@{
        run_id = $RunName
        repo_root = $RepoRoot
        python = $Python
        gpu = $gpu
        smoke = [bool]$Smoke
        status = "running"
        started_at = (Get-Date).ToString("o")
        updated_at = (Get-Date).ToString("o")
        completed_at = $null
        total_tasks = $tasks.Count
        completed_tasks = 0
        current_task = $null
        failure = $null
        figure_number = "5"
        large_typography = $true
        selection_partition = "development"
        selection_metric = "mae"
        model_labels = [ordered]@{
            pos = "ViSRTI-GP"
            neg = "ViSRTI-GPT"
        }
        tasks = $tasks
    }
    if ($SelectionManifest) {
        $state.selection_provenance = $selectionProvenance
        $state.figure_number = '5'
        $state.model_labels.pos = $configs[0].model
        $state.model_labels.neg = $configs[1].model
    }
}

function Save-State {
    $state.updated_at = (Get-Date).ToString("o")
    [System.IO.File]::WriteAllText(
        $manifestPath,
        ($state | ConvertTo-Json -Depth 10),
        [System.Text.UTF8Encoding]::new($true)
    )
}

function Find-TaskOutput {
    param($Task, [datetime]$StartedAt)
    $pattern = "*_MMF_GNN_$($Task.mode)_fold_$($Task.fold)_scaffold_seed1234_dim48_layerH6_layerO6_batch64_lr0.0001_iter$iteration"
    if ($SelectionManifest) {
        $pattern = "*_MMF_GNN_$($Task.mode)_fold_$($Task.fold)_scaffold_seed1234_dim48_layerH6_layerO6_batch*_lr*_iter*"
    }
    Get-ChildItem -LiteralPath (Join-Path $RepoRoot "log") -Directory |
        Where-Object { $_.Name -like $pattern -and $_.CreationTime -ge $StartedAt.AddSeconds(-2) } |
        Where-Object { -not $SelectionManifest -or (Test-SelectedParameters -Directory $_ -Task $Task) } |
        Sort-Object CreationTime -Descending |
        Select-Object -First 1
}

function Test-SelectedParameters {
    param([System.IO.DirectoryInfo]$Directory, $Task)
    $paramsPath = Join-Path $Directory.FullName 'training_params.json'
    if (-not (Test-Path -LiteralPath $paramsPath)) { return $false }
    $actual = Get-Content -LiteralPath $paramsPath -Raw | ConvertFrom-Json -AsHashtable
    $expected = $Task.expected_parameters | ConvertTo-Json -Depth 10 | ConvertFrom-Json -AsHashtable
    foreach ($key in $expected.Keys) {
        if (-not $actual.Contains($key)) { return $false }
        if (($actual[$key] | ConvertTo-Json -Compress -Depth 10) -ne ($expected[$key] | ConvertTo-Json -Compress -Depth 10)) {
            return $false
        }
    }
    return $actual.cv_dataname -eq "MMF_GNN_$($Task.mode)" -and [int]$actual.fold -eq [int]$Task.fold
}

function Test-TaskOutput {
    param([System.IO.DirectoryInfo]$Directory)
    if ($null -eq $Directory) { return $false }
    foreach ($required in @(
        "training_params.json", "MAEs.txt", "test_prediction.txt", "model.pt", "best.pt",
        "train_prediction_train.txt", "train_prediction_dev.txt", "train_prediction_test.txt"
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory.FullName $required))) {
            return $false
        }
    }
    return $true
}

try {
    $cudaCheck = & $Python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed." }
    $state.cuda_preflight = ($cudaCheck -join " ")
    Save-State

    foreach ($task in $tasks) {
        if ($SelectionManifest) {
            foreach ($inputFile in $task.frozen_split_files) {
                if ((Get-FileHash -LiteralPath $inputFile.path -Algorithm SHA256).Hash -ne $inputFile.sha256) {
                    throw "Frozen scaffold input changed before task $($task.id): $($inputFile.path)"
                }
            }
        }
        if ($task.status -eq "completed") {
            if ($SelectionManifest) {
                $completedOutput = Get-Item -LiteralPath $task.log_dir
                if (-not (Test-TaskOutput -Directory $completedOutput) -or
                    -not (Test-SelectedParameters -Directory $completedOutput -Task $task)) {
                    throw "Completed task $($task.id) does not match its recorded selected configuration."
                }
            }
            continue
        }

        if ($Resume -and $null -ne $task.started_at) {
            $existingStart = [datetime]::Parse($task.started_at)
            $existing = Find-TaskOutput -Task $task -StartedAt $existingStart
            if (Test-TaskOutput -Directory $existing) {
                $task.log_dir = $existing.FullName
                $task.status = "completed"
                $task.exit_code = 0
                if ($null -eq $task.completed_at) { $task.completed_at = (Get-Date).ToString("o") }
                if ($null -eq $task.duration_seconds) {
                    $task.duration_seconds = [math]::Round(((Get-Date) - $existingStart).TotalSeconds, 1)
                }
                $state.completed_tasks = @($tasks | Where-Object { $_.status -eq "completed" }).Count
                Save-State
                continue
            }
        }

        $taskStart = Get-Date
        $task.status = "running"
        $task.started_at = $taskStart.ToString("o")
        $task.command_log = Join-Path $taskLogRoot ($task.id + ".log")
        $state.current_task = $task.id
        Save-State

        & $Python -u $task.script @($task.arguments) 2>&1 |
            Tee-Object -FilePath $task.command_log
        $task.exit_code = $LASTEXITCODE
        if ($task.exit_code -ne 0) {
            throw "Task $($task.id) failed with exit code $($task.exit_code)."
        }

        $created = Find-TaskOutput -Task $task -StartedAt $taskStart
        if ($null -eq $created) { throw "Could not locate the output directory for $($task.id)." }
        if (-not (Test-TaskOutput -Directory $created)) { throw "Task $($task.id) has incomplete outputs." }

        $task.log_dir = $created.FullName
        $task.status = "completed"
        $task.completed_at = (Get-Date).ToString("o")
        $task.duration_seconds = [math]::Round(((Get-Date) - $taskStart).TotalSeconds, 1)
        $state.completed_tasks = @($tasks | Where-Object { $_.status -eq "completed" }).Count
        Save-State
    }

    $state.current_task = "aggregate_results"
    Save-State
    & $Python -u "scripts\data-val\aggregate_scaffold_results.py" --run-root $runRoot 2>&1 |
        Tee-Object -FilePath (Join-Path $taskLogRoot "aggregate_results.log")
    if ($LASTEXITCODE -ne 0) { throw "Result aggregation failed." }

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
