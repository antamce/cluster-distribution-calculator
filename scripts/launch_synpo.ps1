[CmdletBinding()]
param(
    [switch]$ResolveOnly,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$script:EnvironmentName = "synpo-microscopy"
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot
$script:EnvironmentFile = Join-Path $script:ProjectRoot "environment.yml"
$script:CandidatePaths = [System.Collections.Generic.List[string]]::new()
$script:CandidateSources = [System.Collections.Generic.List[string]]::new()
$script:SeenCandidates = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)

function Get-LauncherConfigPath {
    if (-not [string]::IsNullOrWhiteSpace($env:SYNPO_LAUNCHER_CONFIG_DIR)) {
        return Join-Path $env:SYNPO_LAUNCHER_CONFIG_DIR "launcher-conda.txt"
    }
    $applicationData = [Environment]::GetFolderPath("ApplicationData")
    if ([string]::IsNullOrWhiteSpace($applicationData)) {
        $applicationData = Join-Path $env:USERPROFILE "AppData\Roaming"
    }
    return Join-Path $applicationData "Synpo\launcher-conda.txt"
}

function Resolve-CondaExecutable {
    param([string]$Candidate)

    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $null
    }
    $expanded = [Environment]::ExpandEnvironmentVariables($Candidate.Trim().Trim('"'))
    $item = Get-Item -LiteralPath $expanded -ErrorAction SilentlyContinue
    if ($null -eq $item) {
        return $null
    }
    if (-not $item.PSIsContainer) {
        if ($item.Name -in @("conda.exe", "conda.bat")) {
            return $item.FullName
        }
        return $null
    }

    $directories = @($item.FullName)
    if ($item.Name -in @("Scripts", "condabin", "bin")) {
        $directories += $item.Parent.FullName
    }
    foreach ($directory in $directories) {
        foreach ($relativePath in @(
            "Scripts\conda.exe",
            "condabin\conda.bat",
            "bin\conda.exe",
            "conda.exe",
            "conda.bat"
        )) {
            $path = Join-Path $directory $relativePath
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                return (Get-Item -LiteralPath $path).FullName
            }
        }
    }
    return $null
}

function Add-CondaCandidate {
    param(
        [string]$Candidate,
        [string]$Source
    )

    $executable = Resolve-CondaExecutable $Candidate
    if ($null -eq $executable -or -not $script:SeenCandidates.Add($executable)) {
        return
    }
    $null = & $executable --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        return
    }
    $script:CandidatePaths.Add($executable)
    $script:CandidateSources.Add($Source)
}

function Get-SavedCondaPath {
    $configPath = Get-LauncherConfigPath
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        return $null
    }
    $line = Get-Content -LiteralPath $configPath -ErrorAction SilentlyContinue |
        Where-Object { $_ -like "conda=*" } |
        Select-Object -First 1
    if ($null -eq $line) {
        return $null
    }
    return $line.Substring("conda=".Length)
}

function Save-CondaPath {
    param([string]$CondaPath)

    $configPath = Get-LauncherConfigPath
    $configDirectory = Split-Path -Parent $configPath
    New-Item -ItemType Directory -Path $configDirectory -Force | Out-Null
    Set-Content -LiteralPath $configPath -Encoding UTF8 -Value "conda=$CondaPath"
}

function Add-AutomaticCandidates {
    Add-CondaCandidate $env:CONDA_EXE "active"
    Add-CondaCandidate (Get-SavedCondaPath) "saved"

    foreach ($commandName in @("conda.exe", "conda.bat", "conda")) {
        Get-Command $commandName -All -ErrorAction SilentlyContinue | ForEach-Object {
            Add-CondaCandidate $_.Source "PATH"
        }
    }

    $distributionNames = @("miniconda3", "anaconda3", "miniforge3", "mambaforge")
    $parents = @(
        $env:USERPROFILE,
        $env:LOCALAPPDATA,
        $env:ProgramData,
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)},
        (Join-Path $env:SystemDrive "tools")
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    foreach ($parent in $parents) {
        foreach ($name in $distributionNames) {
            Add-CondaCandidate (Join-Path $parent $name) "common location"
        }
    }

    $registeredEnvironments = Join-Path $env:USERPROFILE ".conda\environments.txt"
    if (Test-Path -LiteralPath $registeredEnvironments -PathType Leaf) {
        foreach ($prefix in Get-Content -LiteralPath $registeredEnvironments) {
            if ((Split-Path -Leaf $prefix.Trim()) -eq $script:EnvironmentName) {
                $envsDirectory = Split-Path -Parent $prefix.Trim()
                if ((Split-Path -Leaf $envsDirectory) -eq "envs") {
                    Add-CondaCandidate (Split-Path -Parent $envsDirectory) "environment registry"
                }
            }
        }
    }

    $registryRoot = "HKCU:\Software\Python\ContinuumAnalytics"
    if (Test-Path $registryRoot) {
        Get-ChildItem $registryRoot -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
            $properties = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
            foreach ($name in @("InstallPath", "ExecutablePath", "sys_prefix")) {
                Add-CondaCandidate $properties.$name "Windows registry"
            }
        }
    }
}

function Get-EnvironmentPrefix {
    param([string]$CondaPath)

    $marker = "SYNPO_ENVIRONMENT_PREFIX="
    $output = & $CondaPath run -n $script:EnvironmentName python -c (
        "import sys; print('" + $marker + "' + sys.prefix)"
    ) 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $line = $output | Where-Object { $_ -like "$marker*" } | Select-Object -Last 1
    if ($null -eq $line) {
        return $null
    }
    $prefix = $line.Substring($marker.Length).Trim()
    if (Test-Path -LiteralPath (Join-Path $prefix "python.exe") -PathType Leaf) {
        return $prefix
    }
    return $null
}

function Select-Index {
    param(
        [System.Collections.Generic.List[int]]$Indexes,
        [string]$Prompt
    )

    if ($Indexes.Count -eq 1 -or $NonInteractive) {
        return $Indexes[0]
    }
    Write-Host $Prompt
    for ($display = 0; $display -lt $Indexes.Count; $display++) {
        $index = $Indexes[$display]
        Write-Host "  $($display + 1). $($script:CandidatePaths[$index]) [$($script:CandidateSources[$index])]"
    }
    while ($true) {
        $answer = Read-Host "Enter 1-$($Indexes.Count)"
        $number = 0
        if ([int]::TryParse($answer, [ref]$number) -and $number -ge 1 -and $number -le $Indexes.Count) {
            return $Indexes[$number - 1]
        }
    }
}

function Select-CondaFolder {
    if ($NonInteractive) {
        return $null
    }
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = "Select the Anaconda, Miniconda, Miniforge, or Mambaforge installation folder"
        $dialog.ShowNewFolderButton = $false
        if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            return $dialog.SelectedPath
        }
    }
    catch {
        Write-Host "The graphical folder picker was unavailable: $($_.Exception.Message)"
        return (Read-Host "Enter the Conda installation folder, or leave blank to cancel")
    }
    return $null
}

function Confirm-EnvironmentCreation {
    param([string]$CondaPath)

    if ($NonInteractive) {
        return $false
    }
    Write-Host "The '$script:EnvironmentName' environment was not found."
    $answer = Read-Host "Create it now from environment.yml using $CondaPath? [Y/n]"
    return [string]::IsNullOrWhiteSpace($answer) -or $answer -match "^[Yy]"
}

function Resolve-SynpoRuntime {
    Add-AutomaticCandidates
    if ($script:CandidatePaths.Count -eq 0) {
        Write-Host "Conda was not found automatically. Please select its installation folder."
        $selectedFolder = Select-CondaFolder
        Add-CondaCandidate $selectedFolder "selected"
    }
    if ($script:CandidatePaths.Count -eq 0) {
        throw "Conda was not found. Install Anaconda, Miniconda, or Miniforge, then try again."
    }

    $environmentIndexes = [System.Collections.Generic.List[int]]::new()
    $prefixes = @{}
    for ($index = 0; $index -lt $script:CandidatePaths.Count; $index++) {
        $prefix = Get-EnvironmentPrefix $script:CandidatePaths[$index]
        if ($null -ne $prefix) {
            $environmentIndexes.Add($index)
            $prefixes[$index] = $prefix
        }
    }

    if ($environmentIndexes.Count -eq 0) {
        $allIndexes = [System.Collections.Generic.List[int]]::new()
        0..($script:CandidatePaths.Count - 1) | ForEach-Object { $allIndexes.Add($_) }
        $index = Select-Index $allIndexes "Choose the Conda installation that should create Synpo's environment:"
        $condaPath = $script:CandidatePaths[$index]
        if (-not (Confirm-EnvironmentCreation $condaPath)) {
            throw "The Synpo environment is required. No existing environment was changed."
        }
        & $condaPath env create --file $script:EnvironmentFile
        if ($LASTEXITCODE -ne 0) {
            throw "Conda could not create the Synpo environment."
        }
        $prefix = Get-EnvironmentPrefix $condaPath
        if ($null -eq $prefix) {
            throw "The environment was created, but its Python executable could not be located."
        }
    }
    else {
        $preferred = $environmentIndexes | Where-Object {
            $script:CandidateSources[$_] -eq "active"
        } | Select-Object -First 1
        if ($null -eq $preferred) {
            $preferred = $environmentIndexes | Where-Object {
                $script:CandidateSources[$_] -eq "saved"
            } | Select-Object -First 1
        }
        if ($null -eq $preferred) {
            $preferred = Select-Index $environmentIndexes "Several Synpo environments were found. Choose one:"
        }
        $index = [int]$preferred
        $condaPath = $script:CandidatePaths[$index]
        $prefix = $prefixes[$index]
    }

    Save-CondaPath $condaPath
    return [PSCustomObject]@{
        Conda = $condaPath
        Prefix = $prefix
    }
}

try {
    $runtime = Resolve-SynpoRuntime
    if ($ResolveOnly) {
        Write-Output "CONDA=$($runtime.Conda)"
        Write-Output "ENVIRONMENT=$($runtime.Prefix)"
        exit 0
    }

    $env:PYTHONPATH = Join-Path $script:ProjectRoot "src"
    & $runtime.Conda run --no-capture-output -p $runtime.Prefix python -m synpo
    exit $LASTEXITCODE
}
catch {
    Write-Host "Synpo launcher error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
