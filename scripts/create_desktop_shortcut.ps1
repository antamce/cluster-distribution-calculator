param(
    [string]$ShortcutName = "Synpo Microscopy Processor"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "launch_synpo.bat"
$icon = Join-Path $projectRoot "src\synpo\assets\synpo.ico"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher not found: $launcher"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "$ShortcutName.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = "Launch Synpo through its dedicated Conda environment"
if (Test-Path -LiteralPath $icon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Save()

Write-Host "Created $shortcutPath"
