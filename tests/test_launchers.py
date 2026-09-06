from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_windows_batch_launcher_delegates_to_portable_resolver() -> None:
    launcher = (ROOT / "launch_synpo.bat").read_text(encoding="utf-8")

    assert "scripts\\launch_synpo.ps1" in launcher
    assert "%USERPROFILE%\\miniconda3" not in launcher
    assert "%USERPROFILE%\\anaconda3" not in launcher


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell parser test")
def test_windows_resolver_has_valid_powershell_syntax() -> None:
    script = ROOT / "scripts" / "launch_synpo.ps1"
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )

    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax test")
def test_macos_launcher_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["/bin/bash", "-n", str(ROOT / "launch_synpo.command")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_macos_environment_keeps_catalina_compatible_qt() -> None:
    environment = (ROOT / "environment-macos-legacy.yml").read_text(encoding="utf-8")

    assert "python=3.10" in environment
    assert "PySide6==6.2.4" in environment
    assert "pyside6>=6.7" not in environment.lower()


def test_macos_launcher_selects_legacy_and_current_environments() -> None:
    launcher = (ROOT / "launch_synpo.command").read_text(encoding="utf-8")

    assert "10.15*" in launcher
    assert "11.*" in launcher
    assert "environment-macos-legacy.yml" in launcher
    assert 'environment_file="$SCRIPT_DIR/environment.yml"' in launcher
    assert "conda info --base" in launcher
    assert "choose folder" in launcher
