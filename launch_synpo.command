#!/bin/bash

# Finder launches .command files with a minimal environment, so this script
# discovers Conda without depending on PATH or a particular installation folder.

set -u

ENVIRONMENT_NAME="synpo-microscopy"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${SYNPO_LAUNCHER_CONFIG_DIR:-${HOME}/Library/Application Support/Synpo}"
CONFIG_FILE="${CONFIG_DIR}/launcher-conda.txt"
CONDA_PATHS=()
CONDA_SOURCES=()

resolve_conda() {
    candidate="$1"
    [ -n "$candidate" ] || return 1
    candidate="${candidate%\"}"
    candidate="${candidate#\"}"

    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        case "$(basename "$candidate")" in
            conda) printf '%s\n' "$candidate"; return 0 ;;
        esac
    fi

    if [ -d "$candidate" ]; then
        for executable in \
            "$candidate/bin/conda" \
            "$candidate/condabin/conda" \
            "$candidate/conda" \
            "$(dirname "$candidate")/bin/conda"
        do
            if [ -f "$executable" ] && [ -x "$executable" ]; then
                printf '%s\n' "$executable"
                return 0
            fi
        done
    fi
    return 1
}

add_candidate() {
    candidate="$(resolve_conda "$1" 2>/dev/null)" || return 0
    for existing in "${CONDA_PATHS[@]}"; do
        [ "$existing" = "$candidate" ] && return 0
    done
    "$candidate" --version >/dev/null 2>&1 || return 0
    CONDA_PATHS+=("$candidate")
    CONDA_SOURCES+=("$2")
}

saved_conda=""
if [ -f "$CONFIG_FILE" ]; then
    saved_conda="$(sed -n 's/^conda=//p' "$CONFIG_FILE" | head -n 1)"
fi

add_candidate "${CONDA_EXE:-}" "active"
add_candidate "$saved_conda" "saved"
path_conda="$(command -v conda 2>/dev/null || true)"
add_candidate "$path_conda" "PATH"

# A login shell can know Conda even when Finder's environment does not.
for shell_path in /bin/zsh /bin/bash; do
    if [ -x "$shell_path" ]; then
        shell_base="$("$shell_path" -lic 'conda info --base 2>/dev/null' 2>/dev/null | tail -n 1)"
        add_candidate "$shell_base" "shell initialization"
    fi
done

for root in \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "$HOME/miniforge3" \
    "$HOME/mambaforge" \
    "$HOME/opt/miniconda3" \
    "$HOME/opt/anaconda3" \
    "$HOME/opt/miniforge3" \
    "/opt/miniconda3" \
    "/opt/anaconda3" \
    "/opt/miniforge3" \
    "/usr/local/miniconda3" \
    "/usr/local/anaconda3" \
    "/usr/local/miniforge3" \
    "/opt/homebrew/Caskroom/miniconda/base" \
    "/opt/homebrew/Caskroom/miniforge/base"
do
    add_candidate "$root" "common location"
done

choose_conda_folder() {
    /usr/bin/osascript <<'APPLESCRIPT' 2>/dev/null
try
    set chosenFolder to choose folder with prompt "Select the Anaconda, Miniconda, Miniforge, or Mambaforge installation folder"
    return POSIX path of chosenFolder
on error number -128
    return ""
end try
APPLESCRIPT
}

if [ "${#CONDA_PATHS[@]}" -eq 0 ]; then
    printf '%s\n' "Conda was not found automatically. Please select its installation folder."
    selected_folder="$(choose_conda_folder)"
    if [ -z "$selected_folder" ]; then
        read -r -p "Enter the Conda installation folder, or press Return to cancel: " selected_folder
    fi
    add_candidate "$selected_folder" "selected"
fi

if [ "${#CONDA_PATHS[@]}" -eq 0 ]; then
    printf '%s\n' "Synpo launcher error: Conda was not found."
    printf '%s\n' "Install Anaconda, Miniconda, or Miniforge, then try again."
    read -r -p "Press Return to close." _
    exit 1
fi

environment_prefix() {
    conda_path="$1"
    marker="SYNPO_ENVIRONMENT_PREFIX="
    output="$("$conda_path" run -n "$ENVIRONMENT_NAME" python -c \
        "import sys; print('${marker}' + sys.prefix)" 2>/dev/null)" || return 1
    prefix="$(printf '%s\n' "$output" | sed -n "s/^${marker}//p" | tail -n 1)"
    [ -n "$prefix" ] && [ -x "$prefix/bin/python" ] || return 1
    printf '%s\n' "$prefix"
}

AVAILABLE_INDEXES=()
AVAILABLE_PREFIXES=()
for ((index=0; index<${#CONDA_PATHS[@]}; index++)); do
    prefix="$(environment_prefix "${CONDA_PATHS[$index]}")" || continue
    AVAILABLE_INDEXES+=("$index")
    AVAILABLE_PREFIXES+=("$prefix")
done

select_from_indexes() {
    prompt="$1"
    shift
    indexes=("$@")
    if [ "${#indexes[@]}" -eq 1 ]; then
        printf '%s\n' "${indexes[0]}"
        return
    fi
    printf '%s\n' "$prompt" >&2
    for ((display=0; display<${#indexes[@]}; display++)); do
        item="${indexes[$display]}"
        printf '  %d. %s [%s]\n' "$((display + 1))" \
            "${CONDA_PATHS[$item]}" "${CONDA_SOURCES[$item]}" >&2
    done
    while true; do
        read -r -p "Enter 1-${#indexes[@]}: " answer </dev/tty
        case "$answer" in
            ''|*[!0-9]*) ;;
            *)
                if [ "$answer" -ge 1 ] && [ "$answer" -le "${#indexes[@]}" ]; then
                    printf '%s\n' "${indexes[$((answer - 1))]}"
                    return
                fi
                ;;
        esac
    done
}

if [ "${#AVAILABLE_INDEXES[@]}" -eq 0 ]; then
    ALL_INDEXES=()
    for ((index=0; index<${#CONDA_PATHS[@]}; index++)); do
        ALL_INDEXES+=("$index")
    done
    selected_index="$(select_from_indexes \
        "Choose the Conda installation that should create Synpo's environment:" \
        "${ALL_INDEXES[@]}")"
    selected_conda="${CONDA_PATHS[$selected_index]}"

    macos_version="$(sw_vers -productVersion 2>/dev/null || printf '0')"
    architecture="$(uname -m)"
    case "$macos_version" in
        10.15*)
            if [ "$architecture" != "x86_64" ]; then
                printf '%s\n' "Synpo's Catalina environment supports Intel Macs only."
                exit 1
            fi
            environment_file="$SCRIPT_DIR/environment-macos-legacy.yml"
            environment_label="the Intel macOS Catalina environment"
            ;;
        11.*)
            environment_file="$SCRIPT_DIR/environment-macos-legacy.yml"
            environment_label="the legacy macOS 11 environment"
            ;;
        10.*)
            printf '%s\n' "Synpo supports macOS 10.15 Catalina, macOS 11, or macOS 12 and newer."
            exit 1
            ;;
        *)
            environment_file="$SCRIPT_DIR/environment.yml"
            environment_label="the current macOS environment"
            ;;
    esac

    printf "The '%s' environment was not found.\n" "$ENVIRONMENT_NAME"
    read -r -p "Create $environment_label now? [Y/n] " answer
    case "$answer" in
        [Nn]*) printf '%s\n' "No existing environment was changed."; exit 1 ;;
    esac
    "$selected_conda" env create --file "$environment_file" || {
        printf '%s\n' "Synpo launcher error: Conda could not create the environment."
        exit 1
    }
    selected_prefix="$(environment_prefix "$selected_conda")" || {
        printf '%s\n' "The environment was created, but its Python could not be located."
        exit 1
    }
else
    selected_index=""
    for ((available=0; available<${#AVAILABLE_INDEXES[@]}; available++)); do
        candidate_index="${AVAILABLE_INDEXES[$available]}"
        if [ "${CONDA_SOURCES[$candidate_index]}" = "active" ]; then
            selected_index="$candidate_index"
            selected_prefix="${AVAILABLE_PREFIXES[$available]}"
            break
        fi
    done
    if [ -z "$selected_index" ]; then
        for ((available=0; available<${#AVAILABLE_INDEXES[@]}; available++)); do
            candidate_index="${AVAILABLE_INDEXES[$available]}"
            if [ "${CONDA_SOURCES[$candidate_index]}" = "saved" ]; then
                selected_index="$candidate_index"
                selected_prefix="${AVAILABLE_PREFIXES[$available]}"
                break
            fi
        done
    fi
    if [ -z "$selected_index" ]; then
        selected_index="$(select_from_indexes \
            "Several Synpo environments were found. Choose one:" \
            "${AVAILABLE_INDEXES[@]}")"
        for ((available=0; available<${#AVAILABLE_INDEXES[@]}; available++)); do
            if [ "${AVAILABLE_INDEXES[$available]}" = "$selected_index" ]; then
                selected_prefix="${AVAILABLE_PREFIXES[$available]}"
                break
            fi
        done
    fi
    selected_conda="${CONDA_PATHS[$selected_index]}"
fi

mkdir -p "$CONFIG_DIR"
printf 'conda=%s\n' "$selected_conda" > "$CONFIG_FILE"

export PYTHONPATH="$SCRIPT_DIR/src"
cd "$SCRIPT_DIR"
"$selected_conda" run --no-capture-output -p "$selected_prefix" python -m synpo
status=$?
if [ "$status" -ne 0 ]; then
    printf '\nSynpo exited with status %d.\n' "$status"
    read -r -p "Press Return to close." _
fi
exit "$status"
