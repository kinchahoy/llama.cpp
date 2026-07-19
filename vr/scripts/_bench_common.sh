#!/usr/bin/env bash

vr_gpu_max_temp() {
    local sensor="${1:-edge}"
    local devices="${2:-}"

    if ! command -v rocm-smi >/dev/null 2>&1; then
        return
    fi

    devices="${devices//ROCm/}"
    rocm-smi --showtemp 2>/dev/null |
        awk -v sensor="$sensor" -v devices="$devices" '
            $0 ~ ("Sensor " sensor) {
                gpu = $1
                gsub(/[^0-9]/, "", gpu)
                if (devices != "" && index("/" devices "/", "/" gpu "/") == 0) {
                    next
                }
                value = $NF
                gsub(/[^0-9.]/, "", value)
                if (value + 0 > maximum) {
                    maximum = value + 0
                }
                found = 1
            }
            END {
                if (found) {
                    printf "%.0f\n", maximum
                }
            }'
}

vr_wait_for_cool() {
    local target="$1"
    local sensor="${2:-edge}"
    local timeout="${3:-120}"
    local poll="${4:-3}"
    local devices="${5:-}"
    local temperature
    local waited=0

    [[ -n "$target" ]] || return

    while true; do
        temperature="$(vr_gpu_max_temp "$sensor" "$devices")"
        if [[ -z "$temperature" ]]; then
            echo "[thermal] temperature unavailable; continuing without a wait"
            return
        fi
        if (( temperature <= target )); then
            echo "[thermal] ${sensor}=${temperature}C target=${target}C waited=${waited}s"
            return
        fi
        if (( waited >= timeout )); then
            echo "[thermal] timeout: ${sensor}=${temperature}C target=${target}C waited=${waited}s"
            return
        fi
        sleep "$poll"
        waited=$((waited + poll))
    done
}

vr_capture_gpu_state() {
    local output="$1"
    local label="$2"

    {
        printf 'timestamp=%s label=%s\n' "$(date --iso-8601=seconds)" "$label"
        if command -v rocm-smi >/dev/null 2>&1; then
            rocm-smi -c -P -t 2>&1 || true
        else
            echo "rocm-smi unavailable"
        fi
    } >> "$output"
}

vr_capture_hardware() {
    local output="$1"

    {
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
        if command -v rocm-smi >/dev/null 2>&1; then
            rocm-smi --showproductname --showserial --showuniqueid --showbus 2>&1 || true
        else
            echo "rocm-smi unavailable"
        fi
    } > "$output"
}

vr_write_status() {
    local output_dir="$1"
    local state="$2"
    local current="$3"
    local completed="$4"
    local total="$5"
    local temporary="$output_dir/.status.$$"

    {
        printf 'state=%s\n' "$state"
        printf 'pid=%s\n' "$$"
        printf 'updated=%s\n' "$(date --iso-8601=seconds)"
        printf 'completed=%s\n' "$completed"
        printf 'total=%s\n' "$total"
        printf 'current=%s\n' "$current"
    } > "$temporary"
    mv "$temporary" "$output_dir/status"
}
