#!/usr/bin/env bash

interval=1
log="rocm-temps-$(date +%F-%H%M%S).csv"

echo "timestamp,device,edge,junction,memory" > "$log"

declare -A max_edge
declare -A max_junction
declare -A max_memory
declare -A cur_edge
declare -A cur_junction
declare -A cur_memory

while true; do
  ts="$(date '+%F %T')"
  csv="$(rocm-smi --showtemp --csv)"

  while IFS=, read -r device edge junction memory; do
    [[ "$device" == device || -z "$device" ]] && continue

    edge="${edge//[[:space:]]/}"
    junction="${junction//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"

    echo "$ts,$device,$edge,$junction,$memory" >> "$log"

    cur_edge[$device]="$edge"
    cur_junction[$device]="$junction"
    cur_memory[$device]="$memory"

    edge_i=${edge%.*}
    junction_i=${junction%.*}
    memory_i=${memory%.*}

    [[ -z "${max_edge[$device]}" || "$edge_i" -gt "${max_edge[$device]}" ]] && max_edge[$device]=$edge_i
    [[ -z "${max_junction[$device]}" || "$junction_i" -gt "${max_junction[$device]}" ]] && max_junction[$device]=$junction_i
    [[ -z "${max_memory[$device]}" || "$memory_i" -gt "${max_memory[$device]}" ]] && max_memory[$device]=$memory_i
  done <<< "$(tail -n +2 <<< "$csv")"

  clear
  echo "Logging to: $log"
  echo
  echo "=== Current / Max temps ==="

  for dev in $(printf '%s\n' "${!cur_edge[@]}" | sort); do
    echo "$dev:"
    echo "  edge:     ${cur_edge[$dev]}°C current, ${max_edge[$dev]}°C max"
    echo "  junction: ${cur_junction[$dev]}°C current, ${max_junction[$dev]}°C max"
    echo "  memory:   ${cur_memory[$dev]}°C current, ${max_memory[$dev]}°C max"
  done

  sleep "$interval"
done
