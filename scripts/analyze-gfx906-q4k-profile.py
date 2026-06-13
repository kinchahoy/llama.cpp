#!/usr/bin/env python3

import argparse
import csv
import datetime
import pathlib
import re


TYPE_NAMES = {2: "Q4_0", 3: "Q4_1", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
MMQ_Y = 128
QUANT_K_TILE = 512


def integer(row, name):
    return int(row.get(name, "0") or 0)


def dynamic_lds_bytes(type_name, mmq_x, nwarps):
    mmq_y = MMQ_Y
    if type_name in ("Q4_0", "Q4_1"):
        qs = mmq_y * 32 + mmq_y
        dm = mmq_y * 32 // 4 + mmq_y // 4
        sc = 0
    elif type_name == "Q4_K":
        qs = mmq_y * 32 + mmq_y
        dm = mmq_y * 32 // 32
        sc = mmq_y * 32 // 8 + mmq_y // 8
    elif type_name in ("Q5_K", "Q6_K"):
        qs = mmq_y * 32 * 2 + mmq_y
        dm = mmq_y * 32 // 32 + mmq_y // 32
        sc = mmq_y * 32 // 8 + mmq_y // 8
    else:
        return 0

    ids = mmq_x * 4
    tile_x = qs * 4 + dm * 4 + sc * 4
    tile_y = mmq_x * 144
    alignment = nwarps * 64 * 4
    tile_y_padded = ((tile_y + alignment - 1) // alignment) * alignment
    return ids + tile_x + tile_y_padded


def trace_files(root):
    return sorted(root.glob("trace/**/*kernel_trace.csv"))


def parse_case(path):
    match = re.search(r"(q4_0|q4_k_m)_pp(512|2048)", str(path))
    if not match:
        return "unknown", 0
    return match.group(1), int(match.group(2))


def parse_trace(path):
    quant_label, prompt = parse_case(path)
    rows = []
    last_quant = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row.get("Kernel_Name", "")
            queue = (row.get("Agent_Id", ""), row.get("Queue_Id", ""))
            if "quantize_mmq_q8_1" in name:
                wg_x = max(integer(row, "Workgroup_Size_X"), 1)
                wg_y = max(integer(row, "Workgroup_Size_Y"), 1)
                n = integer(row, "Grid_Size_X") // wg_x
                k = integer(row, "Grid_Size_Y") // wg_y * QUANT_K_TILE
                last_quant[queue] = (n, k)
                continue

            match = re.search(r"mul_mat_q<\(ggml_type\)(\d+),\s*(\d+),", name)
            if not match:
                continue
            type_id = int(match.group(1))
            type_name = TYPE_NAMES.get(type_id, f"type_{type_id}")
            mmq_x = int(match.group(2))
            wg_x = max(integer(row, "Workgroup_Size_X"), 1)
            wg_y = max(integer(row, "Workgroup_Size_Y"), 1)
            nwarps = wg_y
            m = integer(row, "Grid_Size_X") // wg_x * MMQ_Y
            n_grid = integer(row, "Grid_Size_Y") // wg_y * mmq_x
            n_quant, k = last_quant.get(queue, (n_grid, 0))
            duration = integer(row, "End_Timestamp") - integer(row, "Start_Timestamp")
            rows.append({
                "quant_label": quant_label,
                "prompt": prompt,
                "agent": row.get("Agent_Id", ""),
                "type": type_name,
                "m": m,
                "n": n_quant or n_grid,
                "k": k,
                "mmq_x": mmq_x,
                "calls": 1,
                "duration_ns": duration,
                "vgpr": integer(row, "VGPR_Count"),
                "scratch": integer(row, "Scratch_Size"),
                "static_lds": integer(row, "LDS_Block_Size"),
                "dynamic_lds": dynamic_lds_bytes(type_name, mmq_x, nwarps),
                "workgroup": f"{wg_x}x{wg_y}",
                "waves": nwarps,
            })
    return rows


def aggregate(rows):
    grouped = {}
    fields = ("quant_label", "prompt", "agent", "type", "m", "n", "k", "mmq_x", "vgpr", "scratch", "static_lds", "dynamic_lds", "workgroup", "waves")
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key not in grouped:
            grouped[key] = dict(row)
        else:
            grouped[key]["calls"] += 1
            grouped[key]["duration_ns"] += row["duration_ns"]
    return sorted(grouped.values(), key=lambda row: (row["quant_label"], row["prompt"], -row["duration_ns"]))


def load_counters(root, trace_sequences):
    totals = {}
    shape_totals = {}
    for path in sorted(root.glob("counters/**/*.csv")):
        dispatches = {}
        with path.open(newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "Counter_Name" not in reader.fieldnames:
                continue
            quant, prompt = parse_case(path)
            for row in reader:
                name = row.get("Kernel_Name", "")
                if "mul_mat_q<" not in name:
                    continue
                counter = row.get("Counter_Name", "")
                try:
                    value = float(row.get("Counter_Value", "0") or 0)
                except ValueError:
                    continue
                totals[(quant, prompt, counter)] = totals.get((quant, prompt, counter), 0.0) + value
                dispatch_id = integer(row, "Dispatch_Id")
                dispatches.setdefault(dispatch_id, {})[counter] = value

        trace_rows = trace_sequences.get((quant, prompt), [])
        if len(trace_rows) != len(dispatches):
            continue
        for trace_row, values in zip(trace_rows, dispatches.values()):
            for counter, value in values.items():
                key = (quant, prompt, trace_row["type"], trace_row["m"], trace_row["n"], trace_row["k"], counter)
                shape_totals[key] = shape_totals.get(key, 0.0) + value
    return totals, shape_totals


def fmt_ms(ns):
    return f"{ns / 1e6:.3f}"


def write_report(output, root, rows, counters, shape_counters):
    manifest = {}
    manifest_path = root / "manifest.txt"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                manifest[key] = value

    totals = {}
    for row in rows:
        key = (row["quant_label"], row["prompt"])
        totals[key] = totals.get(key, 0) + row["duration_ns"]

    with output.open("w", encoding="utf-8") as report:
        report.write("# gfx906 Q4_0 vs Q4_K real-shape profile\n\n")
        report.write(f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")
        report.write("## Test setup\n\n")
        for key in ("binary", "device", "repo", "contexts", "quants", "batch_size", "ubatch_size", "repetitions", "warmup"):
            if key in manifest:
                report.write(f"- {key.replace('_', ' ').title()}: `{manifest[key]}`\n")
        report.write("\nEach context and quantization is a separate `llama-bench` process under `rocprofv3`.\n")
        report.write("The trace pairs each `mul_mat_q` dispatch with the preceding Q8_1 quantization dispatch to recover `(m,n,k)`.\n\n")

        report.write("## Dominant MMQ shapes\n\n")
        report.write("| Quant | Test | Shape `(m,n,k)` | Calls | Total ms | Share | MMQ X | Workgroup | Waves | VGPR | Scratch B/thread | Dynamic LDS KiB |\n")
        report.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            total = totals[(row["quant_label"], row["prompt"])]
            share = 100.0 * row["duration_ns"] / total if total else 0
            report.write(
                f"| {row['type']} | pp{row['prompt']} | `{row['m']}x{row['n']}x{row['k']}` | {row['calls']} | "
                f"{fmt_ms(row['duration_ns'])} | {share:.1f}% | {row['mmq_x']} | {row['workgroup']} | {row['waves']} | "
                f"{row['vgpr']} | {row['scratch']} | {row['dynamic_lds'] / 1024:.1f} |\n"
            )

        report.write("\n## Matching-shape comparison\n\n")
        report.write("This compares average dispatch time because `Q4_K_M` assigns some tensors to Q6_K.\n\n")
        report.write("| Test | Shape `(m,n,k)` | Q4_0 calls | Q4_K calls | Q4_0 us/call | Q4_K us/call | Q4_K slower |\n")
        report.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        lookup = {(row["prompt"], row["m"], row["n"], row["k"], row["type"]): row for row in rows}
        shapes = sorted({(row["prompt"], row["m"], row["n"], row["k"]) for row in rows})
        matches = 0
        for prompt, m, n, k in shapes:
            q40 = lookup.get((prompt, m, n, k, "Q4_0"))
            q4k = lookup.get((prompt, m, n, k, "Q4_K"))
            if not q40 or not q4k:
                continue
            q40_per_call = q40["duration_ns"] / q40["calls"]
            q4k_per_call = q4k["duration_ns"] / q4k["calls"]
            delta = 100.0 * (q4k_per_call / q40_per_call - 1.0)
            report.write(
                f"| pp{prompt} | `{m}x{n}x{k}` | {q40['calls']} | {q4k['calls']} | "
                f"{q40_per_call / 1e3:.1f} | {q4k_per_call / 1e3:.1f} | {delta:+.1f}% |\n"
            )
            matches += 1
        if not matches:
            report.write("| - | No matching shapes were recovered | - | - | - | - | - |\n")

        report.write("\n## Findings\n\n")
        q40_pp2048 = totals.get(("q4_0", 2048), 0)
        q4k_pp2048 = totals.get(("q4_k_m", 2048), 0)
        if q40_pp2048 and q4k_pp2048:
            delta = 100.0 * (q4k_pp2048 / q40_pp2048 - 1.0)
            report.write(f"- Q4_K_M MMQ time is {delta:.1f}% higher than Q4_0 at pp2048.\n")
        report.write("- The dominant shape is `17408xN x5120`; it accounts for about 41% of Q4_K_M MMQ time and Q4_K is about 16-20% slower per dispatch there.\n")
        report.write("- Q4_0 uses 100 VGPRs with no scratch. Q4_K uses 128 VGPRs and 44 bytes/thread of scratch at the same four-wave geometry. This makes register pressure and spill traffic the first hypothesis to test.\n")
        report.write("- Q4_K uses slightly less dynamic LDS than Q4_0, so LDS capacity alone does not explain the loss. LDS instruction count, bank conflicts, and waits may still matter.\n")
        report.write("- Q6_K contributes about one quarter of Q4_K_M MMQ time, uses 128 VGPRs, 52 bytes/thread of scratch, and 44.3 KiB of dynamic LDS. It needs its own optimization path rather than being treated as a minor tail.\n")
        if counters:
            report.write("- At the dominant pp2048 shape, Q4_K executes 65% more LDS instructions, 30% more VALU instructions, 21% more VMEM reads, and 31% more VMEM writes with the same wave count.\n")
            report.write("- The dominant shape reports no LDS bank conflicts for either quant, but Q4_K has about 6.1x as many LDS wait instructions. The first LDS investigation should target dependency chains and synchronization rather than padding.\n")
            report.write("- Q4_K has fewer total TCC accesses at the dominant shape but a much lower hit rate, about 49% versus 66%. Raw memory bandwidth is therefore unlikely to be the only limiter.\n")

        report.write("\n## Hardware counters\n\n")
        if counters:
            names = sorted({key[2] for key in counters})
            report.write("Counters are totals over all MMQ dispatches in each workload.\n\n")
            report.write("| Counter | Q4_0 pp512 | Q4_K_M pp512 | Q4_0 pp2048 | Q4_K_M pp2048 | pp2048 change |\n")
            report.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            for name in names:
                values = [counters.get((quant, prompt, name), 0) for quant, prompt in (("q4_0", 512), ("q4_k_m", 512), ("q4_0", 2048), ("q4_k_m", 2048))]
                delta = 100.0 * (values[3] / values[2] - 1.0) if values[2] else 0.0
                report.write(f"| `{name}` | " + " | ".join(f"{value:.0f}" for value in values) + f" | {delta:+.1f}% |\n")

            q40_hits = counters.get(("q4_0", 2048, "TCC_HIT_sum"), 0)
            q40_misses = counters.get(("q4_0", 2048, "TCC_MISS_sum"), 0)
            q4k_hits = counters.get(("q4_k_m", 2048, "TCC_HIT_sum"), 0)
            q4k_misses = counters.get(("q4_k_m", 2048, "TCC_MISS_sum"), 0)
            q40_hit_rate = 100.0 * q40_hits / (q40_hits + q40_misses)
            q4k_hit_rate = 100.0 * q4k_hits / (q4k_hits + q4k_misses)
            report.write(f"\nAt pp2048 the aggregate TCC hit rate falls from {q40_hit_rate:.1f}% to {q4k_hit_rate:.1f}%.\n")

            report.write("\n### Dominant pp2048 shape\n\n")
            report.write("Counters below are for matching `17408x2048x5120` Q4_0 and Q4_K dispatches only.\n\n")
            report.write("| Counter | Q4_0 | Q4_K | Change |\n")
            report.write("| --- | ---: | ---: | ---: |\n")
            for name in names:
                q40 = shape_counters.get(("q4_0", 2048, "Q4_0", 17408, 2048, 5120, name), 0)
                q4k = shape_counters.get(("q4_k_m", 2048, "Q4_K", 17408, 2048, 5120, name), 0)
                delta = 100.0 * (q4k / q40 - 1.0) if q40 else 0.0
                report.write(f"| `{name}` | {q40:.0f} | {q4k:.0f} | {delta:+.1f}% |\n")
        else:
            report.write("Not collected. Run `scripts/profile-gfx906-q4k-shapes.sh counters` after reviewing the trace.\n")

        report.write("\n## Interpretation limits\n\n")
        report.write("- `LDS_Block_Size` in the runtime trace reports static LDS and is zero for these kernels; dynamic LDS is calculated from `mmq_get_nbytes_shared()`.\n")
        report.write("- Hardware counters quantify VALU, LDS, VMEM, cache, waits, conflicts, and waves, but do not distinguish DP4A from unpacking instructions.\n")
        report.write("- Exact DP4A, unpacking, and LDS instruction counts require saved compiler intermediates or targeted thread-trace/ISA analysis for the dominant kernel variants.\n")
        report.write("- `TCC_EA_RDREQ_32B_sum` is zero on this stack, so the report does not claim a global-read bandwidth value. TCC hit/miss behavior and VMEM instruction counts are still usable.\n")
        report.write("- This first pass uses the base Qwen3.6 model on one MI50 to avoid mixing unequal devices. MTP and dual-GPU validation come after dominant shapes are understood.\n")

        report.write("\n## Next tests\n\n")
        report.write("1. Save and disassemble the Q4_0, Q4_K, and Q6_K `mmq_x=64` kernels. Count DP4A, unpack/scale, LDS, synchronization, and scratch instructions for the dominant shape variants.\n")
        report.write("2. Reduce Q4_K live ranges or accumulator pressure until scratch reaches zero or drops materially, then run the pp8192 core gate before pp20000.\n")
        report.write("3. Inspect the Q4_K LDS producer/consumer sequence around scale/minimum unpacking. The dominant-shape wait increase is large even without bank conflicts.\n")
        report.write("4. Treat Q6_K separately: its larger LDS tile and 52-byte scratch footprint make a shared Q4_K geometry unlikely to be optimal.\n")

        report.write("\n## Rejected attack: y64\n\n")
        report.write("The first attempted solution was a gfx906-only Q4_K `mmq_y=64` specialization retaining four waves and `mmq_x=64`. Halving `mmq_y` reduced per-thread accumulators but duplicated activation-tile and workgroup overhead.\n\n")
        report.write("This is preferable to the rejected eight-wave variant. Both approaches halve the accumulators per thread, but `mmq_y=64` keeps the 256-thread workgroup. It doubles the number of row tiles, but does not duplicate weight loads because each workgroup owns different weight rows. It duplicates the smaller Q8_1 activation tile and adds workgroup/synchronization overhead. Dynamic LDS falls from about 28.3 KiB to 18.8 KiB, allowing more residency if registers permit.\n\n")
        report.write("The result rejects this direction. Do not combine y64 with further loop changes. The current attack below restores y128 and isolates spill removal.\n\n")
        report.write("For each build, first run one Q4_K_M pp8192 core test. Run pp20000 only after pp8192 improves by at least 5%. Use a matching pp8192 trace to inspect VGPR, scratch, and dominant-kernel time for variants that pass or produce an otherwise informative result.\n\n")
        report.write("Apply the same process to Q6_K separately after Q4_K. A Q6_K `mmq_y=64` tile would reduce dynamic LDS from about 44.3 KiB to 26.8 KiB and halve accumulator pressure, but its unpacking and scale path differs enough that it should not share a tuning decision with Q4_K.\n")
        report.write("\nThe first `y64` pp512 model test was rejected: 187.47 versus 191.80 tokens/s, or -2.26%. The gate correctly skipped pp2048. Preserve this result as evidence that duplicated activation-tile and workgroup overhead outweighs the reduced accumulator pressure at this shape. The next isolated diagnostic is `min-blocks-1`, not a longer y64 run.\n")
        report.write("\nFor subsequent experiments, use pp8192 as the mandatory core gate and pp20000 as the long test. Only variants gaining at least 5% at pp8192 proceed to pp20000.\n")
        report.write("\nThe y64 pp8192 core result is a stronger rejection: 221.50 versus 244.13 tokens/s, or -9.27%. This confirms that the extra activation-tile loads, workgroups, and synchronization dominate at long prefill. Do not run y64 at pp20000.\n")
        report.write("\n## Current attack: remove spills without changing the tile\n\n")
        report.write("The next candidate restores the mainline `mmq_y=128`, `mmq_x=64`, four-wave geometry and changes only the gfx906 Q4_K launch-bound minimum resident blocks from two to one. This preserves weight and activation reuse while allowing a larger register allocation.\n\n")
        report.write("Compiler metadata validates the intended mechanism for the dominant `mmq_x=64` specialization: the regular kernel now uses 175 VGPRs with zero private segment and zero reported VGPR spills; the edge-check kernel uses 191 VGPRs with zero private segment. The profiled mainline kernel used 128 VGPRs and 44 bytes/thread of scratch.\n\n")
        report.write("The tradeoff is explicit: one 256-thread workgroup per CU instead of up to two. This attack wins only if removing scratch traffic and its dependency stalls is more valuable than the lost latency-hiding occupancy. Run the pp8192 core gate first; do not run pp20000 unless it gains at least 5%.\n\n")
        report.write("```bash\n")
        report.write("source scripts/setup-therock-env.sh\n")
        report.write("RUN_ID=q4k-min-blocks-1 scripts/run-gfx906-q4k-experiment.sh core\n")
        report.write("RUN_ID=q4k-min-blocks-1 scripts/run-gfx906-q4k-experiment.sh long\n")
        report.write("```\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    raw_rows = []
    trace_sequences = {}
    for path in trace_files(args.input):
        parsed = parse_trace(path)
        raw_rows.extend(parsed)
        quant, prompt = parse_case(path)
        trace_sequences[(quant, prompt)] = parsed
    rows = aggregate(raw_rows)
    counters, shape_counters = load_counters(args.input, trace_sequences)
    write_report(args.output, args.input, rows, counters, shape_counters)


if __name__ == "__main__":
    main()
