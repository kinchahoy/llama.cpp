# usage: python3 -I token_budget.py TRACE_CSV MODEL N_TOKENS [--plain BENCH_JSON] [--prof BENCH_JSON] [--json OUT_JSON]
#   TRACE_CSV : rocprofv3 --kernel-trace CSV of one llama-bench TG run (-p 0 -n N_TOKENS -r 1)
#   MODEL     : qwen35 (27B), qwen4exp (Flash Next) or gemma4
#   --plain   : llama-bench JSON of the same test without the profiler, for the scaled column
#   --plain-ts T,SD : instead of --plain, unprofiled t/s and stddev taken from a record (add --plain-note "(source)")
#   --prof    : llama-bench JSON written by the profiled run, printed for comparison
#   --id --commit --build --bin --modeldesc --settings [--measured] : provenance cells of the printed column
#                (--bin is the llama-bench binary to hash; Measured defaults to the trace file mtime)
# Splits each decode token (LM head end to LM head end) into model steps, per GPU and on the critical path.
# Each step has busy time (kernels) and idle-before time (the GPU gap before a kernel is charged to that kernel's step).
# Each GPU column sums to that GPU's mean token time. The critical path takes, for each AllReduce segment,
# the GPU that reached the AllReduce last; the small rest goes to the rounding row.
import csv
import hashlib
import json
import os
import re
import sys
import time

STEPS = [
    ("start_idle", "Token start: host gap, no kernel"),
    ("start_input", "Token start: input copies, embedding"),
    ("warmup", "Warm-up tokens: excess over steady tokens, spread over all"),
    ("norm", "Residual add + norm (Flash Next: hyper-connection combine + mix)"),
    ("ple", "Per-layer n-gram embedding (Flash Next)"),
    ("mix_in", "Mixer: input projections (mat-vec)"),
    ("attn_misc", "Attention: q/k norm, rope, KV write, gate"),
    ("attn_fa", "Attention: flash attention"),
    ("qsa", "Attention: QSA indexer (Flash Next)"),
    ("recr", "Recurrent: conv, gated delta net, gates, state"),
    ("mix_out", "Mixer: output projection (mat-vec)"),
    ("ar_mix", "AllReduce after mixer: exchange"),
    ("ar_mix_wait", "AllReduce after mixer: wait for other GPU"),
    ("ffn_gu", "FFN dense: gate/up mat-vec"),
    ("ffn_act", "FFN dense: activation"),
    ("ffn_down", "FFN dense: down mat-vec"),
    ("moe_router", "MoE: router + top-k"),
    ("moe_routed", "MoE: routed experts"),
    ("moe_shared", "MoE: shared expert + gate"),
    ("moe_reduce", "MoE: weighted reduce + combine"),
    ("ar_ffn", "AllReduce after FFN: exchange"),
    ("ar_ffn_wait", "AllReduce after FFN: wait for other GPU"),
    ("out_norm", "Output: final norm (+ softcap)"),
    ("out_head", "Output: LM head mat-vec"),
    ("q8", "q8_1 quantize of mat-vec inputs"),
    ("glue", "Glue: copies, crops, other"),
    ("idle", "Critical-path rounding (GPU switch at AllReduce ends)"),
]
KEYS = [k for k, _ in STEPS]


def base(name):
    n = name[5:] if name.startswith("void ") else name
    for ch in "<(":
        i = n.find(ch)
        if i != -1:
            n = n[:i]
    return n.strip()


def load(path):
    per = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g = f"{r['Grid_Size_X']}x{r['Grid_Size_Y']}x{r['Grid_Size_Z']}"
            per.setdefault(r["Agent_Id"], []).append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], g))
    for v in per.values():
        v.sort()
    return per


def is_mv(b):
    return b.startswith("mul_mat_vec")


def is_main_norm(name, b):
    return b == "add_rms_norm_mul_f32" or name.startswith("void rms_norm_f32<1024")


def classify_segment(seg, kind, model, head_grid):
    """seg: list of (name, grid). Returns a step key per kernel (AR kernels get 'ar')."""
    out = []
    mv_idx = [i for i, (n, g) in enumerate(seg) if is_mv(base(n)) and "__hip_bfloat16" not in n]
    last_mv = mv_idx[-1] if mv_idx else -1
    post_ar = True       # kernels right after the AllReduce, before the first projection
    seen_hc_post = False
    ple = None           # None, "ffn" or "mix"
    qsa = kind == "attn" and model == "qwen4exp"
    moe_phase = "router"
    for i, (n, g) in enumerate(seg):
        b = base(n)
        gy = g.split("x")[1]
        step = None
        if b == "ggml_cuda_ar_kernel":
            step = "ar"
        elif b in ("quantize_q8_1", "rp_quantize_q8_1_planar"):
            step = "q8"
        elif b == "softcap_f32":
            step = "out_norm"
        elif kind == "out":
            if is_mv(b) and g == head_grid:
                step = "out_head"
            else:
                step = "out_norm"
        if step is None and model == "qwen4exp":
            if post_ar and (b.startswith("dsv4_hc") or b in ("unary_op_kernel", "scale_f32")) and not seen_hc_post:
                step = "norm"
                if b == "dsv4_hc_post_f32":
                    seen_hc_post = True
            elif b == "dsv4_hc_post_f32":
                step = "norm"
                seen_hc_post = True
            elif ple == "mix" and b == "dsv4_hc_mix_prep_f32":
                ple = None
            elif kind != "ffn" and seen_hc_post and ple is None and n.startswith("void rms_norm_f32<1024") and not any(base(x[0]) == "dsv4_hc_mix_prep_f32" for x in seg[:i]):
                ple = "mix"
            elif kind == "ffn" and b == "k_get_rows_float" and g == "256x360x1":
                ple = "ffn"
            if step is None and ple is not None:
                step = "ple"
                if ple == "ffn" and n.startswith("void rms_norm_f32<1024"):
                    ple = None
            if step is None and b.startswith("dsv4_hc"):
                step = "norm"
            if step is None and b.startswith("flash_attn"):
                qsa = False
                step = "attn_fa"
            if step is None and qsa:
                if b.startswith("lightning") or b.startswith("top_k") or b in ("fill_kernel", "concat_cont") or "__hip_bfloat16" in n:
                    step = "qsa"
                elif is_mv(b):
                    qsa = False
            elif step is None and kind == "attn" and (n.startswith("void fill_kernel<__half") or (is_mv(b) and "__hip_bfloat16" in n)):
                qsa = True
                step = "qsa"
            if step is None and qsa and not is_mv(b):
                step = "qsa"
            if step is None and kind == "ffn":
                if b == "topk_moe_cuda" or (b == "mul_mat_vec_f" and g == "65536x1x1"):
                    step = "moe_router"
                elif is_mv(b) and gy != "1":
                    step = "moe_routed"
                    moe_phase = "routed"
                elif b == "moe_weighted_reduction_f32" or (b == "k_bin_bcast" and gy != "1"):
                    step = "moe_reduce"
                elif b == "k_bin_bcast":
                    step = "moe_reduce"
                elif is_mv(b) or b in ("unary_gated_op_kernel", "unary_op_kernel"):
                    step = "moe_shared" if moe_phase != "router" else "glue"
                else:
                    step = "glue"
        if step is None:
            if b == "__amd_rocclr_copyBuffer" or b == "__amd_rocclr_copyBufferRectAligned":
                step = "glue"
            elif is_main_norm(n, b) or (post_ar and model == "gemma4" and b == "k_bin_bcast"):
                step = "norm"
            elif is_mv(b):
                if kind == "ffn":
                    step = "ffn_down" if i == last_mv else "ffn_gu"
                else:
                    step = "mix_out" if i == last_mv else "mix_in"
            elif b.startswith("flash_attn"):
                step = "attn_fa"
            elif kind == "attn":
                step = "attn_misc"
            elif kind == "recr":
                step = "recr"
            elif kind == "ffn" and b in ("unary_gated_op_kernel", "unary_op_kernel"):
                step = "ffn_act"
            else:
                step = "glue"
        if step not in ("norm", "glue", "ar"):
            post_ar = False
        out.append(step)
    return out


START_OK = ("__amd_rocclr_copyBuffer", "k_bin_bcast", "scale_f32", "k_get_rows_float", "softcap_f32")


def analyze_token(ks, model, head_grid):
    """ks: kernels of one GPU for one token: (start, end, name, grid), first item is the previous head.
    Returns list of (segment index, step, ns) and AR (start, dur) list."""
    t0 = ks[0][1]
    body = ks[1:]
    ar_pos = [i for i, k in enumerate(body) if base(k[2]) == "ggml_cuda_ar_kernel"]
    bounds = [-1] + ar_pos + [len(body) - 1]
    steps = [None] * len(body)
    segs = []
    for s in range(len(bounds) - 1):
        lo, hi = bounds[s] + 1, bounds[s + 1]
        seg = body[lo:hi + 1]
        names = [x[2] for x in seg]
        if s == len(bounds) - 2:
            kind = "out"
        elif any("flash_attn" in x for x in names):
            kind = "attn"
        elif any("gated_delta_net" in x for x in names):
            kind = "recr"
        else:
            kind = "ffn"
        segs.append(kind)
        for j, st in enumerate(classify_segment([(x[2], x[3]) for x in seg], kind, model, head_grid)):
            if st == "ar":
                st = "ar_ffn" if kind == "ffn" else "ar_mix"
            steps[lo + j] = (s, st)
    items = []
    ars = []
    prev = t0
    in_start = True
    for i, (s, e, n, g) in enumerate(body):
        seg_i, st = steps[i]
        was_start = in_start
        if in_start and base(n) not in START_OK:
            in_start = False
        es = max(s, prev)
        if in_start and st != "out_norm":
            st = "start_input"
        if es > prev:
            # a gap is charged to the step that waits to start after it; gaps up to the first compute kernel are token start
            items.append((seg_i, "start_idle" if was_start else st, es - prev, "i"))
        d = max(0, e - es)
        if st in ("ar_mix", "ar_ffn"):
            ars.append((es, d, len(items), st))
        items.append((seg_i, st, d, "b"))
        prev = max(prev, e)
    return items, ars, segs


def main():
    args = sys.argv[1:]
    opts = {}
    pos = []
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            opts[args[i][2:]] = args[i + 1]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    path, model, n_tok = pos[0], pos[1], int(pos[2])
    per = load(path)
    agents = sorted(a for a in per if per[a] and any("mul_mat_vec" in k[2] for k in per[a]))
    gpu_names = {a: f"ROCm{j}" for j, a in enumerate(agents)}

    heads = {}
    head_grid = None
    for a in agents:
        mv = [k for k in per[a] if "mul_mat_vec" in k[2]]
        gmax = max(int(k[3].split("x")[0]) for k in mv)
        head_grid = f"{gmax}x1x1"
        idx = [j for j, k in enumerate(per[a]) if "mul_mat_vec" in k[2] and k[3] == head_grid]
        heads[a] = idx[-n_tok:]
    n_int = n_tok - 1

    zero = lambda: {k: [0, 0] for k in KEYS}     # [busy, idle-before] ns
    tot = {a: zero() for a in agents}
    crit = zero()
    span = {a: 0 for a in agents}
    nk = {a: 0 for a in agents}
    n_ar = 0
    seg_kinds = None
    fa_ns = {a: 0 for a in agents}
    topk_ns = {a: 0 for a in agents}
    mv_ns = {a: 0 for a in agents}
    mv_n = {a: 0 for a in agents}
    # warm-up tokens (eager launches before the HIP graph is captured) are much longer than the rest;
    # they are kept out of the step rows and shown as one row of excess time
    a0 = agents[0]
    ivl = [per[a0][heads[a0][t]][1] - per[a0][heads[a0][t - 1]][1] for t in range(1, n_tok)]
    med = sorted(ivl)[len(ivl) // 2]
    warm = {t for t in range(1, n_tok) if ivl[t - 1] > 1.2 * med}
    n_steady = n_int - len(warm)
    span_all = {a: per[a][heads[a][-1]][1] - per[a][heads[a][0]][1] for a in agents}
    for t in range(1, n_tok):
        if t in warm:
            continue
        res = {}
        for a in agents:
            lo, hi = heads[a][t - 1], heads[a][t]
            ks = per[a][lo:hi + 1]
            items, ars, segs = analyze_token(ks, model, head_grid)
            res[a] = (items, ars, ks)
            span[a] += ks[-1][1] - ks[0][1]
            nk[a] += len(ks) - 1
            for s, e, n, g in ks[1:]:
                if "flash_attn" in n:
                    fa_ns[a] += e - s
                if "topk_moe" in n:
                    topk_ns[a] += e - s
                if "mul_mat_vec" in n:
                    mv_ns[a] += e - s
                    mv_n[a] += 1
            seg_kinds = segs
        nars = {len(res[a][1]) for a in agents}
        if len(nars) != 1:
            sys.exit(f"token {t}: AllReduce count differs between GPUs: {nars}")
        n_ar = nars.pop()
        # AR wait: AR kernel time before the other GPU started the same AllReduce
        for a in agents:
            items, ars, _ = res[a]
            for k, (es, d, idx, st) in enumerate(ars):
                other = max(res[b][1][k][0] for b in agents if b != a)
                w = min(d, max(0, other - es))
                seg_i = items[idx][0]
                items[idx] = (seg_i, st, d - w, "b")
                items.append((seg_i, st + "_wait", w, "b"))
            for _, st, ns, kind in items:
                tot[a][st][kind == "i"] += ns
        # critical path: per segment, the GPU that started that AllReduce last; last segment: later head end
        late = [max(agents, key=lambda a: res[a][1][k][0]) for k in range(n_ar)]
        late.append(max(agents, key=lambda a: res[a][2][-1][1]))
        for a in agents:
            for seg_i, st, ns, kind in res[a][0]:
                if late[seg_i] == a:
                    crit[st][kind == "i"] += ns

    ref = span[a0]
    rounding = ref - sum(b + i for b, i in crit.values())
    crit["idle"][1] += rounding
    # rows are steady-token means; the warm-up row adds the warm-up excess spread over all tokens
    ms = lambda ns: ns / 1e6 / n_steady
    for a in agents:
        excess = (span_all[a] - span[a]) - len(warm) * span[a] / n_steady
        tot[a]["warmup"][1] = excess * n_steady / n_int
    crit["warmup"][1] = tot[a0]["warmup"][1]
    tok_ms = span_all[a0] / 1e6 / n_int
    steady_ms = ms(ref)

    plain = json.load(open(opts["plain"]))[0] if "plain" in opts else None
    if "plain-ts" in opts:
        # no plain JSON next to the trace: t/s and stddev from a record, e.g. --plain-ts 37.45,0.21
        ts, sd = (opts["plain-ts"].split(",") + ["0"])[:2]
        plain = {"avg_ts": float(ts), "stddev_ts": float(sd)}
    prof = json.load(open(opts["prof"]))[0] if "prof" in opts else None
    plain_ms = 1000.0 / plain["avg_ts"] if plain else None
    prof_ms = 1000.0 / prof["avg_ts"] if prof else None
    scale = plain_ms / tok_ms if plain_ms else None

    print(f"trace: {path}")
    print(f"model: {model}, decode tokens in trace: {n_tok}, token intervals: {n_int}, warm-up intervals (> 1.2 x median {med / 1e6:.2f} ms): "
          f"{len(warm)} {sorted(round(ivl[t - 1] / 1e6, 1) for t in warm)}, LM head grid {head_grid}")
    print(f"all-token mean {tok_ms:.3f} ms/token, steady-token mean {steady_ms:.3f} ms/token (ROCm0)")
    for a in agents:
        print(f"{gpu_names[a]} ({a}) steady: {ms(span[a]):.3f} ms/token, {nk[a] / n_steady:.1f} kernels/token, mat-vec {mv_n[a] / n_steady:.1f} calls {ms(mv_ns[a]):.3f} ms, "
              f"flash_attn* {ms(fa_ns[a]):.3f} ms, topk_moe {ms(topk_ns[a]):.3f} ms")
    kinds = {k: seg_kinds.count(k) for k in ("attn", "recr", "ffn", "out")}
    print(f"AllReduce per token: {n_ar}; segments: {kinds}")
    if prof_ms:
        print(f"llama-bench profiled: {prof_ms:.3f} ms/token (includes the first token)")
    if scale:
        print(f"llama-bench unprofiled: {plain['avg_ts']:.2f} t/s = {plain_ms:.3f} ms/token; scaled = critical x {scale:.4f}")
    print(f"critical-path rounding added to idle: {ms(rounding):.4f} ms")
    print()
    print("ms/token, busy + idle-before (a gap is charged to the step that follows it)")
    hdr = f"{'step':<66} " + " ".join(f"{gpu_names[a]:>15}" for a in agents) + f" {'critical':>15} {'sum':>7} {'%':>5}"
    if scale:
        hdr += f" {'scaled':>7}"
    print(hdr)
    pair = lambda v: f"{ms(v[0]):6.3f} + {ms(v[1]):5.3f}"
    for k, label in STEPS:
        c = crit[k]
        if max(ms(sum(tot[a][k])) for a in agents) < 0.0005 and ms(sum(c)) < 0.0005:
            continue
        line = f"{label:<66} " + " ".join(f"{pair(tot[a][k]):>15}" for a in agents) + f" {pair(c):>15} {ms(sum(c)):7.3f} {100 * ms(sum(c)) / tok_ms:5.1f}"
        if scale:
            line += f" {ms(sum(c)) * scale:7.3f}"
        print(line)
    tb = lambda d: [sum(v[0] for v in d.values()), sum(v[1] for v in d.values())]
    line = f"{'total':<66} " + " ".join(f"{pair(tb(tot[a])):>15}" for a in agents) + f" {pair(tb(crit)):>15} {ms(sum(tb(crit))):7.3f} {100.0:5.1f}"
    if scale:
        line += f" {ms(sum(tb(crit))) * scale:7.3f}"
    print(line)

    # paste-ready column for vr/TOKEN-BUDGET.md: one cell per row, in the table's row order
    measured = opts.get("measured") or time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(os.path.getmtime(path)))
    bin_sha = "-"
    if "bin" in opts:
        # llama-bench is a small launcher; libggml-hip.so next to it holds the kernels
        shas = []
        for p in (opts["bin"], os.path.join(os.path.dirname(opts["bin"]), "libggml-hip.so")):
            if os.path.exists(p):
                with open(p, "rb") as f:
                    shas.append(hashlib.sha256(f.read()).hexdigest()[:12])
        bin_sha = " / ".join(shas)
    plain_cell = f"{plain['avg_ts']:.2f} +- {plain['stddev_ts']:.2f} ({plain_ms:.2f})" if plain else "-"
    if plain and "plain-note" in opts:
        plain_cell += " " + opts["plain-note"]
    head_rows = [
        ("Column id", opts.get("id", "-")),
        ("Measured", measured),
        ("Source commit", opts.get("commit", "-")),
        ("Build (sha256 llama-bench / libggml-hip.so)", f"`{opts.get('build', '-')}` (`{bin_sha}`)"),
        ("Model, quant, file revision", opts.get("modeldesc", "-")),
        ("Settings", opts.get("settings", "-")),
        ("Unprofiled TG t/s (ms/token)", plain_cell),
        ("Profiled ms/token: all tokens; steady; llama-bench", f"{tok_ms:.2f}; {steady_ms:.2f}; " + (f"{prof_ms:.2f}" if prof_ms else "-")),
        ("Kernels / AllReduces per token", f"{nk[agents[0]] / n_steady:.0f} / {n_ar}"),
    ]
    print()
    print("COLUMN (append each cell to the matching row of the main table)")
    for label, v in head_rows:
        print(f"{label} | {v}")
    for k, label in STEPS:
        c = crit[k]
        cell = "-" if ms(sum(c)) < 0.005 else f"{ms(c[0]):.2f} + {ms(c[1]):.2f} ({100 * ms(sum(c)) / tok_ms:.0f}%)"
        print(f"{label} | {cell}")
    print(f"Total, profiled | {ms(sum(tb(crit))):.2f}")
    if scale:
        print(f"Total, scaled to unprofiled | {plain_ms:.2f}")

    if "json" in opts:
        out = {
            "trace": path, "model": model, "n_tokens": n_tok, "intervals": n_int, "warmup_intervals": len(warm), "steady_ms": steady_ms, "head_grid": head_grid,
            "ar_per_token": n_ar, "segments": kinds, "token_ms": tok_ms, "bench_prof_ms": prof_ms, "bench_plain_ms": plain_ms,
            "provenance": dict(head_rows),
            "gpus": {gpu_names[a]: {"token_ms": ms(span[a]), "kernels": nk[a] / n_steady, "matvec_calls": mv_n[a] / n_steady,
                                    "matvec_ms": ms(mv_ns[a]), "flash_attn_ms": ms(fa_ns[a]), "topk_moe_ms": ms(topk_ns[a]),
                                    "steps_busy_idle": {k: [ms(v[0]), ms(v[1])] for k, v in tot[a].items()}} for a in agents},
            "critical_busy_idle": {k: [ms(v[0]), ms(v[1])] for k, v in crit.items()},
            "scale": scale,
        }
        with open(opts["json"], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

