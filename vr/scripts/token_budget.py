# usage: python3 -I token_budget.py TRACE_CSV MODEL N_TOKENS [--phase tg|pp] [--plain BENCH_JSON] [--prof BENCH_JSON] [--json OUT_JSON]
#            [--append-csv DIR --record LINK [--notes TEXT]]
#        python3 -I token_budget.py append-json OUT_JSON --record LINK [--notes TEXT] [--dir DIR]
#        python3 -I token_budget.py render [--doc DOC_MD] [--dir DIR]
#   TRACE_CSV : rocprofv3 --kernel-trace CSV of one llama-bench run, -r 1
#               TG: -p 0 -n N_TOKENS; PP (--phase pp): -p N_TOKENS -n 0, with t_memory_copy_trace.csv next to it
#   MODEL     : qwen35 (27B), qwen4exp (Flash Next) or gemma4
#   --plain   : llama-bench JSON of the same test without the profiler, for the scaled column
#   --plain-ts T,SD : instead of --plain, unprofiled t/s and stddev taken from a record (add --plain-note "(source)")
#   --prof    : llama-bench JSON written by the profiled run (PP: also gives -p/-b/-ub/-d/-r and the bench time)
#   --id --commit --build --bin --modeldesc --settings [--measured] : provenance cells of the printed column
#                (--bin is the llama-bench binary to hash; Measured defaults to the trace file mtime)
#   --append-csv DIR : append the datapoint to DIR/token-budget-columns.csv and DIR/token-budget-steps.csv (refuses a known id)
#   append-json : append a datapoint from a JSON written by --json (used to backfill old columns)
#   render    : rewrite the tables between <!-- tb:NAME --> and <!-- /tb:NAME --> in DOC_MD from the CSVs
# TG: splits each decode token (LM head end to LM head end) into model steps, per GPU and on the critical path.
# Each step has busy time (kernels) and idle-before time (the GPU gap before a kernel is charged to that kernel's step).
# Each GPU column sums to that GPU's mean token time. The critical path takes, for each AllReduce segment,
# the GPU that reached the AllReduce last; the small rest goes to the rounding row.
# PP: see pp_main; the same split per ubatch, with time unions because kernels and copies overlap on several streams.
import bisect
import csv
import math
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
    if not ar_pos:
        # single GPU: no AllReduce, end each segment before the residual add + norm where TP would reduce
        # (the residual add is fused into the 1024-thread rms_norm here; the first one of the token is the input norm)
        norms = [i for i, k in enumerate(body) if k[2].startswith("void rms_norm_f32<1024") or base(k[2]) == "add_rms_norm_mul_f32"]
        ar_pos = [i - 1 for i in norms[1:]]
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
    if pos[0] == "render":
        return render(opts)
    if pos[0] == "append-json":
        with open(pos[1]) as f:
            d = json.load(f)
        return append_csv(opts.get("dir", DEF_DIR), *csv_rows(d, opts))
    if opts.get("phase") == "pp":
        return pp_main(pos, opts)
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
        if len(agents) == 1:
            late = [a0] * len(seg_kinds)
        else:
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

    if "json" in opts or "append-csv" in opts:
        out = {
            "phase": "tg", "plain_ts": plain["avg_ts"] if plain else None, "plain_sd": plain["stddev_ts"] if plain else None,
            "plain_note": opts.get("plain-note", ""),
            "trace": path, "model": model, "n_tokens": n_tok, "intervals": n_int, "warmup_intervals": len(warm), "steady_ms": steady_ms, "head_grid": head_grid,
            "ar_per_token": n_ar, "segments": kinds, "token_ms": tok_ms, "bench_prof_ms": prof_ms, "bench_plain_ms": plain_ms,
            "provenance": dict(head_rows),
            "gpus": {gpu_names[a]: {"token_ms": ms(span[a]), "kernels": nk[a] / n_steady, "matvec_calls": mv_n[a] / n_steady,
                                    "matvec_ms": ms(mv_ns[a]), "flash_attn_ms": ms(fa_ns[a]), "topk_moe_ms": ms(topk_ns[a]),
                                    "steps_busy_idle": {k: [ms(v[0]), ms(v[1])] for k, v in tot[a].items()}} for a in agents},
            "critical_busy_idle": {k: [ms(v[0]), ms(v[1])] for k, v in crit.items()},
            "scale": scale,
        }
        if "json" in opts:
            with open(opts["json"], "w") as f:
                json.dump(out, f, indent=1)
        if "append-csv" in opts:
            append_csv(opts["append-csv"], *csv_rows(out, opts))


# ---------------------------------------------------------------- prompt processing (PP)

PP_STEPS = [
    ("run_start", "Run start: bench timer start to the first compute kernel (host prep, input upload)"),
    ("start_idle", "Ubatch start: host gap before the first compute kernel"),
    ("start_input", "Ubatch start: input upload, embedding"),
    ("norm", "Residual add + norm"),
    ("hc", "Hyper-connection pre/post with its small GEMMs (Flash Next)"),
    ("ple", "Per-layer n-gram embedding (Flash Next)"),
    ("mix_in_attn", "Mixer GEMM: attention input projections (q+gate, k, v)"),
    ("mix_in_recr", "Mixer GEMM: GDN input projections (qkv, z, beta/alpha)"),
    ("attn_misc", "Attention: q/k norm, rope, KV write, gate"),
    ("attn_fa", "Attention: flash attention (batched)"),
    ("qsa", "Attention: QSA indexer and key pooling (Flash Next)"),
    ("recr_gdn", "Recurrent: chunked gated delta net"),
    ("recr", "Recurrent: conv, l2 norms, gates, state copies"),
    ("mix_out", "Mixer GEMM: output projection (slabs)"),
    ("ar_mix", "AllReduce after mixer: exposed"),
    ("ffn_gu", "FFN dense GEMM: gate/up"),
    ("ffn_act", "FFN dense: activation"),
    ("ffn_down", "FFN dense GEMM: down (slabs)"),
    ("moe_router", "MoE: router + top-k"),
    ("moe_ids", "MoE: ids helper, tile list (repack_tile_off)"),
    ("moe_routed", "MoE: routed expert GEMM + activation"),
    ("moe_shared", "MoE: shared expert + gate"),
    ("moe_reduce", "MoE: weighted reduce + combine"),
    ("ar_ffn", "AllReduce after FFN: exposed"),
    ("out_norm", "Output: last-token rows, final norm (+ softcap)"),
    ("out_head", "Output: LM head (last token)"),
    ("q8", "q8_1 quantize of GEMM inputs"),
    ("copy", "Copies H2D/D2H outside AllReduce, exposed"),
    ("glue", "Glue: crops, copies, other"),
    ("idle", "Critical-path rounding (GPU switch at AllReduce ends)"),
]
PP_INFO = [("ar_overlap", "AllReduce overlapped with compute (info, not in total)")]
PP_KEYS = [k for k, _ in PP_STEPS] + [k for k, _ in PP_INFO]
AR_MARK = ("ggml_cuda_ar_add_kernel", "ggml_cuda_ar_kernel")
QSA_NAMES = ("lightning", "k_argsort", "top_k", "fill_kernel", "concat_cont", "__amd_rocclr_copyBufferRect", "convert_unary", "soft_max")


def is_gemm(b, name):
    return b.startswith(("mmq_gemm", "mul_mat_q", "mul_mat_f", "mul_mat_vec", "sgemm_f32", "gemm")) or "Cijk_" in name


def is_q8(b):
    return b.startswith("quantize_")


def classify_pp(seg, kind, model, first):
    """seg: compute kernels of one AllReduce segment of one ubatch in stream order, as (name, grid).
    Returns a PP step key per kernel."""
    n = len(seg)
    bs = [base(x[0]) for x in seg]
    gemm = [is_gemm(bs[i], seg[i][0]) for i in range(n)]
    # trailing GEMMs (slabs of the output or down projection) come after the last other compute kernel
    last_other = max([i for i in range(n) if not gemm[i] and not is_q8(bs[i])], default=-1)
    hc_end = -1
    if model == "qwen4exp" and kind != "out":
        hc_end = next((i for i in range(n) if bs[i] == "dsv4_hc_pre_f32"), -1)
    head = -1
    if kind == "out":
        gs = [(int(seg[i][1].split("x")[0]), i) for i in range(n) if gemm[i]]
        head = max(gs)[1] if gs else -1
    out = []
    in_start = first
    post_ar = True
    seen_post = False
    ple = False
    moe = "router"
    qsa = True
    for i, (nm, g) in enumerate(seg):
        b = bs[i]
        step = None
        if in_start:
            if b.startswith("__amd_rocclr_copyBuffer") or b in ("scale_f32", "k_get_rows_float", "k_bin_bcast"):
                step = "start_input"
            else:
                in_start = False
        if step is not None:
            pass
        elif is_q8(b):
            step = "q8"
        elif kind == "out":
            step = "out_head" if i == head else "out_norm"
        elif i <= hc_end:
            # Flash Next: hyper-connection post/pre around the layer input, with the per-layer n-gram block inside it once
            if b == "dsv4_hc_post_f32":
                seen_post = True
            if ple:
                if b.startswith("rms_norm_f32") and i + 1 < n and is_q8(bs[i + 1]):
                    ple = False
                    step = "hc"
                else:
                    step = "ple"
            elif seen_post and (b in ("k_bin_bcast", "reduce_rows_f32", "op_clamp_kernel", "cpy_scalar") or b.startswith(("concat", "__amd_rocclr_copyBuffer"))):
                ple = True
                step = "ple"
            else:
                step = "hc"
        elif model == "qwen4exp" and kind == "ffn":
            if moe == "router":
                if b == "mm_ids_helper":
                    step, moe = "moe_ids", "routed"
                else:
                    step = "moe_router"
            elif moe == "routed":
                if b in ("mm_ids_helper", "repack_tile_off"):
                    step = "moe_ids"
                elif b in ("k_bin_bcast", "moe_weighted_reduction_f32"):
                    step, moe = "moe_reduce", "reduce"
                else:
                    step = "moe_routed"
            elif moe == "reduce":
                if b == "k_get_rows_float" and g == "256x360x1":
                    step, moe = "ple", "ple"
                elif gemm[i]:
                    step, moe = "moe_shared", "shared"
                else:
                    step = "moe_reduce"
            elif moe == "ple":
                step = "ple"
                if b.startswith("rms_norm_f32"):
                    moe = "shared"
            else:
                step = "moe_reduce" if b in ("k_bin_bcast", "moe_weighted_reduction_f32") else "moe_shared"
        elif model == "qwen4exp" and kind == "attn" and not b.startswith("flash_attn"):
            if gemm[i] and "Cijk_" not in nm:
                qsa = False
                step = "mix_out" if i > last_other else "mix_in_attn"
            elif b.startswith(QSA_NAMES) or "Cijk_" in nm:
                qsa = True
                step = "qsa"
            else:
                step = "qsa" if qsa else "attn_misc"
        if step is None:
            if post_ar and (is_main_norm(nm, b) or (model == "gemma4" and b == "k_bin_bcast")):
                step = "norm"
            elif gemm[i]:
                if kind == "ffn":
                    step = "ffn_down" if i > last_other else "ffn_gu"
                else:
                    step = "mix_out" if i > last_other else ("mix_in_attn" if kind == "attn" else "mix_in_recr")
            elif b.startswith("flash_attn"):
                step = "attn_fa"
                qsa = False
            elif kind == "attn":
                step = "attn_misc"
            elif kind == "recr":
                step = "recr_gdn" if b.startswith("gated_delta_net") else "recr"
            elif kind == "ffn" and b.startswith("unary"):
                step = "ffn_act"
            else:
                step = "glue"
        if step not in ("norm", "glue", "start_input", "q8"):
            post_ar = False
        out.append(step)
    return out


def load_pp(path):
    """Kernels and memory copies per GPU agent: (start, end, name, grid, stream, is_copy)."""
    per = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g = f"{r['Grid_Size_X']}x{r['Grid_Size_Y']}x{r['Grid_Size_Z']}"
            per.setdefault(r["Agent_Id"], []).append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], g, "k" + r["Stream_Id"], False))
    cpath = os.path.join(os.path.dirname(path), os.path.basename(path).replace("kernel_trace", "memory_copy_trace"))
    if not os.path.exists(cpath):
        sys.exit(f"PP mode needs the memory copy trace: {cpath} (rocprofv3 --memory-copy-trace)")
    with open(cpath, newline="") as f:
        for r in csv.DictReader(f):
            d2h = "DEVICE_TO_HOST" in r["Direction"]
            a = r["Source_Agent_Id"] if d2h else r["Destination_Agent_Id"]
            if a in per:
                per[a].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), "D2H" if d2h else "H2D", "", "c" + r["Stream_Id"], True))
    for v in per.values():
        v.sort()
    return per


def n_ubatches(n, b, ub):
    cnt = 0
    while n > 0:
        t = min(n, b)
        cnt += math.ceil(t / ub)
        n -= t
    return cnt


def union_gaps(acts):
    """acts sorted by start. Returns the gaps (start, end) between merged busy intervals."""
    gaps = []
    end = acts[0][1]
    for s, e, *_ in acts[1:]:
        if s > end:
            gaps.append((end, s))
        end = max(end, e)
    return gaps


def pp_ubatch_windows(acts, marks, n_total, n_meas):
    """Cut one GPU's timeline into ubatches. marks: AllReduce marker kernels in time order.
    Between the last AllReduce of a ubatch and the first of the next, the largest gap with no kernel and no copy
    is the host gap between the two graphs. Only the measured ubatches are cut: among the marker pairs in the last
    (n_meas + 0.5) ubatches' worth of markers, the n_meas pairs with the largest gaps start the measured ubatches.
    Returns windows (gap_start, first_activity, end, markers) and the cut quality."""
    gaps = union_gaps(acts)
    gs = [g[0] for g in gaps]
    per_ub = len(marks) / n_total
    lo_j = max(0, int(len(marks) - (n_meas + 0.5) * per_ub))
    best = []
    for j in range(lo_j, len(marks) - 1):
        lo = bisect.bisect_left(gs, marks[j][1])
        hi = bisect.bisect_left(gs, marks[j + 1][0])
        cand = max(gaps[lo:hi], key=lambda g: g[1] - g[0], default=None)
        if cand:
            best.append((cand[1] - cand[0], j, cand))
    ranked = sorted(best, reverse=True)
    if len(ranked) < n_meas:
        sys.exit(f"found {len(marks)} AllReduce markers, not enough for {n_total} ubatches")
    cuts = sorted(ranked[:n_meas], key=lambda x: x[1])
    quality = (ranked[n_meas - 1][0], ranked[n_meas][0] if len(ranked) > n_meas else 0)
    # end of the last ubatch: busy intervals after the last marker until a gap of more than 5 ms
    last_end = None
    for g0, g1 in gaps[bisect.bisect_left(gs, marks[-1][1]):]:
        if g1 - g0 > 5e6:
            last_end = g0
            break
    if last_end is None:
        last_end = max(a[1] for a in acts)
    wins = []
    for u, (_, j, g) in enumerate(cuts):
        end = cuts[u + 1][2][0] if u + 1 < n_meas else last_end
        m_hi = cuts[u + 1][1] + 1 if u + 1 < n_meas else len(marks)
        wins.append((g[0], g[1], end, marks[j + 1:m_hi]))
    return wins, quality


def pp_sweep(items, w0, w1, seg_of, first_comp, pre_key):
    """items: (start, end, prio, step, seg, is_ar) clipped to [w0, w1]. Lower prio wins an overlap; ties go to the
    earlier start. Idle time (nothing active) is charged to the step that starts after it; idle before the first
    activity goes to pre_key, idle before first_comp to ubatch start.
    Returns {(seg, step): [busy, idle]} and {seg: overlapped AllReduce ns}."""
    ev = []
    for k, it in enumerate(items):
        s, e = max(it[0], w0), min(it[1], w1)
        if e > s:
            ev.append((s, 1, k))
            ev.append((e, 0, k))
    ev.sort()
    first_act = ev[0][0] if ev else w1
    res = {}
    ovl = {}
    active = set()
    t = w0
    pend = 0
    for tt, typ, k in ev:
        if tt > t:
            if active:
                win = min(active, key=lambda j: (items[j][2], items[j][0]))
                key = (items[win][4], items[win][3])
                r = res.setdefault(key, [0, 0])
                r[0] += tt - t
                if pend:
                    if t <= first_comp:
                        res.setdefault((0, pre_key if t <= first_act else "start_idle"), [0, 0])[1] += pend
                    else:
                        r[1] += pend
                    pend = 0
                if not items[win][5]:
                    ars = [j for j in active if items[j][5]]
                    if ars:
                        sg = items[ars[0]][4]
                        ovl[sg] = ovl.get(sg, 0) + tt - t
            else:
                pend += tt - t
            t = tt
        if typ:
            active.add(k)
        else:
            active.discard(k)
    if w1 > t:
        pend += w1 - t
    if pend:
        res.setdefault((seg_of(w1), "glue"), [0, 0])[1] += pend
    return res, ovl


def pp_main(pos, opts):
    """PP budget: per ubatch of the measured llama-bench run, per GPU (time unions) and on the critical path.
    Rules:
    - Ubatches: llama-bench runs warm-up (-p N), then per rep the depth prefill (-d) and the measured -p N. The last
      ceil(N / ub) ubatches are measured; boundaries are the host gaps between graphs (pp_ubatch_windows).
    - Segments: split at the AllReduce marker kernels (ggml_cuda_ar_add_kernel, ggml_cuda_ar_kernel) on the compute stream.
    - AllReduce activity: markers, f32 -> bf16 converts on the AllReduce stream (or right before a marker) and the
      memory copies on the AllReduce copy streams. Exposed = no compute kernel on that GPU at the same time;
      overlapped = AllReduce activity under a compute kernel (info row, not summed).
    - Overlaps: compute kernels win over AllReduce work, which wins over other copies. Idle (nothing active) goes to
      the step that starts after it; idle and copies before the first compute kernel of a ubatch are ubatch start.
    - Critical path: per segment, the GPU whose last compute kernel before the AllReduce ended later.
    - Unit: ms per 1K prompt tokens (sum over the measured ubatches x 1000 / N)."""
    path, model, n_prompt = pos[0], pos[1], int(pos[2])
    prof = json.load(open(opts["prof"]))[0] if "prof" in opts else None
    nb = int(opts.get("batch", prof["n_batch"] if prof else 2048))
    nub = int(opts.get("ubatch", prof["n_ubatch"] if prof else 512))
    depth = int(opts.get("depth", prof["n_depth"] if prof else 0))
    reps = len(prof["samples_ns"]) if prof else 1
    n_meas = n_ubatches(n_prompt, nb, nub)
    n_total = int(opts.get("ubatches-total", n_meas * (1 + reps) + (n_ubatches(depth, nb, nub) if depth else 0)))
    per = load_pp(path)
    agents = sorted(a for a in per if any(base(x[2]) in AR_MARK for x in per[a]))
    if not agents:
        sys.exit("no AllReduce kernels: PP mode needs -sm tensor on two GPUs")
    gpu_names = {a: f"ROCm{j}" for j, a in enumerate(agents)}
    to_ms = lambda ns: ns / 1e6 * 1000.0 / n_prompt
    per_gpu = {}
    for a in agents:
        acts = per[a]
        marks = [x for x in acts if not x[5] and base(x[2]) in AR_MARK]
        main_stream = max({x[4] for x in marks}, key=lambda s: sum(1 for x in marks if x[4] == s))
        wins, quality = pp_ubatch_windows(acts, marks, n_total, n_meas)
        # copy streams that carry AllReduce chunks: at least half an AllReduce worth of copies per ubatch after its first AllReduce
        cnt = {}
        for g0, f0, e, mk in wins:
            for x in acts:
                if x[5] and mk[0][0] <= x[0] < e:
                    cnt[x[4]] = cnt.get(x[4], 0) + 1
        n_ar = len(wins[-1][3])
        ar_streams = {s for s, c in cnt.items() if c >= n_ar * n_meas / 2}
        # the measured run starts bench time before the end of its last ubatch; the first ubatch is cut at its first
        # compute kernel and the time before it (host prep, input upload) is the run start row
        lo0 = wins[-1][2] - prof["samples_ns"][-1] if prof else wins[0][1]
        ub = []
        for wi, (g0, f0, e, mk) in enumerate(wins):
            lo = g0
            if wi == 0:
                lo = next(x[0] for x in acts if x[0] >= f0 and not x[5] and x[4] == main_stream and not base(x[2]).startswith("__amd_rocclr"))
            inwin = [x for x in acts if x[1] > lo and x[0] < e]
            mains = [x for x in inwin if not x[5] and x[4] == main_stream]
            mk_ends = [x[1] for x in mains if base(x[2]) in AR_MARK]
            is_ar_k = set()
            for j, x in enumerate(mains):
                if base(x[2]) in AR_MARK or (x[2].startswith("void convert_unary") and "__hip_bfloat16>" in x[2] and j + 1 < len(mains) and base(mains[j + 1][2]) in AR_MARK):
                    is_ar_k.add(id(x))
            comp = [x for x in mains if id(x) not in is_ar_k]
            segs = [[] for _ in range(len(mk_ends) + 1)]
            for x in comp:
                segs[bisect.bisect_left(mk_ends, x[1])].append(x)
            kinds = []
            for s, sg in enumerate(segs):
                names = [x[2] for x in sg]
                if s == len(segs) - 1:
                    kinds.append("out")
                elif any("flash_attn" in x for x in names):
                    kinds.append("attn")
                elif any("gated_delta_net" in x for x in names):
                    kinds.append("recr")
                else:
                    kinds.append("ffn")
            seg_of = lambda t: min(bisect.bisect_left(mk_ends, t), len(mk_ends))
            first_comp = comp[0][0] if comp else e
            items = []
            for s, sg in enumerate(segs):
                for x, st in zip(sg, classify_pp([(x[2], x[3]) for x in sg], kinds[s], model, s == 0)):
                    items.append((x[0], x[1], 0, st, s, False))
            for x in inwin:
                if x[4] == main_stream and not x[5] and id(x) not in is_ar_k:
                    continue
                sg = seg_of(x[1])
                ar_st = "ar_ffn" if kinds[sg] in ("ffn", "out") else "ar_mix"
                if not x[5] and (id(x) in is_ar_k or (x[2].startswith("void convert_unary") and "__hip_bfloat16>" in x[2])):
                    items.append((x[0], x[1], 1, ar_st, sg, True))
                elif x[5] and x[4] in ar_streams:
                    items.append((x[0], x[1], 2, ar_st, sg, True))
                elif x[5]:
                    items.append((x[0], x[1], 3, "start_input" if x[0] < first_comp else "copy", sg, False))
                else:
                    items.append((x[0], x[1], 0, "start_input" if x[0] < first_comp else "glue", sg, False))
            res, ovl = pp_sweep(items, lo, e, seg_of, first_comp, "start_idle")
            if wi == 0:
                res.setdefault((0, "run_start"), [0, 0])[1] += max(0, lo - lo0)
            last_comp = [max((x[1] for x in sg), default=0) for sg in segs]
            last_comp[-1] = e
            ub.append({"res": res, "ovl": ovl, "kinds": kinds, "last": last_comp, "span": e - (lo0 if wi == 0 else lo), "first_gap": f0 - g0,
                       "kernels": sum(1 for x in inwin if not x[5]), "comp": comp})
        per_gpu[a] = {"ub": ub, "quality": quality, "n_ar": n_ar, "ar_streams": sorted(ar_streams), "main": main_stream}

    a0 = agents[0]
    tot = {a: {k: [0, 0] for k in PP_KEYS} for a in agents}
    crit = {k: [0, 0] for k in PP_KEYS}
    for u in range(n_meas):
        nseg = {len(per_gpu[a]["ub"][u]["kinds"]) for a in agents}
        if len(nseg) != 1:
            sys.exit(f"ubatch {u}: segment count differs between GPUs: {nseg}")
        nseg = nseg.pop()
        late = [max(agents, key=lambda a: per_gpu[a]["ub"][u]["last"][s]) for s in range(nseg)]
        for a in agents:
            d = per_gpu[a]["ub"][u]
            for (s, st), (bz, idl) in d["res"].items():
                tot[a][st][0] += bz
                tot[a][st][1] += idl
                if late[s] == a:
                    crit[st][0] += bz
                    crit[st][1] += idl
            for s, v in d["ovl"].items():
                tot[a]["ar_overlap"][0] += v
                if late[s] == a:
                    crit["ar_overlap"][0] += v
    span = {a: sum(d["span"] for d in per_gpu[a]["ub"]) for a in agents}
    sumk = lambda d: sum(v[0] + v[1] for k, v in d.items() if k != "ar_overlap")
    rounding = span[a0] - sumk(crit)
    crit["idle"][1] += rounding
    bench_ns = prof["samples_ns"][-1] if prof else None
    run_start = crit["run_start"][1]

    plain = json.load(open(opts["plain"]))[0] if "plain" in opts else None
    plain_ms = 1e6 / plain["avg_ts"] if plain else None
    all_ms = to_ms(sumk(crit))
    steady_ms = to_ms(span[a0] - run_start)
    bench_ms = to_ms(bench_ns) if bench_ns else None
    scale = plain_ms / all_ms if plain_ms else None
    info = {}
    for a in agents:
        gm = gc = fa = 0
        for d in per_gpu[a]["ub"]:
            for x in d["comp"]:
                b = base(x[2])
                if is_gemm(b, x[2]):
                    gm += x[1] - x[0]
                    gc += 1
                if b.startswith("flash_attn"):
                    fa += x[1] - x[0]
        info[a] = {"gemm_ms": to_ms(gm), "gemm_calls": gc / n_meas, "flash_attn_ms": to_ms(fa),
                   "kernels": sum(d["kernels"] for d in per_gpu[a]["ub"]) / n_meas}

    print(f"trace: {path}")
    print(f"model: {model}, PP {n_prompt} tokens, -b {nb} -ub {nub} depth {depth}: {n_meas} measured ubatches of {n_total} in the trace")
    for a in agents:
        p = per_gpu[a]
        q = p["quality"]
        print(f"{gpu_names[a]} ({a}): compute stream {p['main']}, AllReduce copy streams {p['ar_streams']}, AllReduce per ubatch {p['n_ar']}, "
              f"ubatch cut: smallest boundary gap {q[0] / 1e6:.2f} ms, largest other gap {q[1] / 1e6:.2f} ms, span {span[a] / 1e6:.2f} ms, "
              f"kernels/ubatch {info[a]['kernels']:.0f}, GEMM {info[a]['gemm_ms']:.2f} ms/1K ({info[a]['gemm_calls']:.0f} calls/ubatch), flash_attn* {info[a]['flash_attn_ms']:.2f} ms/1K")
        if q[1] > 0.5 * q[0]:
            print(f"WARNING: weak ubatch cut on {gpu_names[a]}; check --ubatches-total")
    print(f"run start (bench timer start to the first compute kernel, critical): {run_start / 1e6:.2f} ms")
    if bench_ms:
        print(f"llama-bench profiled: {bench_ms:.2f} ms/1K ({1e6 / bench_ms:.1f} t/s)")
    if plain:
        print(f"llama-bench unprofiled: {plain['avg_ts']:.1f} t/s = {plain_ms:.2f} ms/1K; scaled = critical x {scale:.4f}")
    print()
    print("ms per 1K prompt tokens, busy + idle-before")
    print(f"{'step':<82} " + " ".join(f"{gpu_names[a]:>17}" for a in agents) + f" {'critical':>17} {'sum':>8} {'%':>5}")
    pair = lambda v: f"{to_ms(v[0]):8.2f} + {to_ms(v[1]):6.2f}"
    for k, label in PP_STEPS + PP_INFO:
        c = crit[k]
        if max(to_ms(sum(tot[a][k])) for a in agents) < 0.005 and abs(to_ms(sum(c))) < 0.005:
            continue
        print(f"{label:<82} " + " ".join(f"{pair(tot[a][k]):>17}" for a in agents) + f" {pair(c):>17} {to_ms(sum(c)):8.2f} {100 * to_ms(sum(c)) / all_ms:5.1f}")
    print(f"{'total':<82} " + " ".join(f"{to_ms(sumk(tot[a])):17.2f}" for a in agents) + f" {to_ms(sumk(crit)):17.2f}")
    print(f"PP t/s profiled {1e6 / all_ms:.1f}" + (f", unprofiled {plain['avg_ts']:.1f}" if plain else ""))

    measured = opts.get("measured") or time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(os.path.getmtime(path)))
    shas = bin_shas(opts)
    out = {
        "phase": "pp", "trace": path, "model": model, "n_tokens": n_prompt, "batch": nb, "ubatch": nub, "depth": depth,
        "ubatches_measured": n_meas, "ubatches_total": n_total, "ar_per_ubatch": per_gpu[a0]["n_ar"],
        "token_ms": all_ms, "steady_ms": steady_ms, "bench_prof_ms": bench_ms, "bench_plain_ms": plain_ms, "scale": scale,
        "plain_ts": plain["avg_ts"] if plain else None, "plain_sd": plain["stddev_ts"] if plain else None, "plain_note": opts.get("plain-note", ""),
        "provenance": {"Column id": opts.get("id", "-"), "Measured": measured, "Source commit": opts.get("commit", "-"),
                       "Build (sha256 llama-bench / libggml-hip.so)": f"`{opts.get('build', '-')}` (`{' / '.join(shas)}`)",
                       "Model, quant, file revision": opts.get("modeldesc", "-"), "Settings": opts.get("settings", "-"),
                       "Kernels / AllReduces per ubatch": f"{info[a0]['kernels']:.0f} / {per_gpu[a0]['n_ar']}"},
        "gpus": {gpu_names[a]: dict(info[a], token_ms=to_ms(span[a]), steps_busy_idle={k: [to_ms(v[0]), to_ms(v[1])] for k, v in tot[a].items()}) for a in agents},
        "critical_busy_idle": {k: [to_ms(v[0]), to_ms(v[1])] for k, v in crit.items()},
    }
    if "json" in opts:
        with open(opts["json"], "w") as f:
            json.dump(out, f, indent=1)
    if "append-csv" in opts:
        append_csv(opts["append-csv"], *csv_rows(out, opts))


def bin_shas(opts):
    shas = []
    if "bin" in opts:
        for p in (opts["bin"], os.path.join(os.path.dirname(opts["bin"]), "libggml-hip.so")):
            if os.path.exists(p):
                with open(p, "rb") as f:
                    shas.append(hashlib.sha256(f.read()).hexdigest()[:12])
    return shas


# ---------------------------------------------------------------- CSV record and markdown render

DEF_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
COL_CSV = "token-budget-columns.csv"
STEP_CSV = "token-budget-steps.csv"
COL_FIELDS = ["column_id", "phase", "model", "model_desc", "quant", "depth", "ubatch", "batch", "n_tokens", "n_gpus", "measured",
              "source_commit", "build", "sha_llama_bench", "sha_libggml_hip", "settings", "plain_ts", "plain_sd", "plain_note",
              "prof_ms_all", "prof_ms_steady", "bench_prof_ms", "kernels_per_unit", "ar_per_unit", "unit", "scale_to_unprofiled",
              "notes", "record_link"]
STEP_FIELDS = ["column_id", "step_key", "step_label", "gpu", "busy_ms", "idle_ms", "scaled_ms"]
INFO_TG = [("info_matvec_ms", "matvec_ms", "Mat-vec kernel time (info)"), ("info_matvec_calls", "matvec_calls", "Mat-vec calls per token (info)"),
           ("info_flash_attn_ms", "flash_attn_ms", "flash_attn* kernel time (info)"), ("info_topk_moe_ms", "topk_moe_ms", "topk_moe kernel time (info)")]
INFO_PP = [("info_gemm_ms", "gemm_ms", "GEMM kernel time (info)"), ("info_gemm_calls", "gemm_calls", "GEMM calls per ubatch (info)"),
           ("info_flash_attn_ms", "flash_attn_ms", "flash_attn* kernel time (info)")]


def num(v):
    return "" if v is None else repr(float(v))


def csv_rows(d, opts):
    """Column row and step rows from a --json output (old TG JSONs have the provenance cells only)."""
    prov = d["provenance"]
    phase = d.get("phase", "tg")
    cid = opts.get("id") or prov["Column id"]
    if cid == "-":
        sys.exit("needs --id")
    if f" {phase.upper()} " not in cid:
        label, date = cid.rsplit(" ", 1)
        cid = f"{label} {phase.upper()} {date}"
    bm = re.match(r"`(.*?)` \(`(\w*)(?: / (\w*))?`\)", prov["Build (sha256 llama-bench / libggml-hip.so)"])
    st = prov["Settings"]
    ts, sd, note = d.get("plain_ts"), d.get("plain_sd"), d.get("plain_note", "")
    if ts is None and "Unprofiled TG t/s (ms/token)" in prov:
        pm = re.match(r"([\d.]+) \+- ([\d.]+) \(([\d.]+)\)\s*(.*)", prov["Unprofiled TG t/s (ms/token)"])
        ts = 1000.0 / d["bench_plain_ms"] if d.get("bench_plain_ms") else float(pm.group(1))
        sd, note = float(pm.group(2)), pm.group(4)
    get = lambda pat, default="": (re.search(pat, st).group(1) if re.search(pat, st) else default)
    desc = prov["Model, quant, file revision"]
    qm = re.search(r"\b((?:UD-)?I?Q\d\w*|F16|BF16)\b", desc)
    kern = prov.get("Kernels / AllReduces per token") or prov.get("Kernels / AllReduces per ubatch")
    col = {
        "column_id": cid, "phase": phase, "model": d["model"], "model_desc": desc, "quant": qm.group(1) if qm else "",
        "depth": d.get("depth", get(r"depth (\d+)", "0")), "ubatch": d.get("ubatch", get(r"-ub (\d+)")), "batch": d.get("batch", get(r"-b (\d+)")),
        "n_tokens": d["n_tokens"], "n_gpus": len(d["gpus"]), "measured": prov["Measured"], "source_commit": prov["Source commit"],
        "build": bm.group(1), "sha_llama_bench": bm.group(2), "sha_libggml_hip": bm.group(3) or "", "settings": st,
        "plain_ts": num(ts), "plain_sd": num(sd), "plain_note": note, "prof_ms_all": num(d["token_ms"]), "prof_ms_steady": num(d["steady_ms"]),
        "bench_prof_ms": num(d.get("bench_prof_ms")), "kernels_per_unit": kern.split(" / ")[0], "ar_per_unit": kern.split(" / ")[1],
        "unit": "ms/token" if phase == "tg" else "ms/1K prompt tokens", "scale_to_unprofiled": num(d.get("scale")),
        "notes": opts.get("notes", ""), "record_link": opts.get("record", ""),
    }
    steps_def = STEPS if phase == "tg" else PP_STEPS + PP_INFO
    info_def = INFO_TG if phase == "tg" else INFO_PP
    scale = d.get("scale") or 0
    rows = []
    for gpu, gd in list(d["gpus"].items()) + [("critical", None)]:
        src = gd["steps_busy_idle"] if gd else d["critical_busy_idle"]
        for k, label in steps_def:
            b, i = src.get(k, [0, 0])
            sc = num((b + i) * scale) if gd is None and scale and not k.startswith("ar_overlap") else ""
            rows.append({"column_id": cid, "step_key": k, "step_label": label, "gpu": gpu, "busy_ms": num(b), "idle_ms": num(i), "scaled_ms": sc})
        if gd:
            for k, src_k, label in info_def:
                rows.append({"column_id": cid, "step_key": k, "step_label": label, "gpu": gpu, "busy_ms": num(gd.get(src_k, 0)), "idle_ms": "", "scaled_ms": ""})
    return col, rows


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def append_csv(dirpath, col, rows):
    """Append-only: a column id is written once; old rows are never changed."""
    cp, sp = os.path.join(dirpath, COL_CSV), os.path.join(dirpath, STEP_CSV)
    if any(r["column_id"] == col["column_id"] for r in read_csv(cp)):
        sys.exit(f"column_id already in {cp}: {col['column_id']}")
    for p, fields, data in ((cp, COL_FIELDS, [col]), (sp, STEP_FIELDS, rows)):
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            if new:
                w.writeheader()
            w.writerows(data)
    print(f"appended {col['column_id']} to {cp} and {STEP_CSV} ({len(rows)} step rows)", file=sys.stderr)


def fl(v):
    return float(v) if v not in ("", None) else None


def short_id(c):
    return c["column_id"].split(f" {c['phase'].upper()} ")[0]


def head_id(c):
    label, date = c["column_id"].split(f" {c['phase'].upper()} ")
    return f"{label} {date}"


def build_cell(c):
    shas = " / ".join(x for x in (c["sha_llama_bench"], c["sha_libggml_hip"]) if x)
    return f"`{c['build']}` (`{shas}`)"


def table(rows):
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def render_tg_main(cols, st):
    hdr = ["Row"] + [head_id(c) for c in cols]
    out = [hdr, ["---"] * len(hdr)]
    cell_rows = [
        ("Measured", lambda c: c["measured"]),
        ("Source commit", lambda c: c["source_commit"]),
        ("Build (sha256 llama-bench / libggml-hip.so)", build_cell),
        ("Model, quant, file revision", lambda c: c["model_desc"]),
        ("Settings", lambda c: c["settings"]),
        ("Unprofiled TG t/s (ms/token)", lambda c: f"{fl(c['plain_ts']):.2f} +- {fl(c['plain_sd']):.2f} ({1000 / fl(c['plain_ts']):.2f})" + (" " + c["plain_note"] if c["plain_note"] else "")),
        ("Profiled ms/token: all tokens; steady; llama-bench", lambda c: f"{fl(c['prof_ms_all']):.2f}; {fl(c['prof_ms_steady']):.2f}; " + (f"{fl(c['bench_prof_ms']):.2f}" if fl(c["bench_prof_ms"]) else "-")),
        ("Kernels / AllReduces per token", lambda c: f"{fl(c['kernels_per_unit']):.0f} / {c['ar_per_unit']}"),
    ]
    for label, f in cell_rows:
        out.append([label] + [f(c) for c in cols])
    for k, label in STEPS:
        r = [label]
        for c in cols:
            b, i = st[(c["column_id"], "critical", k)]
            r.append("-" if b + i < 0.005 else f"{b:.2f} + {i:.2f} ({100 * (b + i) / fl(c['prof_ms_all']):.0f}%)")
        out.append(r)
    out.append(["Total, profiled"] + [f"{sum(st[(c['column_id'], 'critical', k)][0] + st[(c['column_id'], 'critical', k)][1] for k in KEYS):.2f}" for c in cols])
    out.append(["Total, scaled to unprofiled"] + [f"{1000 / fl(c['plain_ts']):.2f}" for c in cols])
    return table(out)


def gpus_of(c, st):
    return [g for g in ("ROCm0", "ROCm1", "ROCm2", "ROCm3") if (c["column_id"], g, "norm") in st]


def render_tg_gpu(cols, st, info):
    out = [["Datapoint", "GPU", "Mat-vec (calls)", "flash_attn*", "AR exchange", "AR wait", "Token-start gap", "Other idle", "Busy"],
           ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]]
    for c in cols:
        cid = c["column_id"]
        for g in gpus_of(c, st):
            v = lambda k: st[(cid, g, k)]
            busy = sum(v(k)[0] for k in KEYS)
            idle = sum(v(k)[1] for k in KEYS) - v("start_idle")[1] - v("warmup")[1]
            two = int(c["n_gpus"]) > 1
            out.append([short_id(c), g, f"{info[(cid, g, 'info_matvec_ms')]:.2f} ({info[(cid, g, 'info_matvec_calls')]:.0f})", f"{info[(cid, g, 'info_flash_attn_ms')]:.2f}",
                        f"{v('ar_mix')[0] + v('ar_ffn')[0]:.2f}" if two else "-", f"{v('ar_mix_wait')[0] + v('ar_ffn_wait')[0]:.2f}" if two else "-",
                        f"{v('start_idle')[1]:.2f}", f"{idle:.2f}", f"{busy:.2f}"])
    return table(out)


def render_pp_main(cols, st):
    hdr = ["Row"] + [head_id(c) for c in cols]
    out = [hdr, ["---"] * len(hdr)]
    tot = lambda c: sum(sum(st[(c["column_id"], "critical", k)]) for k, _ in PP_STEPS)
    cell_rows = [
        ("Measured", lambda c: c["measured"]),
        ("Source commit", lambda c: c["source_commit"]),
        ("Build (sha256 llama-bench / libggml-hip.so)", build_cell),
        ("Model, quant, file revision", lambda c: c["model_desc"]),
        ("Settings", lambda c: c["settings"]),
        ("Unprofiled PP t/s (ms per 1K tokens)", lambda c: f"{fl(c['plain_ts']):.1f} +- {fl(c['plain_sd']):.1f} ({1e6 / fl(c['plain_ts']):.1f})" + (" " + c["plain_note"] if c["plain_note"] else "")),
        ("Profiled ms per 1K: all; traced ubatches; llama-bench", lambda c: f"{fl(c['prof_ms_all']):.1f}; {fl(c['prof_ms_steady']):.1f}; " + (f"{fl(c['bench_prof_ms']):.1f}" if fl(c["bench_prof_ms"]) else "-")),
        ("Measured ubatches; kernels / AllReduces per ubatch", lambda c: f"{n_ubatches(int(c['n_tokens']), int(c['batch']), int(c['ubatch']))}; {fl(c['kernels_per_unit']):.0f} / {c['ar_per_unit']}"),
    ]
    for label, f in cell_rows:
        out.append([label] + [f(c) for c in cols])
    for k, label in PP_STEPS:
        r = [label]
        for c in cols:
            b, i = st[(c["column_id"], "critical", k)]
            r.append("-" if abs(b + i) < 0.05 else f"{b:.1f} + {i:.1f} ({100 * (b + i) / tot(c):.1f}%)")
        out.append(r)
    for k, label in PP_INFO:
        out.append([label] + [f"{st[(c['column_id'], 'critical', k)][0]:.1f}" for c in cols])
    out.append(["Total, profiled: ms per 1K; ms/token; t/s"] + [f"{tot(c):.1f}; {tot(c) / 1000:.3f}; {1e6 / tot(c):.1f}" for c in cols])
    out.append(["Total, scaled to unprofiled: ms per 1K; ms/token; t/s"] + [f"{1e6 / fl(c['plain_ts']):.1f}; {1e3 / fl(c['plain_ts']):.3f}; {fl(c['plain_ts']):.1f}" for c in cols])
    return table(out)


def render_pp_gpu(cols, st, info):
    out = [["Datapoint", "GPU", "GEMM ms (calls/ubatch)", "flash_attn*", "AR exposed", "AR overlapped", "Start gaps (ubatch + run)", "Other idle", "Busy"],
           ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]]
    for c in cols:
        cid = c["column_id"]
        for g in gpus_of(c, st):
            v = lambda k: st[(cid, g, k)]
            keys = [k for k, _ in PP_STEPS]
            busy = sum(v(k)[0] for k in keys)
            gaps = v("start_idle")[1] + v("run_start")[1]
            idle = sum(v(k)[1] for k in keys) - gaps
            out.append([short_id(c), g, f"{info[(cid, g, 'info_gemm_ms')]:.1f} ({info[(cid, g, 'info_gemm_calls')]:.0f})", f"{info[(cid, g, 'info_flash_attn_ms')]:.1f}",
                        f"{v('ar_mix')[0] + v('ar_ffn')[0]:.1f}", f"{v('ar_overlap')[0]:.1f}", f"{gaps:.1f}", f"{idle:.1f}", f"{busy:.1f}"])
    return table(out)


def render(opts):
    d = opts.get("dir", DEF_DIR)
    doc = opts.get("doc", os.path.join(d, "TOKEN-BUDGET.md"))
    cols = read_csv(os.path.join(d, COL_CSV))
    st, info = {}, {}
    for r in read_csv(os.path.join(d, STEP_CSV)):
        key = (r["column_id"], r["gpu"], r["step_key"])
        if r["step_key"].startswith("info_"):
            info[key] = float(r["busy_ms"])
        else:
            st[key] = (float(r["busy_ms"]), float(r["idle_ms"]))
    tg = [c for c in cols if c["phase"] == "tg"]
    pp = [c for c in cols if c["phase"] == "pp"]
    blocks = {"tg-main": render_tg_main(tg, st), "tg-gpu": render_tg_gpu(tg, st, info),
              "pp-main": render_pp_main(pp, st) if pp else "", "pp-gpu": render_pp_gpu(pp, st, info) if pp else ""}
    with open(doc) as f:
        text = f.read()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- tb:{name} -->\n).*?(<!-- /tb:{name} -->)", re.S)
        if not pat.search(text):
            continue
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    with open(doc, "w") as f:
        f.write(text)
    print(f"rendered {', '.join(k for k in blocks if f'<!-- tb:{k} -->' in text)} into {doc}", file=sys.stderr)


if __name__ == "__main__":
    main()

