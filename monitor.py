#!/usr/bin/env python3
import argparse, json, os, signal, sys, time, psutil, shlex, subprocess
from datetime import datetime

def iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def proc_tree(p: psutil.Process):
    """Return [p] + all children (recursive), skipping zombies/defunct."""
    out = []
    try:
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            out.append(p)
            for c in p.children(recursive=True):
                try:
                    if c.is_running() and c.status() != psutil.STATUS_ZOMBIE:
                        out.append(c)
                except Exception:
                    pass
    except Exception:
        pass
    return out

def safe_open_files(p):
    try:
        return len(p.open_files())
    except Exception:
        return -1

def sample_tree(p: psutil.Process):
    """Aggregate metrics across the process tree."""
    procs = proc_tree(p)
    cpu_pct = 0.0
    cpu_user = 0.0
    cpu_sys  = 0.0
    rss = 0
    vms = 0
    thr = 0
    fds = 0
    rb = 0
    wb = 0
    for pr in procs:
        try:
            cpu_pct += pr.cpu_percent(None)  # primed in main loop
            t = pr.cpu_times()
            cpu_user += getattr(t, "user", 0.0)
            cpu_sys  += getattr(t, "system", 0.0)
            mem = pr.memory_info()
            rss += mem.rss
            vms += mem.vms
            thr += pr.num_threads()
            of = safe_open_files(pr)
            fds += of if of > 0 else 0
            try:
                io = pr.io_counters()
                rb += getattr(io, "read_bytes", 0)
                wb += getattr(io, "write_bytes", 0)
            except Exception:
                pass
        except Exception:
            pass
    return {
        "cpu_percent": cpu_pct,
        "cpu_time_user_s": cpu_user,
        "cpu_time_sys_s": cpu_sys,
        "rss_bytes": rss,
        "vms_bytes": vms,
        "num_threads": thr,
        "open_files": fds if fds > 0 else -1,
        "read_bytes": rb,
        "write_bytes": wb,
        "proc_count": len(procs),
    }

def print_header():
    print("MON_SAMPLE_HEADER,"
          "ts,wall_s,cpu_percent,cpu_time_user_s,cpu_time_sys_s,"
          "rss_bytes,vms_bytes,num_threads,open_files,read_bytes,write_bytes,"
          "proc_count,sys_cpu_percent,sys_mem_percent", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Monitor a process tree over time.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pid", type=int, help="Attach to existing PID")
    g.add_argument("--cmd", type=str, help='Launch and monitor this shell command (e.g. "streamlit run app.py")')
    ap.add_argument("--interval", type=float, default=1.0, help="Sample interval seconds (default 1.0)")
    ap.add_argument("--outfile", type=str, default="", help="Also write samples to this CSV path")
    args = ap.parse_args()

    # Launch or attach
    popen = None
    if args.cmd:
        popen = subprocess.Popen(shlex.split(args.cmd))
        pid = popen.pid
    else:
        pid = args.pid

    try:
        target = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"ERROR: PID {pid} not found", file=sys.stderr)
        sys.exit(1)

    # Prime cpu_percent for tree
    for pr in proc_tree(target):
        try: pr.cpu_percent(None)
        except Exception: pass

    t0 = time.perf_counter()
    peak_rss = 0
    rb0 = wb0 = None
    samples = 0

    # Optional file
    fout = open(args.outfile, "w") if args.outfile else None
    if fout:
        fout.write("ts,wall_s,cpu_percent,cpu_time_user_s,cpu_time_sys_s,"
                   "rss_bytes,vms_bytes,num_threads,open_files,read_bytes,write_bytes,"
                   "proc_count,sys_cpu_percent,sys_mem_percent\n")
        fout.flush()

    print_header()

    # Ctrl+C handling
    stop = False
    def _sigint(_sig,_frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    # System-wide priming
    psutil.cpu_percent(None)

    try:
        while not stop:
            # Stop if launched command exits
            if popen and popen.poll() is not None:
                stop = True

            try:
                if not target.is_running():
                    stop = True
            except Exception:
                stop = True

            wall = time.perf_counter() - t0
            sys_cpu = psutil.cpu_percent(None)
            sys_mem = psutil.virtual_memory().percent

            m = sample_tree(target)
            peak_rss = max(peak_rss, m["rss_bytes"])
            if rb0 is None:
                rb0, wb0 = m["read_bytes"], m["write_bytes"]
            rel_rb = max(0, m["read_bytes"]  - rb0)
            rel_wb = max(0, m["write_bytes"] - wb0)

            line = (f"MON_SAMPLE,{iso()},{wall:.3f},{m['cpu_percent']:.1f},"
                    f"{m['cpu_time_user_s']:.3f},{m['cpu_time_sys_s']:.3f},"
                    f"{m['rss_bytes']},{m['vms_bytes']},{m['num_threads']},"
                    f"{m['open_files']},{rel_rb},{rel_wb},{m['proc_count']},"
                    f"{sys_cpu:.1f},{sys_mem:.1f}")
            print(line, flush=True)
            if fout:
                fout.write(line.split("MON_SAMPLE,",1)[1] + "\n")
                fout.flush()

            samples += 1
            if stop: break
            time.sleep(args.interval)
    finally:
        if fout:
            fout.close()

        # Summary (JSON, single line)
        summary = {
            "ts": iso(),
            "wall_s": round(time.perf_counter() - t0, 3),
            "cpu_time_user_s": round(m.get("cpu_time_user_s", 0.0), 3),
            "cpu_time_sys_s": round(m.get("cpu_time_sys_s", 0.0), 3),
            "peak_rss_bytes": int(peak_rss),
            "total_read_bytes": int(max(0, m.get("read_bytes", 0) - (rb0 or 0))),
            "total_write_bytes": int(max(0, m.get("write_bytes", 0) - (wb0 or 0))),
            "samples": samples,
        }
        print("MON_SUMMARY_JSON," + json.dumps(summary, separators=(",", ":")), flush=True)

        # If we launched the process, try to reap
        if popen and popen.poll() is None:
            try: popen.terminate()
            except Exception: pass
