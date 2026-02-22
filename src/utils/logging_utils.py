import time

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def progress_bar(current, total, width=28):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100
    return f"[{bar}] {pct:5.1f}%"


def log_progress(batch_num, total_batches, done, total, batch_time, elapsed):
    avg = elapsed / done if done else 0
    remaining = total - done
    eta = remaining * avg

    print(
        f"Batch {batch_num}/{total_batches} "
        f"{progress_bar(done, total)} "
        f"{done}/{total} "
        f"batch:{batch_time:.2f}s "
        f"avg:{avg*1000:.0f}ms "
        f"ETA:{fmt_time(eta)}"
    )