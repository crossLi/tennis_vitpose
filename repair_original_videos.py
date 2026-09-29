import sys
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import ANALYSIS_TASKS_DIR, transcode_original_to_playable_mp4, get_task  # noqa: E402

all_task_ids = sorted(p.name for p in ANALYSIS_TASKS_DIR.iterdir() if p.is_dir())

need_transcode: list[str] = []
for task_id in all_task_ids:
    try:
        task = get_task(task_id)
    except Exception:
        continue
    ov = task.get("artifacts", {}).get("original_video") or ""
    ovmp4 = task.get("artifacts", {}).get("original_video_mp4") or ""
    if ov and not ovmp4 and Path(ov).exists() and Path(ov).suffix.lower() != ".mp4":
        need_transcode.append(task_id)

print(f"Total tasks: {len(all_task_ids)}  Need transcode: {len(need_transcode)}")
print(need_transcode[:10])

if len(sys.argv) > 1 and sys.argv[1] == "dry":
    sys.exit(0)

errors: list[tuple[str, str]] = []

def worker(tid: str):
    try:
        result = transcode_original_to_playable_mp4(tid)
        print(f"[OK] {tid} -> {result}")
    except Exception as exc:  # pragma: no cover
        errors.append((tid, str(exc)))
        print(f"[ERR] {tid}: {exc}")

threads = []
for tid in need_transcode:
    t = threading.Thread(target=worker, args=(tid,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print()
print(f"Done. Errors: {len(errors)}")
for tid, err in errors:
    print("  ", tid, err)
