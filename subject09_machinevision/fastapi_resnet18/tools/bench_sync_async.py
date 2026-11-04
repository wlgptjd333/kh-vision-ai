"""동기·비동기 처리 결과와 속도 비교 (교재 02의 13절 비교 규칙을 따른 측정 스크립트, 교재 외 추가 파일).

실행 전: docker compose up -d, Celery Worker, uvicorn 서버가 모두 켜져 있어야 한다.
예: python tools/bench_sync_async.py --images dataset/test --count 10
"""
import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from redis import Redis

from ai.config import MODEL_PATH, REDIS_CACHE_URL

POLL_SEC = 0.05  # 브라우저 화면은 1초 간격이지만 측정은 더 촘촘히 확인한다


def pick_images(root: Path, count: int) -> list[Path]:
    """클래스 폴더마다 돌아가며 골라 클래스가 섞이게 한다."""
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    pools = [sorted(f.glob("*.jpg")) for f in folders]
    picked = []
    i = 0
    while len(picked) < count:
        pool = pools[i % len(pools)]
        idx = i // len(pools)
        if idx < len(pool):
            picked.append(pool[idx])
        i += 1
    return picked


def post_image(client, url, path):
    with path.open("rb") as f:
        return client.post(url, files={"image": (path.name, f.read(), "image/jpeg")})


def run_sync(client, path):
    t0 = time.perf_counter()
    r = post_image(client, "/api/sync/classify/", path)
    total = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    return {"accept_ms": total, "done_ms": total, "data": data}


def run_async(client, path):
    t0 = time.perf_counter()
    r = post_image(client, "/api/async/classify/", path)
    accept = (time.perf_counter() - t0) * 1000
    if r.status_code != 202:
        raise RuntimeError(f"async enqueue HTTP {r.status_code}: {r.text}")
    check_url = r.json()["check_url"]
    while True:
        job = client.get(check_url).json()
        if job["status"] in ("done", "failed"):
            break
        if time.perf_counter() - t0 > 120:
            raise TimeoutError(check_url)
        time.sleep(POLL_SEC)
    done = (time.perf_counter() - t0) * 1000
    if job["status"] == "failed":
        raise RuntimeError(job["error_message"])
    return {"accept_ms": accept, "done_ms": done, "data": job}


def summary(rows, key):
    values = [r[key] for r in rows]
    return round(statistics.median(values), 1), round(statistics.mean(values), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--images", default="dataset/test")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = pick_images(Path(args.images), args.count + 1)
    warmup, images = images[-1], images[:-1]
    cache = Redis.from_url(REDIS_CACHE_URL)
    rows = []

    with httpx.Client(base_url=args.base, timeout=120) as client:
        # 워밍업: 서버와 워커가 각자 모델을 한 번씩 메모리에 올리게 한다 (측정에서 제외)
        cache.flushdb()
        run_sync(client, warmup)
        cache.flushdb()
        run_async(client, warmup)

        # 1) 순차 요청: 같은 이미지 묶음을 MISS → HIT 순서로
        for mode, fn in (("sync", run_sync), ("async", run_async)):
            cache.flushdb()  # Redis 캐시 DB(/1)만 비운다. Broker(/2)·결과(/3)는 그대로
            for phase in ("MISS", "HIT"):
                for path in images:
                    res = fn(client, path)
                    d = res["data"]
                    rows.append({
                        "test": "sequential", "mode": mode, "cache": phase,
                        "image": f"{path.parent.name}/{path.name}",
                        "accept_ms": round(res["accept_ms"], 1), "done_ms": round(res["done_ms"], 1),
                        "inference_ms": d.get("inference_ms"), "cache_hit": d.get("cache_hit"),
                        "top1_label": d["top1"]["label"], "top1_score": round(d["top1"]["score"], 4),
                    })

    # 2) 동시 요청: 같은 이미지 묶음을 한꺼번에 보낸다 (캐시 MISS 상태)
    burst = []
    for mode, fn in (("sync", run_sync), ("async", run_async)):
        cache.flushdb()
        with httpx.Client(base_url=args.base, timeout=300) as client:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(images)) as pool:
                results = list(pool.map(lambda p: fn(client, p), images))
            wall = (time.perf_counter() - t0) * 1000
        burst.append({
            "mode": mode, "requests": len(images),
            "accept_max_ms": round(max(r["accept_ms"] for r in results), 1),
            "all_done_ms": round(wall, 1),
        })

    with (out / "sync_async_requests.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    table = []
    for mode in ("sync", "async"):
        for phase in ("MISS", "HIT"):
            part = [r for r in rows if r["mode"] == mode and r["cache"] == phase]
            inf = [r["inference_ms"] for r in part if r["inference_ms"] is not None]
            table.append({
                "mode": mode, "cache": phase, "n": len(part),
                "accept_median_ms": summary(part, "accept_ms")[0],
                "done_median_ms": summary(part, "done_ms")[0],
                "done_mean_ms": summary(part, "done_ms")[1],
                "inference_median_ms": round(statistics.median(inf), 1) if inf else None,
                "cache_hit_count": sum(1 for r in part if r["cache_hit"]),
            })

    same = {}
    for r in rows:
        if r["cache"] == "MISS":
            same.setdefault(r["image"], {})[r["mode"]] = (r["top1_label"], r["top1_score"])
    agree = sum(1 for v in same.values() if v["sync"] == v["async"])

    result = {
        "model_path": MODEL_PATH, "images": len(images), "poll_sec": POLL_SEC,
        "sequential": table, "burst": burst,
        "same_top1_sync_vs_async": f"{agree}/{len(same)}",
    }
    (out / "sync_async_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
