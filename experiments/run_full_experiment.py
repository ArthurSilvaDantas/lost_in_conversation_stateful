import sys
import os
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)  # data/, prompts/, results/ are all referenced via relative paths

from utils import load_env_vars
load_env_vars(".env")

from simulator_snowball import ConversationSimulatorSnowball
from simulator_snowball_stateful import ConversationSimulatorSnowballStateful

TASK = "math"
MODEL = "gpt-4o-mini"
N_WORKERS = 5
N_RETRIES = 1
LOG_FOLDER = "results/logs"
DATASET_FN = "data/sharded_instructions_600.json"


def already_done_task_ids(log_path):
    try:
        with open(log_path, "r") as f:
            return {json.loads(l)["task_id"] for l in f}
    except FileNotFoundError:
        return set()


with open(DATASET_FN, "r") as f:
    data = json.load(f)

all_samples = [d for d in data if d["task"] == TASK]

done_original = already_done_task_ids(f"{LOG_FOLDER}/{TASK}/snowball/snowball_{TASK}_{MODEL}.jsonl")
done_stateful = already_done_task_ids(f"{LOG_FOLDER}/{TASK}/snowball-stateful/snowball-stateful_{TASK}_{MODEL}.jsonl")

samples_for_original = [s for s in all_samples if s["task_id"] not in done_original]
samples_for_stateful = [s for s in all_samples if s["task_id"] not in done_stateful]

print(f"Task '{TASK}': {len(all_samples)} total samples")
print(f"Original: {len(done_original)} already done, {len(samples_for_original)} remaining")
print(f"Stateful: {len(done_stateful)} already done, {len(samples_for_stateful)} remaining")


def run_original(sample):
    last_exc = None
    for attempt in range(N_RETRIES):
        try:
            sim = ConversationSimulatorSnowball(
                TASK, sample, assistant_model=MODEL, system_model=MODEL, user_model=MODEL,
                dataset_fn=DATASET_FN, log_folder=LOG_FOLDER,
            )
            is_correct, score = sim.run(verbose=False, save_log=True)
            total_cost = sum(msg.get("cost_usd", 0) or 0 for msg in sim.trace)
            n_turns = sim.get_num_turns("assistant")
            return {
                "task_id": sample["task_id"], "conv_type": "snowball",
                "is_correct": is_correct, "score": score, "n_turns": n_turns, "cost_usd": total_cost,
            }
        except Exception as e:
            last_exc = e
    raise last_exc


def run_stateful(sample):
    last_exc = None
    for attempt in range(N_RETRIES):
        try:
            sim = ConversationSimulatorSnowballStateful(
                TASK, sample, assistant_model=MODEL, system_model=MODEL, user_model=MODEL,
                dataset_fn=DATASET_FN, log_folder=LOG_FOLDER,
            )
            is_correct, score = sim.run(verbose=False, save_log=True)
            total_cost = sum(msg.get("cost_usd", 0) or 0 for msg in sim.trace)
            n_turns = sim.get_num_turns("assistant")
            duplicate_count = sum(
                1 for msg in sim.trace
                if msg["role"] == "log" and msg["content"]["type"] == "duplicate_shard_ignored"
            )
            max_turns_hit = any(
                msg["role"] == "log" and msg["content"]["type"] == "max_turns_exceeded"
                for msg in sim.trace
            )
            return {
                "task_id": sample["task_id"], "conv_type": "snowball-stateful",
                "is_correct": is_correct, "score": score, "n_turns": n_turns, "cost_usd": total_cost,
                "duplicate_shards_ignored": duplicate_count, "max_turns_exceeded": max_turns_hit,
            }
        except Exception as e:
            last_exc = e
    raise last_exc


jobs = []
for sample in samples_for_original:
    jobs.append((run_original, sample))
for sample in samples_for_stateful:
    jobs.append((run_stateful, sample))

results = []
errors = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(fn, sample): (fn, sample) for fn, sample in jobs}
    pbar = tqdm.tqdm(as_completed(futures), total=len(futures))
    for future in pbar:
        fn, sample = futures[future]
        try:
            r = future.result()
            results.append(r)
            running_cost = sum(x["cost_usd"] for x in results)
            pbar.set_postfix({"cost": f"${running_cost:.3f}"})
        except Exception as e:
            errors.append({"fn": fn.__name__, "task_id": sample["task_id"], "error": str(e)})
            print(f"[ERROR] {fn.__name__} on {sample['task_id']}: {e}", flush=True)

with open("results/full_experiment_results.json", "w") as f:
    json.dump({"results": results, "errors": errors}, f, indent=2)


def summarize(conv_type):
    rows = [r for r in results if r["conv_type"] == conv_type]
    n = len(rows)
    if n == 0:
        return
    acc = sum(1 for r in rows if r["is_correct"]) / n
    avg_score = sum(r["score"] or 0 for r in rows) / n
    avg_turns = sum(r["n_turns"] for r in rows) / n
    total_cost = sum(r["cost_usd"] for r in rows)
    print(f"\n=== {conv_type} (n={n}) ===", flush=True)
    print(f"Accuracy (is_correct): {acc:.2%}", flush=True)
    print(f"Avg score: {avg_score:.3f}", flush=True)
    print(f"Avg assistant turns: {avg_turns:.2f}", flush=True)
    print(f"Total cost: ${total_cost:.4f}", flush=True)
    if conv_type == "snowball-stateful":
        total_dupes = sum(r.get("duplicate_shards_ignored", 0) for r in rows)
        n_max_turns = sum(1 for r in rows if r.get("max_turns_exceeded"))
        print(f"Total duplicate shards ignored: {total_dupes}", flush=True)
        print(f"Conversations hitting max_turns cap: {n_max_turns}", flush=True)


summarize("snowball")
summarize("snowball-stateful")
print(f"\n=== TOTAL COST: ${sum(r['cost_usd'] for r in results):.4f} ===", flush=True)
print(f"=== ERRORS: {len(errors)} ===", flush=True)
