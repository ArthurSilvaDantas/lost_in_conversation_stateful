import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)  # data/, prompts/, results/ are all referenced via relative paths

# Auto-load .env so the script works without manual `source .env` in the shell
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# If OPENROUTER_API_KEY is set, swap model_openai with model_openrouter transparently
if os.environ.get("OPENROUTER_API_KEY"):
    import model_openrouter
    sys.modules['model_openai'] = model_openrouter

import argparse
import random
import multiprocessing
import json
import tqdm
from simulator_sharded import ConversationSimulatorSharded
from simulator_full import ConversationSimulatorFull
from simulator_snowball import ConversationSimulatorSnowball
from simulator_snowball_stateful import ConversationSimulatorSnowballStateful
from concurrent.futures import ThreadPoolExecutor
from utils_log import get_run_counts
from collections import Counter


def run_simulation(todo):
    dataset_fn = todo["dataset_fn"]
    try:
        assistant_temp = todo.get("assistant_temperature", 1.0)
        user_temp = todo.get("user_temperature", 1.0)
        if todo["conv_type"].startswith("full"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("concat"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], run_concat=True, temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("sharded"):
            conversation_simulator = ConversationSimulatorSharded(todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"], assistant_temperature=assistant_temp, user_temperature=user_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"] == "snowball":
            conversation_simulator = ConversationSimulatorSnowball(todo["sample"]["task"], todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"], log_folder=args.log_folder, dataset_fn=dataset_fn)
        elif todo["conv_type"] == "snowball-stateful":
            conversation_simulator = ConversationSimulatorSnowballStateful(todo["sample"]["task"], todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"], log_folder=args.log_folder, dataset_fn=dataset_fn)
        conversation_simulator.run(verbose=args.verbose)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tqdm.tqdm.write(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file", type=str, default="data/sharded_instructions_600.json", help="Dataset file to use")

    parser.add_argument("--N_full_runs", type=int, default=0, help="Number of full runs per model")
    parser.add_argument("--N_concat_runs", type=int, default=0, help="Number of concat runs per model")
    parser.add_argument("--N_sharded_runs", type=int, default=0, help="Number of sharded runs per model")
    parser.add_argument("--N_snowball_runs", type=int, default=0, help="Number of original-snowball runs per model")
    parser.add_argument("--N_snowball_stateful_runs", type=int, default=0, help="Number of snowball-stateful runs per model")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"],
                        help="List of models to run experiments with")
    parser.add_argument("--tasks", nargs="+", default=["math"], help="Tasks to run experiments with") # "code", "database", "actions", "math", "data2text", "summary", "translation"
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini", help="System model to use")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini", help="User model to use")
    parser.add_argument("--N_workers", type=int, default=1, help="Number of workers to run experiments with")
    parser.add_argument("--log_folder", type=str, default="results/logs", help="Log folder to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    parser.add_argument("--assistant_temperature", type=float, default=1.0, help="Temperature to use for assistant models")
    parser.add_argument("--user_temperature", type=float, default=1.0, help="Temperature to use for user models")

    args = parser.parse_args()

    dataset_fn = args.dataset_file.replace("\\", "/")
    if dataset_fn.startswith("./"):
        dataset_fn = dataset_fn[2:]

    with open(dataset_fn, "r") as f:
        samples = json.load(f)

    samples = [d for d in samples if d["task"] in args.tasks]

    print(f"Loaded {len(samples)} samples")
    random.shuffle(samples)
    all_todos = []

    sharded_extra = f"-at{args.assistant_temperature}-ut{args.user_temperature}" if args.assistant_temperature != 1.0 or args.user_temperature != 1.0 else ""
    st_extra = f"-t{args.assistant_temperature}" if args.assistant_temperature != 1.0 else ""
    sharded_ct, full_ct, concat_ct = f"sharded{sharded_extra}", f"full{st_extra}", f"concat{st_extra}"

    all_tasks = list(set([sample["task"] for sample in samples]))
    for assistant_model in args.models:
        sharded_run_counts, full_run_counts, concat_run_counts = Counter(), Counter(), Counter()
        snowball_run_counts, snowball_st_run_counts = Counter(), Counter()
        for task in all_tasks:
            sharded_run_counts.update(get_run_counts(sharded_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            full_run_counts.update(get_run_counts(full_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            concat_run_counts.update(get_run_counts(concat_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            snowball_run_counts.update(get_run_counts("snowball", task, assistant_model, dataset_fn, log_folder=args.log_folder))
            snowball_st_run_counts.update(get_run_counts("snowball-stateful", task, assistant_model, dataset_fn, log_folder=args.log_folder))
        print(f"Sharded run counts: {sharded_run_counts}")
        print(f"Full run counts: {full_run_counts}")
        print(f"Concat run counts: {concat_run_counts}")
        print(f"Snowball run counts: {snowball_run_counts}")
        print(f"Snowball-stateful run counts: {snowball_st_run_counts}")

        for sample in samples:
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": full_ct, "system_model": args.system_model, "dataset_fn": dataset_fn}] * (args.N_full_runs - full_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": concat_ct, "system_model": args.system_model, "dataset_fn": dataset_fn}] * (args.N_concat_runs - concat_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": sharded_ct, "system_model": args.system_model, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_sharded_runs - sharded_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": "snowball", "system_model": args.system_model, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_snowball_runs - snowball_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": "snowball-stateful", "system_model": args.system_model, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_snowball_stateful_runs - snowball_st_run_counts[sample["task_id"]])

    if args.assistant_temperature != 1.0 or args.user_temperature != 1.0:
        for todo in all_todos:
            todo["assistant_temperature"] = args.assistant_temperature
            todo["user_temperature"] = args.user_temperature

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_simulation, all_todos), total=len(all_todos)))
