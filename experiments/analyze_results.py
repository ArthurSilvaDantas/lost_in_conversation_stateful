import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


def load(fn):
    try:
        with open(fn) as f:
            return [json.loads(l) for l in f]
    except FileNotFoundError:
        return []


def cost(rec):
    return sum(m.get("cost_usd", 0) or 0 for m in rec["trace"])


def n_turns(rec, role="assistant"):
    return sum(1 for m in rec["trace"] if m["role"] == role)


def is_correct_unified(rec):
    return bool(rec.get("is_correct")) or rec.get("score") == 1.0


def print_table(records, label):
    if not records:
        print(f"\n{label}: sem dados")
        return
    print(f"\n=== {label} (n={len(records)}) ===")
    print(f"{'task_id':<28}{'correto':<10}{'score':<8}{'turnos':<8}{'custo':<10}")
    for r in sorted(records, key=lambda x: x["task_id"]):
        score_str = "NA" if r.get("score") is None else str(r["score"])
        print(f"{r['task_id']:<28}{str(is_correct_unified(r)):<10}{score_str:<8}{n_turns(r):<8}${cost(r):.4f}")
    acc = sum(1 for r in records if is_correct_unified(r)) / len(records)
    avg_turns = sum(n_turns(r) for r in records) / len(records)
    total_cost = sum(cost(r) for r in records)
    print(f"\nResumo: acuracia={acc:.1%}  turnos_medios={avg_turns:.1f}  custo_total=${total_cost:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="math")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--paired-only", action="store_true", help="Compara apenas task_ids presentes nos dois conv_types")
    args = parser.parse_args()

    orig_fn = f"results/logs/{args.task}/snowball/snowball_{args.task}_{args.model}.jsonl"
    state_fn = f"results/logs/{args.task}/snowball-stateful/snowball-stateful_{args.task}_{args.model}.jsonl"

    orig = load(orig_fn)
    state = load(state_fn)

    if args.paired_only:
        common = {r["task_id"] for r in orig} & {r["task_id"] for r in state}
        orig = [r for r in orig if r["task_id"] in common]
        state = [r for r in state if r["task_id"] in common]
        print(f"Filtrando para {len(common)} task_ids presentes em ambos os conv_types")

    print_table(orig, f"snowball (original) — {args.task}/{args.model}")
    print_table(state, f"snowball-stateful — {args.task}/{args.model}")


if __name__ == "__main__":
    main()
