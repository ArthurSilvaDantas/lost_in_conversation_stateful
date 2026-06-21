import random
import tqdm

from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from system_agent import SystemAgent
from user_agent import UserAgent
from tasks import get_task
from model_openai import generate
from concurrent.futures import ThreadPoolExecutor
from utils_log import get_run_counts
from collections import Counter

snowball_message = """Just to reiterate,
{HINTS_SO_FAR}

Also,

{LAST_HINT}
"""

class ConversationSimulatorSnowball:
    def __init__(self, task_name, sample, assistant_model="gpt-4o-mini", system_model="gpt-4o-mini", user_model="gpt-4o-mini", log_folder="logs", dataset_fn=None):
        assert task_name in ["database", "wmt", "summary", "math", "apis", "data2text", "python", "actions", "code", "translation"]

        self.task_name = task_name
        self.task = get_task(task_name)
        self.dataset_fn = dataset_fn if dataset_fn is not None else self.task.get_dataset_file()
        self.log_folder = log_folder
        self.sample = sample
        self.assistant_model = assistant_model
        self.system_model = system_model
        self.user_model = user_model
        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(task_name, system_model, sample)

        self.system_message = self.task.generate_system_prompt(self.sample)
        self.answer_description = self.task.get_answer_description()
        self.user_response_template = snowball_message

        self.max_turns = max(10, 3 * len(sample["shards"]))
        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):

        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None
        user_response_history = []

        shards = self.sample["shards"]

        while not is_completed:
            if self.get_num_turns("assistant") >= self.max_turns:
                break

            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "hint_revealed"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all hints revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break # no need to keep going, nothing else to reveal

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            # 1. get a user response
            user_response, hint_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample)

            if hint_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "hint_revealed", "shard_id": hint_revealed_id}, "timestamp": date_str()})
                user_response_history.append(user_response)
                if verbose:
                    print_colored(f"[log] hint revealed: {hint_revealed_id}", "blue")

            if len(revealed_shard_ids) > 0:
                # Create a new user response that includes all hints so far
                user_response = self.user_response_template.format(HINTS_SO_FAR="\n".join([f"- {hint}" for hint in user_response_history[:-1]]), LAST_HINT=user_response_history[-1])

            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            # 2. get the assistant's response
            assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=1.0, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            # 3. evaluate if the last turn in the conversation asks a clarification question, or proposes a solution.
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)

            # log it
            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})
            if verbose:
                print_colored(f"[log] system verification: {system_verification_response}", "blue")

            if system_verification_response["response_type"] == "answer_attempt":
                # 4. extract the answer
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:
                    # we skip eval, and only run the evaluator on the last turn
                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

                    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                # let's log that
                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
                    if verbose:
                        print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")
                # For now, if the system last assistant response is incorrect;
            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue # end of the turn
        if save_log:
            log_conversation("snowball", self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score


def single_run(todo):
    conversation_simulator = ConversationSimulatorSnowball(args.task, todo["sample"], assistant_model=todo["assistant_model"], system_model=todo["system_model"], user_model=todo["user_model"])
    conversation_simulator.run(verbose=args.verbose, save_log=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="database", choices=["code", "database", "actions", "math", "data2text", "summary"])
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--N_snowball_runs", type=int, default=2, help="Number of snowball runs per model")
    parser.add_argument("--N_workers", type=int, default=7, help="Number of workers to run experiments with")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "claude3-haiku", "claude3.5-sonnet", "gemini-1.5-pro"],
                        help="List of models to run experiments with")
    args = parser.parse_args()

    task = get_task(args.task)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples ({args.task})")
    dataset_fn = task.get_dataset_file()
    random.shuffle(samples)
    all_todos = []

    for assistant_model in args.models:
        run_counts = get_run_counts("snowball", args.task, assistant_model, dataset_fn)

        for sample in samples:
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": "snowball", "system_model": args.system_model, "user_model": args.user_model}] * (args.N_snowball_runs - run_counts[sample["task_id"]])

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(single_run, all_todos), total=len(all_todos)))
