import random
import tqdm
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation, get_run_counts
from system_agent import SystemAgent
from user_agent import UserAgent
from tasks import get_task
from model_openai import generate


class ConversationState(Enum):
    TURN_START = auto()
    REVEAL_NEW_SHARD = auto()
    BUILD_USER_MESSAGE = auto()
    ASSISTANT_RESPONDING = auto()
    CLASSIFY_RESPONSE = auto()
    EXTRACT_AND_EVALUATE = auto()
    TURN_END = auto()
    CONVERSATION_END = auto()


# Diagram of valid transitions:
#
#   TURN_START
#     ├─(shards remain)──► REVEAL_NEW_SHARD
#     └─(all revealed)───► CONVERSATION_END
#
#   REVEAL_NEW_SHARD ──────────────────────► BUILD_USER_MESSAGE
#
#   BUILD_USER_MESSAGE ────────────────────► ASSISTANT_RESPONDING
#
#   ASSISTANT_RESPONDING ──────────────────► CLASSIFY_RESPONSE
#
#   CLASSIFY_RESPONSE
#     ├─(answer_attempt)─► EXTRACT_AND_EVALUATE
#     └─(clarification)──► TURN_END
#
#   EXTRACT_AND_EVALUATE
#     ├─(correct)────────► CONVERSATION_END
#     └─(incorrect)──────► TURN_END
#
#   TURN_END
#     ├─(shards remain)──► TURN_START
#     └─(all revealed)───► CONVERSATION_END
#
#   CONVERSATION_END  (terminal)


VALID_TRANSITIONS = {
    ConversationState.TURN_START: {
        ConversationState.REVEAL_NEW_SHARD,
        ConversationState.CONVERSATION_END,
    },
    ConversationState.REVEAL_NEW_SHARD: {
        ConversationState.BUILD_USER_MESSAGE,
    },
    ConversationState.BUILD_USER_MESSAGE: {
        ConversationState.ASSISTANT_RESPONDING,
    },
    ConversationState.ASSISTANT_RESPONDING: {
        ConversationState.CLASSIFY_RESPONSE,
    },
    ConversationState.CLASSIFY_RESPONSE: {
        ConversationState.EXTRACT_AND_EVALUATE,
        ConversationState.TURN_END,
    },
    ConversationState.EXTRACT_AND_EVALUATE: {
        ConversationState.CONVERSATION_END,
        ConversationState.TURN_END,
    },
    ConversationState.TURN_END: {
        ConversationState.TURN_START,
        ConversationState.CONVERSATION_END,
    },
    ConversationState.CONVERSATION_END: set(),
}

# Identical to the original snowball simulator ("Lost in Conversation" paper).
# The state machine is the ONLY variable — template is kept constant across all tasks.
SNOWBALL_RECAP_TEMPLATE = """Just to reiterate,
{RECAP_BLOCK}

Also,

{NEW_SHARD}
"""


class ConversationSimulatorSnowballStateful:
    def __init__(
        self,
        task_name,
        sample,
        assistant_model="gpt-4o-mini",
        system_model="gpt-4o-mini",
        user_model="gpt-4o-mini",
        log_folder="logs",
        dataset_fn=None,
    ):
        assert task_name in ["database", "translation", "summary", "math", "actions", "data2text", "code"]

        self.task_name = task_name
        self.task = get_task(task_name)
        self.dataset_fn = dataset_fn if dataset_fn is not None else self.task.get_dataset_file()
        self.sample = sample
        self.assistant_model = assistant_model
        self.system_model = system_model
        self.user_model = user_model
        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(task_name, system_model, sample)

        self.log_folder = log_folder
        self.system_message = self.task.generate_system_prompt(self.sample)
        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        self._recap_template = SNOWBALL_RECAP_TEMPLATE

        # State machine
        self.state = ConversationState.TURN_START
        self.revealed_shard_ids: set = set()
        self.user_response_history: list = []  # one raw shard text per revealed shard
        self.turn_count = 0
        # Safety cap: the user agent (LLM) does not always respect the
        # "no repeated shards" instruction. Without a cap, a model stuck
        # re-revealing the same shard would never terminate.
        self.max_turns = max(10, 3 * len(sample["shards"]))

        # Scratch space for the current turn (set in REVEAL_NEW_SHARD, consumed in BUILD_USER_MESSAGE)
        self._current_raw_response: str = ""
        self._current_shard_id: int = -1
        self._current_cost_usd: float = 0.0

        # Final results
        self.is_correct = False
        self.score = None

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: ConversationState, verbose=False, metadata=None):
        assert new_state in VALID_TRANSITIONS[self.state], (
            f"Invalid transition: {self.state.name} → {new_state.name}"
        )
        if verbose:
            print_colored(f"[state] {self.state.name} → {new_state.name}", "purple")

        entry = {
            "role": "log",
            "content": {
                "type": "state_transition",
                "from": self.state.name,
                "to": new_state.name,
            },
            "timestamp": date_str(),
        }
        if metadata:
            entry["content"].update(metadata)
        self.trace.append(entry)
        self.state = new_state

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self, verbose=False, save_log=True):
        is_reasoning_model = (
            "o1" in self.assistant_model
            or "o3" in self.assistant_model
            or "deepseek-r1" in self.assistant_model
        )
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        shards = self.sample["shards"]

        while self.state != ConversationState.CONVERSATION_END:

            # ----------------------------------------------------------
            # TURN_START: decide whether there is anything left to reveal
            # ----------------------------------------------------------
            if self.state == ConversationState.TURN_START:
                if len(self.revealed_shard_ids) == len(shards):
                    self._transition(ConversationState.CONVERSATION_END, verbose)
                elif self.turn_count >= self.max_turns:
                    self.trace.append({
                        "role": "log",
                        "content": {"type": "max_turns_exceeded", "turn_count": self.turn_count},
                        "timestamp": date_str(),
                    })
                    if verbose:
                        print_colored(f"[log] max turns ({self.max_turns}) exceeded, ending conversation", "blue")
                    self._transition(ConversationState.CONVERSATION_END, verbose)
                else:
                    self._transition(ConversationState.REVEAL_NEW_SHARD, verbose)

            # ----------------------------------------------------------
            # REVEAL_NEW_SHARD: user agent picks and reveals the next shard.
            # If the user agent re-reveals an already-revealed shard_id
            # (it does not always respect "no repeated shards"), we treat
            # it as a no-op rather than polluting the recap with duplicates.
            # ----------------------------------------------------------
            elif self.state == ConversationState.REVEAL_NEW_SHARD:
                self.turn_count += 1
                raw_response, shard_id, cost_usd = self.user_agent.generate_response(
                    self.trace, self.sample
                )
                self._current_raw_response = raw_response
                self._current_cost_usd = cost_usd

                is_duplicate = shard_id != -1 and shard_id in self.revealed_shard_ids
                self._current_shard_id = -1 if is_duplicate else shard_id

                if is_duplicate:
                    self.trace.append({
                        "role": "log",
                        "content": {"type": "duplicate_shard_ignored", "shard_id": shard_id},
                        "timestamp": date_str(),
                    })
                    if verbose:
                        print_colored(f"[log] duplicate shard ignored: {shard_id}", "blue")
                elif shard_id != -1:
                    self.revealed_shard_ids.add(shard_id)
                    self.user_response_history.append(raw_response)
                    self.trace.append({
                        "role": "log",
                        "content": {"type": "hint_revealed", "shard_id": shard_id},
                        "timestamp": date_str(),
                    })
                    if verbose:
                        print_colored(f"[log] hint revealed: {shard_id}", "blue")

                self._transition(
                    ConversationState.BUILD_USER_MESSAGE,
                    verbose,
                    metadata={"new_shard_id": self._current_shard_id, "raw_shard_id_returned": shard_id},
                )

            # ----------------------------------------------------------
            # BUILD_USER_MESSAGE: construct the structured user message.
            # All turns use the recap template (mirrors original snowball).
            # On the first shard, recap_block is empty — identical to the
            # "Just to reiterate,\n\nAlso,\n\n{hint}" the original produces.
            # ----------------------------------------------------------
            elif self.state == ConversationState.BUILD_USER_MESSAGE:
                shard_was_revealed = self._current_shard_id != -1

                # Mirror original snowball: apply the recap template whenever at least
                # one shard has been revealed — even on the first shard (recap_block
                # will be an empty string, producing the same "Just to reiterate,\n\nAlso,\n"
                # that the original produces when HINTS_SO_FAR is empty).
                if shard_was_revealed and len(self.user_response_history) > 1:
                    recap_shards = self.user_response_history[:-1]
                    new_shard = self.user_response_history[-1]
                    recap_block = "\n".join(f"- {s}" for s in recap_shards)
                    user_message = self._recap_template.format(
                        RECAP_BLOCK=recap_block, NEW_SHARD=new_shard
                    )
                    msg_structure = "recap+new"
                else:
                    user_message = self._current_raw_response
                    msg_structure = "new_only" if shard_was_revealed else "no_shard"

                self.trace.append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": date_str(),
                    "cost_usd": self._current_cost_usd,
                    "message_structure": msg_structure,
                    "new_shard_id": self._current_shard_id,
                    "n_recap_shards": len(self.user_response_history) - 1 if shard_was_revealed else 0,
                })
                if verbose:
                    print_colored(f"[user] {user_message}", "green")

                self._transition(ConversationState.ASSISTANT_RESPONDING, verbose)

            # ----------------------------------------------------------
            # ASSISTANT_RESPONDING: the LLM under test generates a reply
            # ----------------------------------------------------------
            elif self.state == ConversationState.ASSISTANT_RESPONDING:
                assistant_response_obj = generate(
                    extract_conversation(self.trace, to_str=False),
                    model=self.assistant_model,
                    temperature=1.0,
                    return_metadata=True,
                    max_tokens=max_assistant_tokens,
                )
                assistant_response = assistant_response_obj["message"]
                self.trace.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": date_str(),
                    "cost_usd": assistant_response_obj["total_usd"],
                })
                if verbose:
                    print_colored(f"[assistant] {assistant_response}", "red")

                self._transition(ConversationState.CLASSIFY_RESPONSE, verbose)

            # ----------------------------------------------------------
            # CLASSIFY_RESPONSE: system decides if the turn is an answer
            # attempt, a clarification question, or a discussion turn
            # ----------------------------------------------------------
            elif self.state == ConversationState.CLASSIFY_RESPONSE:
                system_verification_response, verification_cost_usd = (
                    self.system_agent.verify_system_response(self.trace)
                )
                self.trace.append({
                    "role": "log",
                    "content": {
                        "type": "system-verification",
                        "response": system_verification_response,
                    },
                    "timestamp": date_str(),
                    "cost_usd": verification_cost_usd,
                })
                if verbose:
                    print_colored(f"[log] system verification: {system_verification_response}", "blue")

                if system_verification_response["response_type"] == "answer_attempt":
                    self._transition(ConversationState.EXTRACT_AND_EVALUATE, verbose)
                else:
                    self._transition(ConversationState.TURN_END, verbose)

            # ----------------------------------------------------------
            # EXTRACT_AND_EVALUATE: extract the answer span and score it
            # ----------------------------------------------------------
            elif self.state == ConversationState.EXTRACT_AND_EVALUATE:
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_last_turn = len(self.revealed_shard_ids) == len(shards)

                self.is_correct, self.score = None, None  # mirror original: reset before each evaluation
                if self.task_name == "summary" and not is_last_turn:
                    evaluation_return = {"score": 0.0}
                    self.score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    assert type(evaluation_return) is dict and (
                        "score" in evaluation_return or "is_correct" in evaluation_return
                    )
                    self.is_correct = evaluation_return.get("is_correct", None)
                    self.score = evaluation_return.get("score", None)

                self.trace.append({
                    "role": "log",
                    "content": {
                        "type": "answer-evaluation",
                        "exact_answer": extracted_answer,
                        "is_correct": self.is_correct,
                        "score": self.score,
                        "evaluation_return": evaluation_return,
                    },
                    "timestamp": date_str(),
                })
                if verbose:
                    print_colored(
                        f"[log] answer evaluation:\n```{extracted_answer}\n```\n"
                        f"({'correct' if self.is_correct else 'incorrect'}; score: {self.score})",
                        "blue",
                    )

                if self.is_correct:
                    self.trace.append({
                        "role": "log",
                        "content": {"type": "conversation-completed", "is_correct": self.is_correct},
                        "timestamp": date_str(),
                    })
                    self._transition(ConversationState.CONVERSATION_END, verbose)
                else:
                    self._transition(ConversationState.TURN_END, verbose)

            # ----------------------------------------------------------
            # TURN_END: decide if the conversation can continue
            # ----------------------------------------------------------
            elif self.state == ConversationState.TURN_END:
                if len(self.revealed_shard_ids) == len(shards):
                    self._transition(ConversationState.CONVERSATION_END, verbose)
                else:
                    self._transition(ConversationState.TURN_START, verbose)

        if save_log:
            log_conversation(
                "snowball-stateful",
                self.task.get_task_name(),
                self.sample["task_id"],
                self.dataset_fn,
                self.assistant_model,
                self.system_model,
                self.user_model,
                self.trace,
                self.is_correct,
                self.score,
                log_folder=self.log_folder,
            )
        return self.is_correct, self.score


# ----------------------------------------------------------------------
# CLI entry point (mirrors simulator_snowball.py)
# ----------------------------------------------------------------------

def single_run(todo):
    simulator = ConversationSimulatorSnowballStateful(
        args.task,
        todo["sample"],
        assistant_model=todo["assistant_model"],
        system_model=todo["system_model"],
        user_model=todo["user_model"],
        log_folder=args.log_folder,
    )
    simulator.run(verbose=args.verbose, save_log=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="database",
                        choices=["code", "database", "actions", "math", "data2text", "summary"])
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--N_runs", type=int, default=2, help="Runs per sample per model")
    parser.add_argument("--N_workers", type=int, default=7)
    parser.add_argument("--log_folder", type=str, default="results/logs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash",
                                 "claude3-haiku", "claude3.5-sonnet", "gemini-1.5-pro"])
    args = parser.parse_args()

    task = get_task(args.task)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples ({args.task})")
    dataset_fn = task.get_dataset_file()
    random.shuffle(samples)

    all_todos = []
    for assistant_model in args.models:
        run_counts = get_run_counts("snowball-stateful", args.task, assistant_model, dataset_fn)
        for sample in samples:
            remaining = args.N_runs - run_counts[sample["task_id"]]
            all_todos += [{
                "sample": sample,
                "assistant_model": assistant_model,
                "system_model": args.system_model,
                "user_model": args.user_model,
            }] * remaining

    random.shuffle(all_todos)
    print(f"Running {len(all_todos)} conversations")
    print(Counter(todo["assistant_model"] for todo in all_todos))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(single_run, all_todos), total=len(all_todos)))
