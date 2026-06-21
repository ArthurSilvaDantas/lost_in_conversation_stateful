from openai import OpenAI
import os, time, json, re

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]
    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")
    return messages

def _base_model_name(model):
    # openrouter uses "openai/gpt-4o-mini" — strip provider prefix for pricing lookup
    return model.split("/")[-1]

# gpt-4o-mini context limit: 128K tokens. Reserve 2K for completion.
# Proxy: 1 token ≈ 4 chars → max input ≈ 126K * 4 = 504K chars.
_MAX_INPUT_CHARS = 480_000

def _truncate_to_fit(messages):
    """Remove turns from the middle when conversation exceeds the context limit.

    Always keeps: messages[0] (system) + the last 6 messages.
    The snowball/stateful user messages already contain a running summary of all
    prior hints, so dropping old turns does not lose information — it only removes
    redundant earlier exchanges, keeping the comparison neutral for both approaches.
    """
    total = sum(len(m.get("content") or "") for m in messages)
    if total <= _MAX_INPUT_CHARS:
        return messages

    keep_tail = 6  # last 3 user-assistant pairs
    while total > _MAX_INPUT_CHARS and len(messages) > 1 + keep_tail:
        # remove the oldest non-system message
        messages = [messages[0]] + messages[2:]
        total = sum(len(m.get("content") or "") for m in messages)

    return messages

class OpenRouter_Model:
    def __init__(self):
        assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY not set"
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def cost_calculator(self, model, usage):
        base_model = _base_model_name(model)
        prompt_tokens = usage['prompt_tokens']
        prompt_tokens_cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached
        completion_tokens = usage['completion_tokens']

        if base_model.startswith("gpt-4o-mini"):
            inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        else:
            inp_token_cost, out_token_cost = 0.0, 0.0  # unknown model — skip cost

        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * 0.5) / 1000) * inp_token_cost \
                  + (completion_tokens / 1000) * out_token_cost
        return total_usd

    def generate(self, messages, model="gpt-4o-mini", timeout=60, max_retries=5, temperature=1.0,
                 is_json=False, return_metadata=False, max_tokens=None, variables={}):
        kwargs = {}
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}

        messages = format_messages(messages, variables)
        messages = _truncate_to_fit(messages)

        # OpenRouter requires provider-prefixed model names
        if "/" not in model:
            model = f"openai/{model}"

        N = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, timeout=timeout,
                    max_completion_tokens=max_tokens, temperature=temperature, **kwargs
                )
                break
            except Exception as e:
                N += 1
                if N >= max_retries:
                    raise Exception(f"Failed to get response from OpenRouter: {e}")
                time.sleep(4)

        response = response.to_dict()
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)

        if not return_metadata:
            return response_text
        return {
            "message": response_text,
            "total_tokens": usage['total_tokens'],
            "prompt_tokens": usage['prompt_tokens'],
            "prompt_tokens_cached": prompt_tokens_cached,
            "completion_tokens": usage['completion_tokens'],
            "total_usd": total_usd,
        }

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        kwargs["return_metadata"] = True
        response = self.generate(messages, model, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response


model = OpenRouter_Model()
generate = model.generate
generate_json = model.generate_json
