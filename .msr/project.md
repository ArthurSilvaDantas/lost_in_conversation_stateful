---
stage: early-research
research-area: agents
last-updated: 2025-05-09
---

# Lost in Conversation

**Team:** Microsoft Research, Salesforce Research
**Contributors:** Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville

---

## What it is

Lost in Conversation is a simulation framework and benchmark for evaluating LLM performance in multi-turn, underspecified conversations. It transforms existing single-turn, fully-specified instructions into "sharded" instructions — sets of smaller information pieces revealed gradually across conversation turns — enabling controlled comparison of LLM behavior in single- vs. multi-turn settings across six generation tasks: code, database, actions, math, data-to-text, and summarization.

🔗 [microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation) · [Paper (arXiv)](https://arxiv.org/abs/2505.06120) · [Dataset (HuggingFace)](https://huggingface.co/datasets/Microsoft/lost_in_conversation)

## Core idea

LLM evaluation has predominantly focused on single-turn, fully-specified instructions, yet real users frequently underspecify their needs and refine them across multiple conversation turns. Lost in Conversation's insight: by "sharding" existing benchmark instructions into ordered sets of atomic information units and simulating their gradual disclosure, we can directly compare LLM performance in single- and multi-turn settings on identical underlying tasks. Large-scale experiments (200,000+ simulated conversations across 15 LLMs) reveal a universal 39% average performance drop in multi-turn settings, decomposed into a minor loss in aptitude and a dramatic increase in unreliability. The key finding is that LLMs make premature assumptions, generate bloated answers that overly rely on earlier (incorrect) attempts, and exhibit a "loss-in-middle-turns" phenomenon — and these behaviors persist even with reasoning models, lower temperatures, or agent-style recapitulation interventions. This is hard to discover without the controlled sharding methodology, as episodic multi-turn benchmarks overestimate LLM multi-turn capabilities by not requiring information fusion across turns.

## Why it matters

**To the field:** Establishes that multi-turn, underspecified conversation is a universal weakness of current LLMs — not a model-specific or task-specific issue. Introduces the sharding methodology as a reusable tool for converting any single-turn benchmark into a multi-turn evaluation, and decomposes performance degradation into aptitude vs. reliability, revealing that unreliability (not aptitude loss) is the primary driver. The finding that even two-turn conversations trigger degradation, and that neither reasoning models nor temperature reduction can mitigate it, reframes how the community should think about multi-turn evaluation.

**Product integration:** Directly relevant to any Microsoft product involving multi-turn LLM interaction (e.g., Copilot, Bing Chat). The findings provide concrete evidence that system-level interventions (recapitulation, snowballing) offer only partial mitigation (~15-20%), motivating the need for native multi-turn reliability improvements in foundation models.

**Future directions:** Opens research questions on training LLMs for multi-turn reliability (not just aptitude), designing better agent-level interventions for underspecified conversations, extending the sharding methodology to creative/multilingual/multimodal tasks, and understanding whether RLHF/DPO-style training can specifically target the identified failure modes (premature answering, answer bloat, middle-turn forgetting).

## Collaborations

- **External / industry:** Salesforce Research (Hiroaki Hayashi, Yingbo Zhou) — co-authorship and experimental contributions

## Current status

**Headline:** Published at ICLR 2026, and received the Outstanding Paper Award!

- 200,000+ conversations simulated across 15 LLMs, 6 tasks, 600 sharded instructions
- Performance degradation observed universally, from small models (Llama3.1-8B) to state-of-the-art (Gemini 2.5 Pro, GPT-4.1)
- Reasoning models (o3, Deepseek-R1) and temperature reduction shown to be ineffective mitigations
- Open-source code and dataset released on GitHub and HuggingFace

## Related landscape

- [MT-Bench — Multi-turn benchmark using LLM-as-a-judge, episodic task design](https://arxiv.org/abs/2306.05685)
- [MT-Bench-101 — Fine-grained multi-turn evaluation with subtask categorization](https://arxiv.org/abs/2402.14762)
- [MINT — Multi-turn interaction with tools and language feedback (ICLR 2024)](https://arxiv.org/abs/2309.10691)
- [MathChat — Multi-turn math reasoning benchmark](https://arxiv.org/abs/2405.19444)
- [Herlihy et al. — LLM response categorization under underspecified queries](https://arxiv.org/abs/2406.01633)
- [ChatBench — Human-AI evaluation moving beyond static benchmarks](https://arxiv.org/abs/2504.07114)

## Real-world impact

- Open-sourced simulation framework enabling reproducible multi-turn LLM evaluation
- Released 600 sharded instructions spanning six generation tasks on GitHub and HuggingFace
- Identified actionable user strategies (retry and consolidate) for improved LLM interaction outcomes
- Provided concrete reliability benchmarks and a call-to-action for LLM builders to prioritize multi-turn reliability

## Publications & links

- [LLMs Get Lost In Multi-Turn Conversation — arXiv, 2025](https://arxiv.org/abs/2505.06120)
- [GitHub: microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation)
- [Dataset: Microsoft/lost_in_conversation (HuggingFace)](https://huggingface.co/datasets/Microsoft/lost_in_conversation)
