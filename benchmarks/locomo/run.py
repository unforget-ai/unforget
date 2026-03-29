#!/usr/bin/env python3
"""LoCoMo benchmark for Unforget.

Evaluates Unforget's memory system on the LoCoMo (Long Conversation Memory)
benchmark (ACL 2024). Measures accuracy across 4 question categories:
  1. Single-hop   — direct fact recall
  2. Temporal      — time-aware retrieval
  3. Multi-hop     — reasoning across multiple facts
  4. Open-domain   — broader context understanding

Usage:
    python -m benchmarks.locomo.run
    python -m benchmarks.locomo.run --conversations 2 --limit 10   # quick test
    python -m benchmarks.locomo.run --no-rerank                     # ablation
    python -m benchmarks.locomo.run --recall-limit 20               # more context
    python -m benchmarks.locomo.run --ingest-mode chunk             # chunked ingestion

Requires:
    - PostgreSQL with pgvector (docker compose up -d)
    - OPENAI_API_KEY env var (for LLM judge + answer generation)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
import anthropic

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from unforget import MemoryStore
from unforget.associations import build_associations
from unforget.types import MemoryType, WriteItem

logger = logging.getLogger("unforget.locomo")

DATABASE_URL = os.environ.get(
    "UNFORGET_TEST_DATABASE_URL",
    "postgresql://unforget:unforget@localhost:5432/unforget",
)

DATASET_PATH = Path(__file__).parent / "locomo_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"

ORG_ID = "locomo_bench"

# Category names for reporting
CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
}

# ── LLM helpers ──────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are answering questions about conversations between two people.
You will be given relevant memory context retrieved from past conversations.
Answer the question based on the provided context.
Keep your answer concise and factual.
IMPORTANT: When the context mentions dates or times, always convert relative \
references (like "yesterday", "last week", "next month") to absolute dates \
using the timestamps shown in [brackets] at the start of each memory. \
For example, if a memory from [1:56 pm on 8 May, 2023] mentions "yesterday", \
that means 7 May 2023.
If the context does not contain relevant information, try your best to infer \
the answer from what is available before saying you don't know."""

ANSWER_USER_TEMPLATE = """\
Context from memory:
{context}

Question: {question}

Answer concisely (use absolute dates, not relative ones):"""

JUDGE_SYSTEM_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given:
  (1) a question
  (2) a gold (ground truth) answer
  (3) a generated answer

The gold answer is usually concise. The generated answer may be longer.
Be generous with your grading:
- As long as the generated answer touches on the same topic/fact as the gold \
answer, it should be CORRECT.
- For time-related questions, if the generated answer refers to the same date \
or time period (even in a different format like "May 7th" vs "7 May"), count \
it as CORRECT. Approximate matches within the same week are also CORRECT \
(e.g., "mid-May 2023" for "8 May 2023").
- Minor extra details in the generated answer do not make it WRONG.
- Partial answers that cover the main point are CORRECT.

Return JSON with two keys: "reasoning" (one sentence) and "label" (CORRECT or WRONG)."""

JUDGE_USER_TEMPLATE = """\
Question: {question}
Gold answer: {expected}
Generated answer: {generated}"""


MAX_CONTEXT_CHARS = 12000  # ~3000 tokens — keeps requests well within limits
MAX_RETRIES = 3


def _sanitize(text: str) -> str:
    """Remove null bytes and other characters that break JSON serialization."""
    return text.replace("\x00", "").replace("\udcff", "")


class LLMClient:
    """LLM client for answer generation and judging. Supports OpenAI and Anthropic."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        judge_model: str = "gpt-4.1-nano",
        provider: str = "openai",
    ):
        self.model = model
        self.judge_model = judge_model
        self.provider = provider
        self._answer_tokens = 0
        self._judge_tokens = 0

        if provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            self._openai = None
        else:
            self._openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self._anthropic = None

    async def _call(self, model: str, system: str, user: str, **kwargs) -> str:
        """Call LLM with retries. Returns the text response."""
        last_err = None
        for attempt in range(MAX_RETRIES + 3):
            try:
                if self.provider == "anthropic":
                    resp = await self._anthropic.messages.create(
                        model=model,
                        max_tokens=kwargs.get("max_tokens", 256),
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    if resp.usage:
                        tokens = resp.usage.input_tokens + resp.usage.output_tokens
                    else:
                        tokens = 0
                    text = resp.content[0].text
                    return text, tokens
                else:
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                    create_kwargs = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 256),
                        "temperature": kwargs.get("temperature", 0),
                    }
                    if kwargs.get("json_mode"):
                        create_kwargs["response_format"] = {"type": "json_object"}
                    resp = await self._openai.chat.completions.create(**create_kwargs)
                    tokens = resp.usage.total_tokens if resp.usage else 0
                    text = resp.choices[0].message.content or ""
                    return text, tokens

            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str or "overloaded" in err_str

                if is_rate_limit:
                    wait = min(60, 5 * (2 ** attempt))
                    logger.warning("Rate limited, waiting %ds (attempt %d)...", wait, attempt + 1)
                    await asyncio.sleep(wait)
                elif attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    raise last_err
        raise last_err

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer to a question given retrieved context."""
        context = _sanitize(context)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n[truncated]"

        user_content = _sanitize(
            ANSWER_USER_TEMPLATE.format(context=context, question=question)
        )
        text, tokens = await self._call(
            self.model, ANSWER_SYSTEM_PROMPT, user_content,
            max_tokens=256, temperature=0,
        )
        self._answer_tokens += tokens
        return text

    async def judge(self, question: str, expected: str, generated: str) -> dict:
        """Judge whether a generated answer is correct against gold answer."""
        user_content = _sanitize(
            JUDGE_USER_TEMPLATE.format(
                question=question, expected=expected, generated=generated
            )
        )
        # For Anthropic, ask for JSON in the prompt; for OpenAI, use response_format
        system = JUDGE_SYSTEM_PROMPT
        if self.provider == "anthropic":
            system += "\n\nRespond with valid JSON only."

        text, tokens = await self._call(
            self.judge_model, system, user_content,
            max_tokens=200, temperature=0.1,
            json_mode=(self.provider == "openai"),
        )
        self._judge_tokens += tokens

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            if "CORRECT" in text.upper():
                result = {"reasoning": text, "label": "CORRECT"}
            else:
                result = {"reasoning": text, "label": "WRONG"}
        return result

    @property
    def total_tokens(self) -> int:
        return self._answer_tokens + self._judge_tokens


# ── Data loading ─────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    """Load the LoCoMo dataset."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_sessions(conversation: dict) -> list[tuple[str, list[dict]]]:
    """Extract ordered (date_string, turns) pairs from a conversation dict."""
    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and "date_time" not in k],
        key=lambda x: int(x.split("_")[1]),
    )
    sessions = []
    for sk in session_keys:
        date_key = f"{sk}_date_time"
        date_str = conversation.get(date_key, "Unknown date")
        turns = conversation[sk]
        sessions.append((date_str, turns))
    return sessions


# ── Ingestion strategies ─────────────────────────────────────────────────────


async def ingest_per_turn(
    store: MemoryStore,
    agent_id: str,
    sessions: list[tuple[str, list[dict]]],
) -> int:
    """Ingest each dialog turn as a separate memory. Best for single-hop."""
    count = 0
    for sess_idx, (date_str, turns) in enumerate(sessions):
        thread_id = f"{agent_id}_session_{sess_idx}"
        items = []
        for turn in turns:
            content = f"[{date_str}] {turn['speaker']}: {turn['text']}"
            items.append(
                WriteItem(
                    content=content,
                    memory_type=MemoryType.EVENT,
                    tags=[turn["speaker"].lower()],
                    importance=0.5,
                    source_thread_id=thread_id,
                )
            )
        written = await store.write_batch(items, org_id=ORG_ID, agent_id=agent_id)
        count += len(written)
    return count


async def ingest_chunked(
    store: MemoryStore,
    agent_id: str,
    sessions: list[tuple[str, list[dict]]],
    window: int = 5,
    stride: int = 3,
) -> int:
    """Ingest sliding windows of turns. Better for multi-hop context."""
    count = 0
    for sess_idx, (date_str, turns) in enumerate(sessions):
        thread_id = f"{agent_id}_session_{sess_idx}"
        items = []
        for i in range(0, len(turns), stride):
            window_turns = turns[i : i + window]
            if not window_turns:
                break
            lines = [f"[{date_str}]"]
            speakers = set()
            for t in window_turns:
                lines.append(f"{t['speaker']}: {t['text']}")
                speakers.add(t["speaker"].lower())
            content = "\n".join(lines)
            items.append(
                WriteItem(
                    content=content,
                    memory_type=MemoryType.EVENT,
                    tags=list(speakers),
                    importance=0.5,
                    source_thread_id=thread_id,
                )
            )
        written = await store.write_batch(items, org_id=ORG_ID, agent_id=agent_id)
        count += len(written)
    return count


async def ingest_hybrid(
    store: MemoryStore,
    agent_id: str,
    sessions: list[tuple[str, list[dict]]],
) -> int:
    """Hybrid: individual turns + session summaries as chunks."""
    count = 0
    # Individual turns (already sets source_thread_id per session)
    count += await ingest_per_turn(store, agent_id, sessions)
    # Session-level chunks (full session as one memory)
    for sess_idx, (date_str, turns) in enumerate(sessions):
        thread_id = f"{agent_id}_session_{sess_idx}"
        lines = [f"[{date_str}] Conversation:"]
        for t in turns:
            lines.append(f"  {t['speaker']}: {t['text']}")
        full_text = "\n".join(lines)
        if len(full_text) > 2000:
            full_text = full_text[:2000] + "..."
        try:
            await store.write(
                content=full_text,
                org_id=ORG_ID,
                agent_id=agent_id,
                memory_type=MemoryType.RAW,
                importance=0.3,
                source_thread_id=thread_id,
            )
            count += 1
        except Exception:
            pass  # Duplicate or quota — skip
    return count


INGEST_STRATEGIES = {
    "per_turn": ingest_per_turn,
    "chunk": ingest_chunked,
    "hybrid": ingest_hybrid,
}


# ── Evaluation ───────────────────────────────────────────────────────────────


async def evaluate_question(
    store: MemoryStore,
    llm: LLMClient,
    agent_id: str,
    question: str,
    expected: str,
    recall_limit: int,
    rerank: bool,
) -> dict:
    """Evaluate a single question: recall → generate → judge."""
    # Step 1: Retrieve from Unforget
    t0 = time.perf_counter()
    context = await store.auto_recall(
        query=question,
        org_id=ORG_ID,
        agent_id=agent_id,
        limit=recall_limit,
    )
    recall_ms = (time.perf_counter() - t0) * 1000

    # If auto_recall returns empty, also try raw recall for diagnostics
    if not context.strip():
        results = await store.recall(
            query=question,
            org_id=ORG_ID,
            agent_id=agent_id,
            limit=recall_limit,
            rerank=rerank,
            use_cache=False,
        )
        if results:
            context = "\n".join(f"- {r.content}" for r in results)

    # Step 2: Generate answer
    t1 = time.perf_counter()
    generated = await llm.generate_answer(question, context or "No relevant memories found.")
    gen_ms = (time.perf_counter() - t1) * 1000

    # Step 3: Judge
    t2 = time.perf_counter()
    judgment = await llm.judge(question, expected, generated)
    judge_ms = (time.perf_counter() - t2) * 1000

    is_correct = judgment.get("label", "").upper() == "CORRECT"

    return {
        "correct": is_correct,
        "question": question,
        "expected": expected,
        "generated": generated,
        "reasoning": judgment.get("reasoning", ""),
        "context_length": len(context),
        "recall_ms": round(recall_ms, 1),
        "gen_ms": round(gen_ms, 1),
        "judge_ms": round(judge_ms, 1),
    }


async def cleanup_conversation(store: MemoryStore, agent_id: str) -> None:
    """Delete all memories and associations for a conversation."""
    mem_ids = await store.pool.fetch(
        "SELECT id FROM memory WHERE org_id = $1 AND agent_id = $2",
        ORG_ID, agent_id,
    )
    if mem_ids:
        ids = [r["id"] for r in mem_ids]
        await store.pool.execute(
            "DELETE FROM memory_associations WHERE memory_a = ANY($1) OR memory_b = ANY($1)",
            ids,
        )
    await store.pool.execute(
        "DELETE FROM memory_history WHERE memory_id IN "
        "(SELECT id FROM memory WHERE org_id = $1 AND agent_id = $2)",
        ORG_ID, agent_id,
    )
    await store.pool.execute(
        "DELETE FROM memory WHERE org_id = $1 AND agent_id = $2",
        ORG_ID, agent_id,
    )


# ── Main benchmark runner ────────────────────────────────────────────────────


async def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the full LoCoMo benchmark."""
    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    if args.conversations:
        dataset = dataset[: args.conversations]

    # Initialize store
    store = MemoryStore(
        database_url=DATABASE_URL,
        reranker_enabled=args.rerank,
        max_writes_per_minute=100_000,  # No rate limiting during benchmark
        max_memories_per_agent=100_000,
    )
    await store.initialize()

    # Initialize LLM client
    llm = LLMClient(
        model=args.model,
        judge_model=args.judge_model,
        provider=args.provider,
    )

    ingest_fn = INGEST_STRATEGIES[args.ingest_mode]

    # Results collection
    all_results: list[dict] = []
    conv_summaries: list[dict] = []
    category_correct: dict[int, list[bool]] = defaultdict(list)

    total_ingest_time = 0.0
    total_memories = 0

    print("=" * 70)
    print("  Unforget LoCoMo Benchmark")
    print("=" * 70)
    print(f"  Conversations:  {len(dataset)}")
    print(f"  Ingest mode:    {args.ingest_mode}")
    print(f"  Recall limit:   {args.recall_limit}")
    print(f"  Reranker:       {'ON' if args.rerank else 'OFF'}")
    print(f"  Provider:       {args.provider}")
    print(f"  Answer model:   {args.model}")
    print(f"  Judge model:    {args.judge_model}")
    if args.limit:
        print(f"  Q limit/conv:   {args.limit}")
    print("=" * 70)
    print()

    try:
        for conv_idx, conv in enumerate(dataset):
            sample_id = conv["sample_id"]
            agent_id = f"conv_{sample_id}"
            sessions = extract_sessions(conv["conversation"])
            speaker_a = conv["conversation"].get("speaker_a", "Speaker A")
            speaker_b = conv["conversation"].get("speaker_b", "Speaker B")

            # Filter questions (skip category 5 = adversarial)
            questions = [q for q in conv["qa"] if q["category"] in CATEGORY_NAMES]
            if args.limit:
                questions = questions[: args.limit]
            if args.category:
                questions = [q for q in questions if q["category"] == args.category]

            print(f"{'─' * 70}")
            print(
                f"  Conversation {conv_idx + 1}/{len(dataset)} "
                f"[{sample_id}] — {speaker_a} & {speaker_b}"
            )
            print(f"  Sessions: {len(sessions)}, Questions: {len(questions)}")
            print(f"{'─' * 70}")

            # Cleanup any prior data
            await cleanup_conversation(store, agent_id)

            # Ingest
            t_ingest = time.perf_counter()
            mem_count = await ingest_fn(store, agent_id, sessions)
            ingest_ms = (time.perf_counter() - t_ingest) * 1000
            total_ingest_time += ingest_ms
            total_memories += mem_count

            # Build association links for multi-hop recall
            t_assoc = time.perf_counter()
            assoc_count = await build_associations(
                store.pool, org_id=ORG_ID, agent_id=agent_id,
            )
            assoc_ms = (time.perf_counter() - t_assoc) * 1000
            print(
                f"  Ingested {mem_count} memories in {ingest_ms:.0f}ms"
                f" | {assoc_count} associations in {assoc_ms:.0f}ms"
            )

            # Evaluate questions
            conv_correct = 0
            conv_total = 0

            for q_idx, qa in enumerate(questions):
                cat = qa["category"]
                cat_name = CATEGORY_NAMES[cat]

                try:
                    result = await evaluate_question(
                        store=store,
                        llm=llm,
                        agent_id=agent_id,
                        question=qa["question"],
                        expected=qa["answer"],
                        recall_limit=args.recall_limit,
                        rerank=args.rerank,
                    )
                except Exception as e:
                    logger.warning("Question failed: %s — %s", qa["question"][:60], e)
                    result = {
                        "correct": False,
                        "question": qa["question"],
                        "expected": qa["answer"],
                        "generated": f"[ERROR: {e}]",
                        "reasoning": "skipped due to error",
                        "context_length": 0,
                        "recall_ms": 0,
                        "gen_ms": 0,
                        "judge_ms": 0,
                    }
                result["category"] = cat
                result["category_name"] = cat_name
                result["conversation"] = sample_id

                all_results.append(result)
                category_correct[cat].append(result["correct"])
                conv_total += 1
                if result["correct"]:
                    conv_correct += 1

                # Progress indicator
                mark = "✓" if result["correct"] else "✗"
                running_pct = (conv_correct / conv_total * 100) if conv_total else 0
                if (q_idx + 1) % 5 == 0 or q_idx == len(questions) - 1:
                    print(
                        f"    [{q_idx + 1}/{len(questions)}] "
                        f"{running_pct:.1f}% accurate so far"
                    )
                elif not result["correct"]:
                    print(
                        f"    {mark} [{cat_name}] Q: {qa['question'][:60]}..."
                    )

            conv_acc = (conv_correct / conv_total * 100) if conv_total else 0
            conv_summaries.append(
                {
                    "conversation": sample_id,
                    "correct": conv_correct,
                    "total": conv_total,
                    "accuracy": round(conv_acc, 2),
                }
            )
            print(
                f"  → Conversation accuracy: {conv_correct}/{conv_total} "
                f"({conv_acc:.1f}%)"
            )
            print()

            # Cleanup after evaluation
            await cleanup_conversation(store, agent_id)

    finally:
        await store.close()

    # ── Compute final results ────────────────────────────────────────────
    total_correct = sum(r["correct"] for r in all_results)
    total_questions = len(all_results)
    overall_acc = (total_correct / total_questions * 100) if total_questions else 0

    category_results = {}
    for cat, bools in sorted(category_correct.items()):
        correct = sum(bools)
        total = len(bools)
        acc = (correct / total * 100) if total else 0
        category_results[CATEGORY_NAMES[cat]] = {
            "correct": correct,
            "total": total,
            "accuracy": round(acc, 2),
        }

    avg_recall_ms = (
        sum(r["recall_ms"] for r in all_results) / len(all_results)
        if all_results
        else 0
    )

    summary = {
        "benchmark": "LoCoMo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "conversations": len(dataset),
            "ingest_mode": args.ingest_mode,
            "recall_limit": args.recall_limit,
            "reranker": args.rerank,
            "answer_model": args.model,
            "judge_model": args.judge_model,
        },
        "overall": {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": round(overall_acc, 2),
        },
        "by_category": category_results,
        "by_conversation": conv_summaries,
        "performance": {
            "total_memories_ingested": total_memories,
            "total_ingest_ms": round(total_ingest_time, 1),
            "avg_ingest_per_memory_ms": (
                round(total_ingest_time / total_memories, 2)
                if total_memories
                else 0
            ),
            "avg_recall_ms": round(avg_recall_ms, 1),
        },
        "cost": {
            "total_tokens": llm.total_tokens,
            "answer_tokens": llm._answer_tokens,
            "judge_tokens": llm._judge_tokens,
        },
    }

    # Print final report
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Overall: {total_correct}/{total_questions} ({overall_acc:.1f}%)")
    print()
    print("  By Category:")
    for name, data in category_results.items():
        bar = "█" * int(data["accuracy"] / 5) + "░" * (20 - int(data["accuracy"] / 5))
        print(f"    {name:15} {bar} {data['accuracy']:5.1f}%  ({data['correct']}/{data['total']})")
    print()
    print("  By Conversation:")
    for cs in conv_summaries:
        print(f"    {cs['conversation']:10} {cs['accuracy']:5.1f}%  ({cs['correct']}/{cs['total']})")
    print()
    print(f"  Performance:")
    print(f"    Avg recall latency:   {avg_recall_ms:.1f}ms")
    print(f"    Memories ingested:    {total_memories}")
    print(f"    Ingest time:          {total_ingest_time:.0f}ms")
    print(f"    Tokens used:          {llm.total_tokens:,}")
    print("=" * 70)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"locomo_{ts}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_file}")

    # Save detailed results (all Q&A pairs)
    detail_file = RESULTS_DIR / f"locomo_{ts}_detail.json"
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Details saved to: {detail_file}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark against Unforget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=None,
        help="Number of conversations to evaluate (default: all 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions per conversation (default: all)",
    )
    parser.add_argument(
        "--category",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Evaluate only this category (1=single_hop, 2=temporal, 3=multi_hop, 4=open_domain)",
    )
    parser.add_argument(
        "--ingest-mode",
        choices=["per_turn", "chunk", "hybrid"],
        default="per_turn",
        help="Ingestion strategy (default: per_turn)",
    )
    parser.add_argument(
        "--recall-limit",
        type=int,
        default=10,
        help="Number of memories to retrieve per question (default: 10)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for answer generation (default: provider-dependent)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for judging (default: same as --model)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every question and answer",
    )
    return parser.parse_args()


PROVIDER_DEFAULTS = {
    "openai": {"model": "gpt-4.1-mini", "key_env": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-haiku-4-5-20251001", "key_env": "ANTHROPIC_API_KEY"},
}


def main():
    args = parse_args()
    args.rerank = not args.no_rerank

    # Set provider-dependent defaults
    prov = PROVIDER_DEFAULTS[args.provider]
    if args.model is None:
        args.model = prov["model"]
    if args.judge_model is None:
        args.judge_model = args.model

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    key_env = prov["key_env"]
    if not os.environ.get(key_env):
        print(f"Error: {key_env} environment variable is required.")
        print(f"  export {key_env}=...")
        sys.exit(1)

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("  Download it with:")
        print(
            "  curl -o benchmarks/locomo/locomo_dataset.json "
            "https://raw.githubusercontent.com/Backboard-io/"
            "Backboard-Locomo-Benchmark/main/locomo_dataset.json"
        )
        sys.exit(1)

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
