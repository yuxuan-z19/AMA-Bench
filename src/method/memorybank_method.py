"""
MemoryBank method adapter for AMA-Bench.

This adapter follows the MemoryBank source design:
1. store timestamped raw memories,
2. summarize chunks into event memories,
3. retrieve with dense vectors,
4. track retention strength and reinforce recalled memories.

The original MemoryBank repo is chatbot-oriented and depends on older
langchain/llama-index/OpenAI APIs.  This implementation keeps the algorithmic
contract behind AMA-Bench's BaseMethod interface.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.method.base_method import BaseMethod


@dataclass
class MemoryBankRecord:
    memory_id: str
    kind: str
    content: str
    created_tick: float
    last_recall_tick: float
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def retention(self, current_tick: float, time_scale: float) -> float:
        elapsed = max(0.0, current_tick - self.last_recall_tick)
        denominator = max(time_scale * max(self.strength, 1e-6), 1e-6)
        return math.exp(-elapsed / denominator)

    def reinforce(self, current_tick: float, increment: float) -> None:
        self.strength += increment
        self.last_recall_tick = current_tick


@dataclass
class MemoryBankMemory:
    task: str
    records: List[MemoryBankRecord]
    embeddings: np.ndarray
    index: Any
    current_tick: float
    global_summary: str
    config: Dict[str, Any]


class MemoryBankMethod(BaseMethod):
    """
    MemoryBank-backed memory method.

    The method constructs a per-episode memory bank from AMA-Bench trajectories.
    Retrieval returns compact MemoryBank context; final answering is still
    handled by MemoryQAInterface using the configured LLM.
    """

    def __init__(
        self,
        config_path: str = None,
        client: Any = None,
        embedding_engine: Any = None,
    ):
        config = self._load_config(config_path) if config_path else {}
        self.config = config
        self.client = client
        self.embedding_engine = embedding_engine

        self.top_k = int(config.get("top_k", 10))
        self.candidate_k = int(config.get("candidate_k", max(self.top_k * 4, self.top_k)))
        self.use_faiss = bool(config.get("use_faiss", True)) and FAISS_AVAILABLE
        self.embedding_model_name = str(
            config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.chunk_size_steps = max(1, int(config.get("chunk_size_steps", 20)))
        self.max_step_chars = int(config.get("max_step_chars", 4000))
        self.max_chunk_chars = int(config.get("max_chunk_chars", 14000))
        self.max_record_chars = int(config.get("max_record_chars", 1800))
        self.max_context_chars = int(config.get("max_context_chars", 18000))
        self.include_task = bool(config.get("include_task", True))
        self.include_raw_steps = bool(config.get("include_raw_steps", True))
        self.include_event_summaries = bool(config.get("include_event_summaries", True))
        self.include_global_summary = bool(config.get("include_global_summary", True))
        self.summary_enabled = bool(config.get("summary_enabled", True))
        self.global_summary_enabled = bool(config.get("global_summary_enabled", True))
        self.summary_max_tokens = int(config.get("summary_max_tokens", 512))
        self.summary_max_retries = int(config.get("summary_max_retries", 2))
        self.summary_temperature = float(config.get("summary_temperature", 0.0))

        forgetting_config = config.get("forgetting", {}) or {}
        self.forgetting_enabled = bool(forgetting_config.get("enabled", True))
        self.forgetting_time_scale = float(forgetting_config.get("time_scale", 400.0))
        self.retention_threshold = float(forgetting_config.get("retention_threshold", 0.0))
        self.apply_retention_to_ranking = bool(
            forgetting_config.get("apply_retention_to_ranking", True)
        )
        self.retention_score_power = float(forgetting_config.get("score_power", 0.15))
        self.reinforce_increment = float(forgetting_config.get("reinforce_increment", 1.0))

        self._tokenizer = None
        self._model = None
        self._device = None
        if self.embedding_engine is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "MemoryBankMethod requires either embedding_engine in the method config "
                    "or local transformers/torch for embeddings."
                )
            self._init_local_embedding_model()

    def _init_local_embedding_model(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self._model = AutoModel.from_pretrained(self.embedding_model_name)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.embedding_engine is not None:
            embeddings = self.embedding_engine.encode(list(texts))
            return np.asarray(embeddings, dtype=np.float32)

        batches = []
        batch_size = int(self.config.get("local_embedding_batch_size", 8))
        max_length = int(self.config.get("local_embedding_max_length", 512))
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            batches.append((summed / counts).cpu().numpy())
        return np.vstack(batches).astype(np.float32)

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)

    def _parse_steps(self, traj_text: str) -> List[Tuple[int, str]]:
        matches = list(re.finditer(r"(?m)^Step\s+(\d+):\s*$", traj_text))
        if not matches:
            chunks = []
            chunk_chars = max(self.max_chunk_chars, 1000)
            for index, start in enumerate(range(0, len(traj_text), chunk_chars)):
                text = traj_text[start : start + chunk_chars].strip()
                if text:
                    chunks.append((index, text))
            return chunks

        steps = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(traj_text)
            step_id = int(match.group(1))
            block = traj_text[start:end].strip()
            if self.max_step_chars and len(block) > self.max_step_chars:
                block = block[: self.max_step_chars] + "\n...[step truncated]..."
            steps.append((step_id, block))
        return steps

    def _chunk_steps(self, steps: List[Tuple[int, str]]) -> List[List[Tuple[int, str]]]:
        chunks = []
        current = []
        current_chars = 0
        for step_id, text in steps:
            text_len = len(text)
            if (
                current
                and (
                    len(current) >= self.chunk_size_steps
                    or current_chars + text_len > self.max_chunk_chars
                )
            ):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append((step_id, text))
            current_chars += text_len
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _compact(text: str, max_chars: int) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return text[:head].rstrip() + "\n...[truncated]...\n" + text[-tail:].lstrip()

    def _summarize_with_llm(self, prompt: str) -> Optional[str]:
        if self.client is None:
            return None
        for attempt in range(self.summary_max_retries):
            try:
                summary = self.client.query(
                    prompt,
                    temperature=self.summary_temperature,
                    max_tokens=self.summary_max_tokens,
                )
                summary = summary.strip()
                if summary:
                    return summary
            except Exception as exc:
                if attempt + 1 >= self.summary_max_retries:
                    print(f"MemoryBank summary LLM failed: {exc}")
        return None

    def _event_summary(self, chunk: List[Tuple[int, str]], task: str) -> str:
        start_step = chunk[0][0]
        end_step = chunk[-1][0]
        chunk_text = "\n\n".join(text for _, text in chunk)
        chunk_text = self._compact(chunk_text, self.max_chunk_chars)
        if self.summary_enabled:
            prompt = (
                "Summarize this agent trajectory segment as MemoryBank event memory.\n"
                "Keep step numbers, actions, observations, state changes, failures, "
                "loops, causal relations, object names, and final progress. "
                "Write concise factual bullets only.\n\n"
                f"Task:\n{task}\n\n"
                f"Trajectory segment: steps {start_step}-{end_step}\n{chunk_text}\n\n"
                "MemoryBank event summary:"
            )
            summary = self._summarize_with_llm(prompt)
            if summary:
                return self._compact(summary, self.max_record_chars)

        return self._compact(
            f"Steps {start_step}-{end_step} extractive event memory:\n{chunk_text}",
            self.max_record_chars,
        )

    def _global_summary(self, event_summaries: List[str], task: str) -> str:
        if not event_summaries:
            return ""
        joined = "\n\n".join(
            f"Event {idx + 1}: {summary}" for idx, summary in enumerate(event_summaries)
        )
        joined = self._compact(joined, self.max_chunk_chars)
        if self.global_summary_enabled and self.summary_enabled:
            prompt = (
                "Create a compact global MemoryBank summary of the whole trajectory. "
                "Preserve facts useful for later QA: task progress, important actions, "
                "state transitions, repeated loops, failures, causal links, and final state.\n\n"
                f"Task:\n{task}\n\n"
                f"Event memories:\n{joined}\n\n"
                "Global MemoryBank summary:"
            )
            summary = self._summarize_with_llm(prompt)
            if summary:
                return self._compact(summary, self.max_record_chars * 2)
        return self._compact(joined, self.max_record_chars * 2)

    def _build_records(self, steps: List[Tuple[int, str]], task: str) -> Tuple[List[MemoryBankRecord], str]:
        records: List[MemoryBankRecord] = []
        if self.include_task and task:
            records.append(
                MemoryBankRecord(
                    memory_id="task",
                    kind="task",
                    content=f"Task description:\n{task}",
                    created_tick=0.0,
                    last_recall_tick=0.0,
                )
            )

        if self.include_raw_steps:
            for order, (step_id, text) in enumerate(steps):
                records.append(
                    MemoryBankRecord(
                        memory_id=f"step_{step_id}",
                        kind="raw_step",
                        content=self._compact(text, self.max_record_chars),
                        created_tick=float(step_id),
                        last_recall_tick=float(step_id),
                        metadata={"step_id": step_id, "order": order},
                    )
                )

        event_summaries = []
        if self.include_event_summaries:
            for chunk_index, chunk in enumerate(self._chunk_steps(steps)):
                summary = self._event_summary(chunk, task)
                event_summaries.append(summary)
                start_step = chunk[0][0]
                end_step = chunk[-1][0]
                created_tick = float(end_step)
                records.append(
                    MemoryBankRecord(
                        memory_id=f"event_{chunk_index}_steps_{start_step}_{end_step}",
                        kind="event_summary",
                        content=summary,
                        created_tick=created_tick,
                        last_recall_tick=created_tick,
                        metadata={
                            "chunk_index": chunk_index,
                            "start_step": start_step,
                            "end_step": end_step,
                        },
                    )
                )

        global_summary = self._global_summary(event_summaries, task)
        if self.include_global_summary and global_summary:
            current_tick = float(steps[-1][0] + 1 if steps else 1)
            records.append(
                MemoryBankRecord(
                    memory_id="global_summary",
                    kind="global_summary",
                    content=global_summary,
                    created_tick=current_tick,
                    last_recall_tick=current_tick,
                )
            )
        return records, global_summary

    def _build_index(self, embeddings: np.ndarray) -> Any:
        if embeddings.size == 0 or not self.use_faiss:
            return None
        normalized = self._l2_normalize(embeddings.copy()).astype(np.float32)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        return index

    def memory_construction(self, traj_text: str, task: str = "") -> MemoryBankMemory:
        steps = self._parse_steps(traj_text)
        current_tick = float((steps[-1][0] + 1) if steps else 1)
        records, global_summary = self._build_records(steps, task)
        searchable_texts = [record.content for record in records]
        embeddings = self._encode_texts(searchable_texts)
        index = self._build_index(embeddings)
        return MemoryBankMemory(
            task=task,
            records=records,
            embeddings=embeddings,
            index=index,
            current_tick=current_tick,
            global_summary=global_summary,
            config=self.config,
        )

    def _similarity_scores(self, memory: MemoryBankMemory, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        if memory.embeddings.size == 0:
            return []
        if memory.index is not None:
            query = self._l2_normalize(query_embedding.copy()).astype(np.float32)
            k = min(max(self.candidate_k, self.top_k), len(memory.records))
            scores, indices = memory.index.search(query, k)
            return [
                (int(idx), float(score))
                for idx, score in zip(indices[0].tolist(), scores[0].tolist())
                if idx >= 0
            ]

        doc_norms = self._l2_normalize(memory.embeddings)
        query_norm = self._l2_normalize(query_embedding)
        similarities = np.dot(doc_norms, query_norm.T).flatten()
        ranked = np.argsort(similarities)[::-1]
        return [(int(idx), float(similarities[idx])) for idx in ranked[: self.candidate_k]]

    def _rank_records(
        self,
        memory: MemoryBankMemory,
        question: str,
    ) -> List[Tuple[MemoryBankRecord, float, float]]:
        query = question
        if self.config.get("include_task_in_retrieval", True) and memory.task:
            query = f"Task: {memory.task}\nQuestion: {question}"
        query_embedding = self._encode_texts([query])
        scored = []
        for index, similarity in self._similarity_scores(memory, query_embedding):
            record = memory.records[index]
            retention = (
                record.retention(memory.current_tick, self.forgetting_time_scale)
                if self.forgetting_enabled
                else 1.0
            )
            if retention < self.retention_threshold:
                continue
            rank_score = similarity
            if self.apply_retention_to_ranking:
                rank_score = similarity * (retention ** self.retention_score_power)
            scored.append((record, rank_score, retention))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.top_k]

    def memory_retrieve(self, memory: MemoryBankMemory, question: str) -> str:
        if not isinstance(memory, MemoryBankMemory):
            raise ValueError("Memory must be a MemoryBankMemory object")

        ranked = self._rank_records(memory, question)
        for record, _, _ in ranked:
            record.reinforce(memory.current_tick, self.reinforce_increment)

        sections = []
        if self.include_task and memory.task:
            sections.append(f"## Task Description\n{memory.task}")
        sections.append(
            "## MemoryBank Context\n"
            "The following memories were stored as raw trajectory memories and "
            "event summaries, retrieved by semantic similarity and MemoryBank "
            "retention metadata."
        )
        if self.include_global_summary and memory.global_summary:
            sections.append(f"### Global Summary\n{memory.global_summary}")

        if ranked:
            lines = []
            for rank, (record, score, retention) in enumerate(ranked, 1):
                lines.append(
                    f"[Memory {rank} | id={record.memory_id} | kind={record.kind} | "
                    f"score={score:.4f} | retention={retention:.4f} | "
                    f"strength={record.strength:.1f}]\n{record.content}"
                )
            sections.append("### Retrieved Memories\n" + "\n\n".join(lines))
        else:
            sections.append("### Retrieved Memories\nNo relevant MemoryBank memories were retrieved.")

        context = "\n\n".join(sections)
        return self._compact(context, self.max_context_chars)
