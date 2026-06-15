"""
HippoRAG Method

HippoRAG 只能通过 OPENAI_API_KEY 环境变量访问 API，因此运行前需要先设置环境变量。
"""

import os
import time
from dataclasses import dataclass
from typing import List, override

from .base import BaseMethod, BaseConfig, BaseMemory


@dataclass
class HippoRAGConfig(BaseConfig):
    """Configuration for HippoRAG method"""

    llm_model_name: str = "Qwen/Qwen3-32B"
    llm_base_url: str = "http://166.111.238.55:11800/v1"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B"
    embedding_base_url: str = "http://166.111.238.55:11801/v1"
    api_key: str = "sk-KUP5iUXThgpibU5fznCsSYRj5n3DHH"
    embedding_dim: int = 2560
    save_dir: str = "outputs/hipporag_ama_bench"
    retrieval_top_k: int = 5
    qa_top_k: int = 5
    max_context_chars: int = 24000  # Truncate to fit 40960 token limit 
    chunk_size: int = 2048
    chunk_overlap: int = 128


@dataclass
class HippoRAGMemory(BaseMemory):
    """Memory object for HippoRAG method"""

    hipporag_instance: object
    doc_ids: List[int]
    save_dir: str 

    def cleanup(self):
        hr = self.hipporag_instance
        if hr is None:
            return

        for attr in [
            "chunk_embedding_store", "entity_embedding_store",
            "fact_embedding_store", "graph", "embedding_model",
            "llm_model", "openie", "rerank_filter",
        ]:
            try:
                setattr(hr, attr, None)
            except Exception:
                pass
        self.hipporag_instance = None


class HippoRAGMethod(BaseMethod):
    """
    HippoRAG-based memory method.

    """

    def __init__(
        self,
        config_path: os.PathLike = None,
        client=None,
        embedding_engine=None,
    ):
        super().__init__(
            config_path=config_path,
            client=client,
            embedding_engine=embedding_engine,
        )
        self.config = self._parse_config()

        # Patch HippoRAG's embedding model factory to support custom OpenAI-compatible models
        self._patch_embedding_factory()

        # Patch HippoRAG's OpenIE parsers to handle LLMs (like Qwen3) that return
        # object-style triples instead of the expected list-style triples.
        self._patch_openie_parsers()

        self._episode_counter = 0

    def _patch_embedding_factory(self):
        """
        Monkey-patch HippoRAG's embedding model factory so that
        Qwen and other custom OpenAI-compatible embedding models work.

        """
        try:
            import hipporag.embedding_model as emb_package
            from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
            from hipporag import HippoRAG

            original_fn = emb_package._get_embedding_model_class
            custom_base_url = self.config.embedding_base_url

            def patched_get_embedding_model_class(
                embedding_model_name: str = "nvidia/NV-Embed-v2",
            ):
                try:
                    return original_fn(embedding_model_name)
                except AssertionError:
                    if custom_base_url:
                        print(
                            f"  ℹ️ Embedding model '{embedding_model_name}' not in "
                            f"HippoRAG defaults, falling back to OpenAI-compatible endpoint."
                        )
                        return OpenAIEmbeddingModel
                    raise

            # Patch the original definition site (for anyone importing from the package)
            emb_package._get_embedding_model_class = patched_get_embedding_model_class

            # Critical: patch HippoRAG.__init__.__globals__ directly.
            # LOAD_GLOBAL resolves names from this dict, NOT from the module's __dict__ proxy.
            HippoRAG.__init__.__globals__["_get_embedding_model_class"] = patched_get_embedding_model_class

            # --- Also patch OpenAIEmbeddingModel.encode to add retry logic ---
            original_encode = OpenAIEmbeddingModel.encode

            def patched_encode(self_emb, texts, max_retries=5):
                import time as _time
                texts = [t.replace("\n", " ") for t in texts]
                texts = [t if t != '' else ' ' for t in texts]

                for attempt in range(max_retries):
                    try:
                        response = self_emb.client.embeddings.create(
                            input=texts, model=self_emb.embedding_model_name
                        )
                        import numpy as _np
                        return _np.array([v.embedding for v in response.data])
                    except Exception as e:
                        error_str = str(e)
                        is_transient = any(
                            code in error_str
                            for code in ["502", "503", "429", "500", "timeout"]
                        )
                        if is_transient and attempt < max_retries - 1:
                            wait = min(2 ** attempt + 1, 30)
                            print(
                                f"  ⚠️ Embedding call failed, retrying in {wait}s... "
                                f"({attempt+1}/{max_retries})"
                            )
                            _time.sleep(wait)
                        else:
                            raise

            OpenAIEmbeddingModel.encode = patched_encode

            # --- Patch batch_encode to remove the ipdb debugger trap ---
            original_batch_encode = OpenAIEmbeddingModel.batch_encode

            def patched_batch_encode(self_emb, texts, **kwargs):
                import numpy as _np
                from copy import deepcopy as _deepcopy
                if isinstance(texts, str):
                    texts = [texts]

                params = _deepcopy(self_emb.embedding_config.encode_params)
                if kwargs:
                    params.update(kwargs)

                if "instruction" in kwargs:
                    if kwargs["instruction"] != '':
                        params["instruction"] = (
                            f"Instruct: {kwargs['instruction']}\nQuery: "
                        )

                batch_size = params.pop("batch_size", 16)

                if len(texts) <= batch_size:
                    results = self_emb.encode(texts)
                else:
                    from tqdm import tqdm as _tqdm
                    pbar = _tqdm(total=len(texts), desc="Batch Encoding")
                    results = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        results.append(self_emb.encode(batch))
                        pbar.update(batch_size)
                    pbar.close()
                    results = _np.concatenate(results)

                import torch as _torch
                if isinstance(results, _torch.Tensor):
                    results = results.cpu()
                    results = results.numpy()
                if self_emb.embedding_config.norm:
                    results = (results.T / _np.linalg.norm(results, axis=1)).T

                return results

            OpenAIEmbeddingModel.batch_encode = patched_batch_encode
        
        except ImportError:
            print("  ⚠️ HippoRAG not installed. Install with: pip install hipporag")
            raise


    def _patch_openie_parsers(self):
        """
        Monkey-patch HippoRAG's NER and triple-extraction parsers.

        Problem: HippoRAG's original regex-based parsers expect compact JSON:
            {"triples": [["subj", "rel", "obj"], ...]}           ← list-of-lists
            {"named_entities": ["entity1", "entity2", ...]}      ← list-of-strings

        But some LLMs (e.g. Qwen3) return object-style JSON:
            {"triples": [{"subject": "s", "relation": "r", "object": "o"}, ...]}

        The original regex ``\\[[^\\]]*\\]`` fails on nested ``{}`` inside the array,
        causing 'NoneType' object has no attribute 'group' for every chunk.

        Fix: Replace the parsers with versions that try the original regex first,
        then fall back to full JSON parsing with format normalization.
        """
        import json
        import re

        # --- Patch NER parser (module-level function) ---
        def patched_extract_ner(real_response):
            # Try original regex first (handles compact format)
            pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is not None:
                try:
                    raw = json.loads(match.group())["named_entities"]
                except (json.JSONDecodeError, KeyError):
                    try:
                        raw = eval(match.group())["named_entities"]
                    except Exception:
                        raw = []
                # Filter to only strings (eval can produce Ellipsis from "...")
                return [e for e in raw if isinstance(e, str)]
            # Fallback: full JSON parse
            try:
                raw = json.loads(real_response)["named_entities"]
                return [e for e in raw if isinstance(e, str)]
            except (json.JSONDecodeError, KeyError):
                return []

        # --- Patch triple extraction (defined inside OpenIE.triple_extraction) ---
        def patched_extract_triples(real_response):
            # Try original regex first (handles compact format)
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is not None:
                try:
                    return json.loads(match.group())["triples"]
                except (json.JSONDecodeError, KeyError):
                    try:
                        return eval(match.group())["triples"]
                    except Exception:
                        return []

            # Fallback: full JSON parse, then normalize object-style to list-style
            try:
                data = json.loads(real_response)
                raw_triples = data["triples"]
            except (json.JSONDecodeError, KeyError):
                return []

            # Normalize: if triples are dicts, convert to lists
            normalized = []
            for t in raw_triples:
                if isinstance(t, list):
                    normalized.append(t)
                elif isinstance(t, dict):
                    # {"subject": "s", "relation": "r", "object": "o"} → ["s", "r", "o"]
                    normalized.append([
                        t.get("subject", t.get("head", "")),
                        t.get("relation", t.get("predicate", "")),
                        t.get("object", t.get("tail", "")),
                    ])
                else:
                    continue
            return normalized

        # Apply patches
        try:
            import hipporag.information_extraction.openie_openai as openie_module
            from hipporag.information_extraction.openie_openai import OpenIE

            # Patch the module-level NER function
            openie_module._extract_ner_from_response = patched_extract_ner

            # Patch triple_extraction method: replace the inline function
            # by wrapping the method.
            original_triple_extraction = OpenIE.triple_extraction

            def patched_triple_extraction(self_openie, chunk_key, passage, named_entities):
                # Sanitize named_entities: filter out non-string items (e.g. Ellipsis)
                named_entities = [e for e in named_entities if isinstance(e, str)]
                # by monkey-patching the module's globals used by the original method
                # Actually, since the original uses a nested def, we need to replace
                # the whole method. Let's do that.
                messages = self_openie.prompt_template_manager.render(
                    name='triple_extraction',
                    passage=passage,
                    named_entity_json=json.dumps({"named_entities": named_entities})
                )

                raw_response = ""
                metadata = {}
                try:
                    raw_response, metadata, cache_hit = self_openie.llm_model.infer(
                        messages=messages,
                    )
                    metadata['cache_hit'] = cache_hit
                    from hipporag.utils.llm_utils import fix_broken_generated_json
                    if metadata['finish_reason'] == 'length':
                        real_response = fix_broken_generated_json(raw_response)
                    else:
                        real_response = raw_response
                    extracted_triples = patched_extract_triples(real_response)
                    from hipporag.utils.llm_utils import filter_invalid_triples
                    triplets = filter_invalid_triples(triples=extracted_triples)
                except Exception as e:
                    from hipporag.utils.logging_utils import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"Exception for chunk {chunk_key}: {e}")
                    metadata.update({'error': str(e)})
                    from hipporag.utils.misc_utils import TripleRawOutput
                    return TripleRawOutput(
                        chunk_id=chunk_key,
                        response=raw_response,
                        metadata=metadata,
                        triples=[]
                    )

                from hipporag.utils.misc_utils import TripleRawOutput
                return TripleRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    metadata=metadata,
                    triples=triplets
                )

            OpenIE.triple_extraction = patched_triple_extraction

            print("  ✅ Patched HippoRAG OpenIE parsers for Qwen3 compatibility")

        except ImportError:
            pass  # HippoRAG not installed, skip

    @override
    def _parse_config(self) -> HippoRAGConfig:
        config_dict = self._load_config(self.config_path)
        return HippoRAGConfig(
            llm_model_name=config_dict.get("llm_model_name"),
            llm_base_url=config_dict.get("llm_base_url"),
            embedding_model_name=config_dict.get("embedding_model_name"),
            embedding_base_url=config_dict.get("embedding_base_url"),
            embedding_dim=config_dict.get("embedding_dim"),
            save_dir=config_dict.get("save_dir"),
            retrieval_top_k=config_dict.get("retrieval_top_k"),
            qa_top_k=config_dict.get("qa_top_k"),
            chunk_size=config_dict.get("chunk_size"),
            chunk_overlap=config_dict.get("chunk_overlap"),
        )

    def _split_trajectory_into_docs(self, traj_text: str, task: str = "") -> List[str]:
        """
        Split trajectory text into document chunks suitable for HippoRAG indexing.

        Strategy: split by "Step N:" boundaries, then optionally merge small chunks.

        Args:
            traj_text: String-formatted trajectory text
            task: Task description (prepended as context)

        Returns:
            List of document strings
        """
        # Split trajectory into per-step documents
        steps = []
        lines = traj_text.split("\n")
        current_step = []

        for line in lines:
            if line.strip().startswith("Step "):
                if current_step:
                    steps.append("\n".join(current_step))
                current_step = [line]
            else:
                current_step.append(line)

        if current_step:
            steps.append("\n".join(current_step))

        if not steps:
            steps = [traj_text]

        # Merge small steps into larger chunks to give HippoRAG more context per doc
        max_step_chars = 10000
        docs = []
        current_chunk = []
        current_len = 0

        for step in steps:
            # Truncate oversized steps (e.g. long SQL dumps, HTML observations)
            if len(step) > max_step_chars:
                step = step[:max_step_chars] + "\n...[truncated]"
            step_len = len(step)
            if current_chunk and (current_len + step_len > self.config.chunk_size):
                docs.append("\n".join(current_chunk))
                current_chunk = [step]
                current_len = step_len
            else:
                current_chunk.append(step)
                current_len += step_len

        if current_chunk:
            docs.append("\n".join(current_chunk))

        # Prepend task description to the first doc if provided
        if task and docs:
            docs[0] = f"Task: {task}\n\n{docs[0]}"

        return docs

    def _create_hipporag_instance(self, episode_save_dir: str):
        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig as HippoRAGBaseConfig

        # HippoRAG's OpenAI client reads API key from OPENAI_API_KEY env var.
        # Set it before instantiation so both LLM and embedding clients can use it.
        if self.config.api_key:
            os.environ["OPENAI_API_KEY"] = self.config.api_key

        return HippoRAG(
            save_dir=episode_save_dir,
            llm_model_name=self.config.llm_model_name,
            llm_base_url=self.config.llm_base_url,
            embedding_model_name=self.config.embedding_model_name,
            embedding_base_url=self.config.embedding_base_url,
        )

    @override
    def memory_construction(self, traj_text: str, task: str = "") -> HippoRAGMemory:
        """
        Build HippoRAG knowledge graph from trajectory text.

        """
        ep_label = f"episode_{self._episode_counter}"
        self._episode_counter += 1
        episode_save_dir = os.path.join(self.config.save_dir, ep_label)

        hipporag = self._create_hipporag_instance(episode_save_dir)

        print(
            f"  HippoRAG initialized (episode dir: {episode_save_dir}): "
            f"LLM={self.config.llm_model_name}, Embedding={self.config.embedding_model_name}"
        )

        # Split trajectory into document chunks
        docs = self._split_trajectory_into_docs(traj_text, task)
        doc_ids = list(range(len(docs)))

        print(f"  HippoRAG indexing {len(docs)} chunks...")

        # Index documents into HippoRAG's knowledge graph with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                hipporag.index(docs=docs)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Indexing failed ({e}), retrying in {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"  Indexing failed after {max_retries} attempts: {e}")
                    raise

        print(f"   HippoRAG indexing complete ({len(docs)} chunks)")

        return HippoRAGMemory(
            hipporag_instance=hipporag,
            doc_ids=doc_ids,
            save_dir=episode_save_dir,
        )

    @override
    def memory_retrieve(self, memory: HippoRAGMemory, question: str) -> str:
        """
        Retrieve relevant context using HippoRAG's graph-based retrieval.

        """
        if not isinstance(memory, HippoRAGMemory):
            raise ValueError("Memory must be a HippoRAGMemory object")

        hipporag = memory.hipporag_instance

        try:
            # Use HippoRAG's retrieve method to get relevant passages
            retrieval_results = hipporag.retrieve(
                queries=[question],
                num_to_retrieve=self.config.retrieval_top_k,
            )

            if not retrieval_results or len(retrieval_results) == 0:
                return "No relevant context found."

            # Extract retrieved documents from QuerySolution
            query_solution = retrieval_results[0]
            retrieved_docs = query_solution.docs

            if not retrieved_docs:
                return "No relevant context found."

            # Take top-k docs for context
            top_docs = retrieved_docs[: self.config.qa_top_k]
        
        #####
        except (ValueError, IndexError) as e:
            # Graceful degradation: when the knowledge graph is empty
            # (e.g., OpenIE failed to extract triples), fall back to
            # returning the raw stored documents as context.
            print(f"  HippoRAG retrieval failed ({e}), falling back to raw documents")
            try:
                # Try to get docs from the chunk embedding store
                if hasattr(hipporag, "chunk_embedding_store") and hasattr(
                    hipporag.chunk_embedding_store, "corpus"
                ):
                    top_docs = list(hipporag.chunk_embedding_store.corpus.values())[
                        : self.config.qa_top_k
                    ]
                else:
                    return "No relevant context found."
            except Exception:
                return "No relevant context found."

        # Format retrieved context
        context_parts = []
        total_chars = 0
        for i, doc in enumerate(top_docs):
            part = f"[Retrieved Passage {i + 1}]:\n{doc}"
            if self.config.max_context_chars and total_chars + len(part) > self.config.max_context_chars:
                # Truncate this passage to fit within budget
                remaining = self.config.max_context_chars - total_chars
                if remaining > 200:
                    part = part[:remaining] + "\n...[truncated]"
                    context_parts.append(part)
                break
            context_parts.append(part)
            total_chars += len(part)

        retrieved_context = "\n\n".join(context_parts)

        return retrieved_context
