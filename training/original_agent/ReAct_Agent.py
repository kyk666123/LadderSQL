from __future__ import annotations

import logging
from pathlib import Path
import os
import random
import re
import textwrap
import time
import concurrent.futures
import json
import agentlightning as agl
from typing import Any, Dict, Literal, Optional, cast
from langchain.chat_models import init_chat_model
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from original_agent.prompt import REACT_SQL_PROMPT, REACT_SQL_BIRD_PROMPT, REACT_SQL_SCHEMA_GROUNDING_PROMPT, SUMMARIZE_PROMPT
from reward_func.reward import binary_reward, binary_reward_bird
from reward_func.composite_reward_dynamic import composite_reward_dynamic
from reward_func.composite_reward_dynamic_bird import composite_reward_dynamic_bird
from reward_func.composite_reward_simple import composite_reward_simple


agl.setup_logging(apply_to=[__name__])

logger = logging.getLogger(__name__)


class _TrajectoryLogger:
    """Writes per-rollout ReAct trajectory details to a dedicated JSONL file.

    Each training run creates a timestamped file under ``log_dir``.  Every call
    to :meth:`log` appends a single JSON line describing one rollout, including
    per-turn input/output character lengths, the LLM output, parsed SQL, and
    the observation.  The file is opened/closed on each write so that the
    logger survives pickling across distributed workers.

    If ``mirror_dir`` is set, the same file is also written to a secondary
    location (e.g. a persistent volume) for persistence.
    """

    def __init__(self, log_dir: str = "react_trajectories", mirror_dir: str | None = None) -> None:
        from datetime import datetime

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"trajectory_{timestamp}.jsonl")
        logger.info(f"Trajectory logger initialized \u2192 {self._path}")

        self._mirror_path: str | None = None
        if mirror_dir:
            os.makedirs(mirror_dir, exist_ok=True)
            self._mirror_path = os.path.join(mirror_dir, f"trajectory_{timestamp}.jsonl")
            logger.info(f"Trajectory mirror \u2192 {self._mirror_path}")

    def log(self, record: dict) -> None:  # type: ignore
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
        if self._mirror_path:
            try:
                with open(self._mirror_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as e:
                logger.warning(f"Failed to write trajectory mirror: {e}")


class State(MessagesState):
    question: str
    query: str
    execution: str
    num_turns: int
    messages: list[AnyMessage]
    sql_exec_count: int
    sql_error_count: int


class _PromptTooLongError(Exception):
    """Raised when vLLM rejects a request due to input exceeding max_model_len."""


class Agent:
    """ReAct-style SQL agent: Think → Action(SQL) → Observation(exec result) loop."""

    def __init__(
        self,
        db_path: str,
        db_schema: str,
        max_turns: int = 5,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        execution_truncate: int = 1024,
        prompt_template: Any = None,
        evidence: str = "",
    ):
        self.db = SQLDatabase.from_uri(db_path)  # type: ignore
        # keep the raw sqlite file path, used to connect directly in the timeout mechanism
        self._db_file_path = db_path.replace("sqlite:///", "") if db_path.startswith("sqlite:///") else db_path
        self.db_schema = db_schema
        self.max_turns = max_turns
        self.execution_truncate = execution_truncate
        self.prompt_template = prompt_template or REACT_SQL_PROMPT
        self.evidence = evidence
        # detailed per-turn ReAct trajectory, used for diagnosis on failure
        self._turn_history: list[Dict[str, Any]] = []
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=3,
                max_tokens=2048,
            )

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            error_msg = str(e).lower()
            # unrecoverable errors (prompt too long, malformed, etc.): mark as terminal
            if "maximum context length" in error_msg or "too many tokens" in error_msg or "400" in str(type(e).__name__ + str(e)):
                logger.warning(f"[ReAct] LLM rejected request (likely prompt too long): {type(e).__name__}: {e}")
                raise _PromptTooLongError(str(e)) from e
            # other errors (network errors that still fail after retries, etc.) are re-raised
            raise
        return result  # type: ignore

    def truncate_execution(self, execution: str) -> str:
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: AnyMessage) -> str | None:
        """Extract the last SQL code block from the LLM response."""
        result: str | None = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):  # type: ignore
            result = match.group(1).strip()  # type: ignore
        return result  # type: ignore

    def _is_stop(self, message: AnyMessage) -> bool:
        """Check whether the LLM signalled [STOP] in its Thought."""
        return "[STOP]" in str(message.content)

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def think_and_act(self, state: State) -> State:
        """Call the LLM to produce Thought + Action (SQL).

        On the first turn the full REACT_SQL_PROMPT is used.
        On subsequent turns the previous Observation is appended to the
        conversation history before calling the LLM again.
        """
        t_start = time.time()
        num_turns = state.get("num_turns", 0)  # type: ignore

        if num_turns == 0:
            # First turn: render the system + user prompt
            invoke_args: Dict[str, Any] = {
                "dialect": self.db.dialect,
                "input": state["question"],
                "table_info": self.db_schema,
            }
            # only inject when the prompt template itself contains the {evidence} placeholder (Bird prompt only).
            # the Spider prompt has no such variable, so it is not injected -- the model never sees the evidence field.
            if "evidence" in self.prompt_template.input_variables:
                invoke_args["evidence"] = self.evidence or ""
            prompt: Any = self.prompt_template.invoke(invoke_args)  # type: ignore
            all_messages: list[AnyMessage] = list(prompt.messages)
        else:
            # Subsequent turns: append the Observation from the last execution
            observation_text = (
                f"Observation:\n```\n"
                f"{self.truncate_execution(state['execution'])}\n"
                f"```"
            )
            all_messages = [
                *state["messages"],
                HumanMessage(content=observation_text),
            ]

        # estimate the prompt token count; stop early when close to the limit to avoid an empty response.
        # use a more conservative estimate: chars/2.5 is closer to the real token count (with code/SQL)
        _input_chars = sum(len(str(m.content)) for m in all_messages)
        _estimated_tokens = int(_input_chars // 2.5)
        # prompt token budget: default reuses the old training config (14336-2048=12288); at sampling time
        # it is overridden by REACT_PROMPT_TOKEN_BUDGET (usually = max_model_len - max_tokens, model-dependent)
        _token_budget = int(os.environ.get("REACT_PROMPT_TOKEN_BUDGET", str(14336 - 2048)))
        if _estimated_tokens > _token_budget:
            logger.warning(
                f"[ReAct] Prompt too long (~{_estimated_tokens} tokens) at turn {num_turns + 1}. "
                f"Stopping early to avoid empty response."
            )
            self._turn_history.append({
                "turn": num_turns + 1,
                "input_chars": _input_chars,
                "estimated_tokens": _estimated_tokens,
                "output": "[SKIPPED - prompt too long]",
                "status": "prompt_too_long_pre_check",
            })
            return {  # type: ignore
                **state,
                "query": state.get("query", ""),
                "num_turns": self.max_turns,  # force the termination condition to trigger
                "messages": all_messages,
            }

        _t_llm = time.time()
        try:
            result = self.invoke_prompt(all_messages)  # type: ignore
        except _PromptTooLongError as exc:
            # the LLM rejected the request (prompt over the limit); stop gracefully, keep the current query
            logger.warning(f"[ReAct] Graceful stop at turn {num_turns + 1} due to prompt too long.")
            self._turn_history.append({
                "turn": num_turns + 1,
                "input_chars": _input_chars,
                "estimated_tokens": _estimated_tokens,
                "output": f"[REJECTED - {exc}]",
                "status": "prompt_too_long_rejected",
            })
            return {  # type: ignore
                **state,
                "query": state.get("query", ""),
                "num_turns": self.max_turns,
                "messages": all_messages,
            }

        _llm_dt = time.time() - _t_llm
        # extract thinking content / generation-token stats (record thinking time and length; 0/None for non-thinking models)
        _reasoning = ""
        _usage: Dict[str, Any] = {}
        try:
            _reasoning = (getattr(result, "additional_kwargs", None) or {}).get("reasoning_content") or ""
        except Exception:
            _reasoning = ""
        try:
            _usage = (getattr(result, "response_metadata", None) or {}).get("token_usage", {}) or {}
        except Exception:
            _usage = {}
        _reasoning_tokens = None
        try:
            _reasoning_tokens = (_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        except Exception:
            _reasoning_tokens = None

        # record this turn's trajectory
        output_content = str(result.content) if result else ""
        self._turn_history.append({
            "turn": num_turns + 1,
            "input_chars": _input_chars,
            "estimated_tokens": _estimated_tokens,
            "output_chars": len(output_content),
            "output_preview": output_content[:300],
            "llm_time": round(_llm_dt, 3),
            "gen_tokens": _usage.get("completion_tokens"),
            "prompt_tokens": _usage.get("prompt_tokens"),
            "total_tokens": _usage.get("total_tokens"),
            "reasoning_tokens": _reasoning_tokens,
            "reasoning_chars": len(_reasoning),
            "status": "ok",
        })

        # Parse SQL from the Action block (if any)
        parsed_query = self.parse_query(result)
        query = parsed_query or state.get("query", "") or result.content  # type: ignore

        logger.info(
            f"[NODE TIMING] think_and_act (turn {num_turns + 1}): "
            f"{time.time() - t_start:.3f}s | stop={self._is_stop(result)}"
        )
        return {  # type: ignore
            **state,
            "query": query,
            "num_turns": num_turns + 1,
            "messages": [*all_messages, result],
        }

    def execute_query(self, state: State, _sql_timeout: int = int(os.environ.get("SQL_TIMEOUT", "15"))) -> State:
        """Execute the current SQL query against the database with timeout.

        Uses threading.Timer + sqlite3.interrupt() to reliably interrupt
        long-running SQLite queries (signal.alarm cannot interrupt C-level calls).
        """
        import sqlite3
        import threading

        t_start = time.time()
        query = state["query"]

        try:
            conn = sqlite3.connect(self._db_file_path)
            _timed_out = False

            def _interrupt():
                nonlocal _timed_out
                _timed_out = True
                try:
                    conn.interrupt()
                except Exception:
                    pass

            timer = threading.Timer(_sql_timeout, _interrupt)
            timer.start()
            try:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                timer.cancel()
                # format the result as a string (consistent with LangChain QuerySQLDatabaseTool)
                execution_result = str(rows)
            except sqlite3.OperationalError as e:
                timer.cancel()
                if _timed_out or "interrupted" in str(e).lower():
                    execution_result = f"Error: Query execution timed out after {_sql_timeout}s. The query is too slow, please optimize it or try a different approach."
                    logger.warning(f"[SQL TIMEOUT] Query exceeded {_sql_timeout}s: {query[:100]}...")
                else:
                    execution_result = f"Error: {e}"
            except Exception as e:
                timer.cancel()
                execution_result = f"Error: {e}"
            finally:
                conn.close()
        except Exception as e:
            execution_result = f"Error: Failed to connect to database: {e}"

        if not isinstance(execution_result, str):
            execution_result = str(execution_result)

        # Detect SQL execution errors in the result
        _result_lower = execution_result.lower()
        is_error = any(
            kw in _result_lower
            for kw in ("error", "operationalerror", "no such", "syntax error", "ambiguous", "timed out")
        )

        logger.info(f"[NODE TIMING] execute_query: {time.time()-t_start:.3f}s")
        return {
            **state,
            "execution": execution_result,
            "sql_exec_count": state.get("sql_exec_count", 0) + 1,
            "sql_error_count": state.get("sql_error_count", 0) + (1 if is_error else 0),
        }

    def should_continue(self, state: State) -> Literal["execute_query", "__end__"]:
        """Decide whether to continue the ReAct loop.

        Stops when:
        - The LLM included [STOP] in its latest response, OR
        - The maximum number of turns has been reached.
        """
        last_msg = state["messages"][-1] if state.get("messages") else None
        if last_msg is not None and self._is_stop(last_msg):
            logger.info("[ReAct] Agent signalled [STOP]. Ending loop.")
            return END  # type: ignore

        if state.get("num_turns", 0) >= self.max_turns:
            logger.info(f"[ReAct] Reached max_turns={self.max_turns}. Ending loop.")
            return END  # type: ignore

        return "execute_query"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def graph(self) -> CompiledStateGraph[State]:
        """Build the ReAct loop graph.

        Flow:
            START → think_and_act → should_continue
            should_continue ──(continue)──▶ execute_query → think_and_act  (loop)
            should_continue ──(stop)────▶ END
        """
        builder = StateGraph(State)
        builder.add_node(self.think_and_act)  # type: ignore
        builder.add_node(self.execute_query)  # type: ignore

        builder.add_edge(START, "think_and_act")
        builder.add_conditional_edges(
            "think_and_act",
            self.should_continue,  # type: ignore
        )
        builder.add_edge("execute_query", "think_and_act")

        return builder.compile()  # type: ignore


class LitAgent(agl.LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        is_train: bool,
        max_turns: int,
        truncate: str,
        trained_agents: Optional[str] = None,
        db_schema_truncate: int = 2048,
        execution_truncate: int = 1024,
        train_temperature: float = 1.1,
        val_temperature: float = 0.0,
        reward_mode: str = "composite",
        val_concurrent: int = 1,
        dataset: str = "spider",
        mode: str = "test",
        llm_endpoints: list[str] | None = None,
        debug: bool = False,
        traj_mirror_dir: str | None = None,
        eval_mode: str = "best_of_n",
        schema_cache_path: str | None = None,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.max_turns = max_turns
        self.truncate = truncate
        self.spider_dir = os.environ.get("SPIDER_DATA_DIR", "data")
        self.execution_truncate = execution_truncate
        self.val_temperature = val_temperature
        self.val_concurrent = val_concurrent
        self.reward_mode = reward_mode
        self.dataset = dataset  # "spider" or "bird"
        self.mode = mode  # used to distinguish train/test/dev etc.
        self.llm_endpoints = llm_endpoints  # list of endpoints for load-balancing across multiple vLLM instances
        self.debug = debug
        self.eval_mode = eval_mode  # "best_of_n" or "pass_at_n"

        # schema_filter is only used in new_llm / bilink / llm / pipeline modes.
        # golden / truncate / full / cache modes do not need it, avoiding a hard dependency on ALICLOUD_BASE_URL
        _needs_alicloud = truncate in ("new_llm", "bilink", "llm", "pipeline")
        if _needs_alicloud:
            self.schema_filter = init_chat_model(
                "glm-5.1",
                model_provider="openai",
                openai_api_base=os.environ["ALICLOUD_BASE_URL"],
                openai_api_key=os.environ["ALICLOUD_API_KEY"],
                temperature=val_temperature,
                max_retries=3,
                max_tokens=4096,
                timeout=240,
            )
        else:
            self.schema_filter = None  # type: ignore

        # three-step Schema Linking Pipeline (pipeline mode only)
        if truncate == "pipeline":
            cache_dir = Path(__file__).parent / "db_schema_cache"
            # determine the DB directory from dataset and mode
            if self.dataset == "bird":
                db_dir = "/root/local_bird_db/dev_databases"
            elif mode in ("test", "test_dev_500"):
                db_dir = os.path.join(os.environ.get("SPIDER_DATA_DIR", "/path/to/dataset/spider"), "test_database")
            else:
                db_dir = os.path.join(os.environ.get("SPIDER_DATA_DIR", "/path/to/dataset/spider"), "database")
            self.schema_linking_pipeline = SchemaLinkingPipeline(
                db_full_schema_path=str(cache_dir / "db_full_schema.json"),
                semantic_similarity_path=str(cache_dir / "semantic_similarity.json"),
                db_dir=db_dir,
            )
            logger.info(f"SchemaLinkingPipeline initialized (db_dir={db_dir})")

        # load the resources needed for enhanced schema linking (new_llm mode: FK connections only)
        cache_dir = Path(__file__).parent / "db_schema_cache"
        full_schema_path = cache_dir / "db_full_schema.json"
        if full_schema_path.exists():
            with open(full_schema_path, encoding="utf-8") as f:
                self._full_schema: dict = json.load(f)
        else:
            self._full_schema = {}

        # load the schema linking cache (switch by dataset)
        if schema_cache_path:
            self.schema_linking_cache_path = schema_cache_path
        elif self.dataset == "bird":
            _cache_filename = "bird_dev_schema_cache_from_rollouts_no_desc.json"
            self.schema_linking_cache_path = str(
                Path(__file__).parent.parent.parent / "schema_preprocessing" / "relevant_schema_cache" / _cache_filename
            )
        else:
            _cache_filename = "spider_glm5_test_schema_cache_20260415.json"
            self.schema_linking_cache_path = str(
                Path(__file__).parent.parent.parent / "schema_preprocessing" / "relevant_schema_cache" / _cache_filename
            )
        self.schema_linking_cache: dict[str, dict] = {}
        if Path(self.schema_linking_cache_path).exists():
            with open(self.schema_linking_cache_path, "r") as f:
                self.schema_linking_cache = json.load(f)
            logger.info(f"Loaded {len(self.schema_linking_cache)} cached schemas from {self.schema_linking_cache_path}")
        
        # load light schema on demand (only used in new_llm / bilink modes)
        self.light_schema: dict = {}
        if self.truncate in ("new_llm", "bilink"):
            _base_dir = Path(__file__).resolve().parent.parent.parent / "schema_preprocessing" / "db_light_schema"
            if self.dataset == "bird":
                light_schema_path = str(_base_dir / "bird_dev_light_schema.json")
            else:
                light_schema_path = str(_base_dir / "spider_test_light_schema.json")
            with open(light_schema_path, "r") as f:
                self.light_schema = json.load(f)
            logger.info(f"Loaded light schema for dataset '{self.dataset}' from {light_schema_path}")

        # summarizer_llm is only used in best_of_n mode + val_concurrent > 1 (summarising parallel inference)
        if val_concurrent > 1 and eval_mode == "best_of_n":
            self.summarizer_llm = init_chat_model(
                "glm-5",
                model_provider="openai",
                openai_api_base=os.environ["ALICLOUD_BASE_URL"],
                openai_api_key=os.environ["ALICLOUD_API_KEY"],
                temperature=0.0,
                max_retries=1,
                max_tokens=4096,
            )
        else:
            self.summarizer_llm = None  # type: ignore

        # Trajectory logger — writes to a separate file, never conflicts with main logs
        self._traj_logger = _TrajectoryLogger(
            log_dir=os.path.join(os.path.dirname(__file__), "react_trajectories"),
            mirror_dir=traj_mirror_dir,
        )

    def _build_enhanced_schema(self, db_id: str, light_schema_text: str) -> str:
        """Build the enhanced schema: original light_schema + FK connections (with bridge-table annotations)."""
        parts = [light_schema_text]

        # inject the connections
        join_text = self._build_join_relationships(db_id)
        if join_text:
            parts.append(join_text)

        return "\n\n".join(parts)

    def _build_join_relationships(self, db_id: str) -> str:
        """Extract FK connections from full_schema."""
        from collections import defaultdict

        db_info = self._full_schema.get(db_id)
        if not db_info:
            return ""

        tables = db_info.get("tables", {})
        fk_edges = []
        for tname, tinfo in tables.items():
            for cname, cinfo in tinfo.get("columns", {}).items():
                ki = cinfo.get("key_info", "")
                if "FOREIGN KEY ->" in ki:
                    m = re.search(r"FOREIGN KEY\s*->\s*(\w+)\.(\w+)", ki)
                    if m:
                        fk_edges.append((tname, cname, m.group(1), m.group(2)))

        if not fk_edges:
            return ""

        lines = ["[Join Relationships]"]
        lines.append("Direct FK connections:")
        for from_t, from_c, to_t, to_c in fk_edges:
            lines.append(f"- {from_t}.{from_c} -> {to_t}.{to_c}")

        # detect bridge tables (tables with 2 or more outgoing FKs)
        outgoing_fks: dict[str, list] = defaultdict(list)
        for from_t, from_c, to_t, to_c in fk_edges:
            outgoing_fks[from_t].append((from_c, to_t, to_c))

        bridge_tables = [(t, fks) for t, fks in outgoing_fks.items() if len(fks) >= 2]
        if bridge_tables:
            lines.append("")
            lines.append("Bridge/Junction tables (connect multiple entities):")
            for bridge_t, fks in bridge_tables:
                connected = [to_t for _, to_t, _ in fks]
                lines.append(f"- {bridge_t} connects: {' <-> '.join(connected)}")

        return "\n".join(lines)

    def get_table_info(self, is_extend: bool, mode: str, db_path: str, rollout_db_id: str, db_schema: str, rollout_question: str, light_schema: dict) -> str: # type: ignore
        if self.truncate == "pipeline":
            return self.schema_linking_pipeline.run(db_id=rollout_db_id, question=rollout_question)

        if self.truncate == "cache":
            # look up the cached schema directly by question_key (format: {"db_id|||question": "schema_string"})
            question_key = f"{rollout_db_id}|||{rollout_question}"
            if question_key in self.schema_linking_cache:
                logger.debug(f"Using cached schema for question_key: {question_key}")
                return self.schema_linking_cache[question_key]
            logger.warning(f"No cached schema found for question_key: {question_key}")
            return ""

        if self.truncate == "new_llm":
            light_schema_text = light_schema[rollout_db_id]

            # pure light_schema control experiment (no extra information added)
            # enhanced_schema = self._build_enhanced_schema(rollout_db_id, light_schema_text)

            llm_prompt = REACT_SQL_SCHEMA_GROUNDING_PROMPT.invoke(
                {"question": rollout_question, "schema": light_schema_text}
            )
            result = self.schema_filter.invoke(llm_prompt).content

            pattern = r"<schema>(.*?)</schema>"
    
            match = re.search(pattern, result, re.DOTALL)
            
            if match:
                raw_schema = match.group(1).strip()
            else:
                logger.warning(f"no <schema> tag found in the output, db={rollout_db_id}, falling back to light_schema")
                raw_schema = light_schema_text

            prefix = "Relevant tables, columns retrieved by llm:"
            llm_linked_schema = f"{prefix}\n{raw_schema}"

            return llm_linked_schema
        
        if self.truncate == "bilink":
            light_schema = light_schema[rollout_db_id]

            llm_prompt = NEW_LIGHT_SCHEMA_GROUNDING_PROMPT.invoke(
                {"question": rollout_question, "schema": light_schema}
            )
            result = self.schema_filter.invoke(llm_prompt).content

            pattern = r"<schema>(.*?)</schema>"
    
            match = re.search(pattern, result, re.DOTALL)
            
            if match:
                print("found <schema> tag, returning its content")
                # group(1) captures the content inside the parentheses
                raw_schema = match.group(1).strip()
                # .strip() removes leading/trailing whitespace (e.g. newlines)
                # return raw_schema.strip()
            else:
                # if no tag is found, choose to return an empty string or raise an error depending on the need
                print("warning: no <schema> tag found in the output")
                return ""

            prefix = "Relevant tables, columns retrieved by llm:"
            llm_linked_schema = f"{prefix}\n{raw_schema}"
            
            vector_prompt = KEYWORD_EXTRACTOR_PROMPT.invoke(
                {"question": rollout_question}
            )
            result = self.schema_filter.invoke(vector_prompt).content
            
            def parse_keywords(raw_output: str):
                text = raw_output.strip()
                
                is_empty_result = (
                    not text or 
                    text.upper() == "NONE" or 
                    text.lower() == "no keywords" or
                    text.lower() == "null"
                )
                
                if is_empty_result:
                    return None
                
                text = text.replace('"', '').replace("'", '')
                
                parts = text.split(',')
                
                cleaned_keywords = []
                for part in parts:
                    k = part.strip()
                    if k:
                        cleaned_keywords.append(k)
                
                if not cleaned_keywords:
                    return None
                    
                return cleaned_keywords

            keywords = parse_keywords(result)
            
            if keywords != None:
                print("keywords extracted")
                retriever = DatabaseCellRetrieval(database_literals=keywords, search_client="/path/to/LadderSQL/examples/my_spider/agent/chroma/spider_test/spider_test", collection_name=rollout_db_id)
                result = retriever.retrieve()
                
                dict_strings = [str(item) for item in result]

                joined_content = "\n".join(dict_strings)

                prefix = "Relevant tables, columns and values retrieved by vector search:"
                vector_linked_schema = f"{prefix}\n{joined_content}"
                return f"{llm_linked_schema}\n{vector_linked_schema}"
                
                # rerank_prompt = LIGHT_SCHEMA_RERANK_PROMPT.invoke(
                #     {"question": rollout_question, "schema": f"{llm_linked_schema}\n{vector_linked_schema}"}
                # )
                # result = self.schema_filter.invoke(rerank_prompt).content
                # return result
            else:
                print("no keywords extracted")
                return llm_linked_schema
                
        if self.truncate == "golden":
            pruned_schema_map: dict[str, str] = {}
            
            if self.dataset == "bird" and mode == "train":
                # BIRD train uses the rule-based gold schema
                grounded_schema_file = Path(__file__).resolve().parent.parent.parent / "schema_preprocessing" / "gold_schema_rule_based" / "bird_train_schema.json"
            elif self.dataset == "bird" and mode != "train":
                # BIRD val uses the gold schema of dev_500
                grounded_schema_file = Path(__file__).resolve().parent.parent.parent / "schema_preprocessing" / "gold_schema_rule_based" / "bird_dev_500_schema.json"
            elif mode == "train" and not is_extend:
                # Spider training-set golden schema
                grounded_schema_file = Path(__file__).resolve().parent.parent.parent / "schema_preprocessing" / "gold_schema_filtered_by_llm" / "spider_train_schema.json"
            elif mode != "train" and not is_extend:
                # Spider eval golden schema (dev_500)
                grounded_schema_file = Path(__file__).resolve().parent.parent.parent / "schema_preprocessing" / "gold_schema_filtered_by_llm" / "spider_test_dev_500_schema.json"
            elif is_extend:
                grounded_schema_file = Path("/path/to/LadderSQL/examples/my_spider/agent/filtered_schema_by_llm") / "filtered_data_4000_ordered_grounded.json"
                
            def _make_key(db_id: str, question: str) -> str:
                # strip leading/trailing spaces and normalise newlines, to avoid key mismatch due to formatting differences
                normalized_question = " ".join(question.strip().split())
                return f"{db_id}||{normalized_question}"
                
            if grounded_schema_file.exists():
                with open(grounded_schema_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    db_id = item["db_id"]
                    question_text = item["question"]
                    key = _make_key(db_id, question_text)
                    pruned_schema_map[key] = item["pruned_schema"]
                logger.info(f"Loaded {len(pruned_schema_map)} pruned schemas from {grounded_schema_file}")
            else:
                logger.warning(f"'golden' mode enabled but file not found: {grounded_schema_file}")
                
            if rollout_db_id and rollout_question:
                key = _make_key(rollout_db_id, rollout_question)
                if key in pruned_schema_map: # type: ignore
                    logger.debug(f"Using pruned schema for db_id={rollout_db_id}, question='{rollout_question[:30]}...'")
                    return pruned_schema_map[key]
                else:
                    # try the normalised key (tolerate different spaces/newlines)
                    std_key_str = _make_key(rollout_db_id, rollout_question)
                    logger.warning(f"Pruned schema not found for key: {std_key_str}")
                    return "No pruned schema available."
            else:
                logger.warning("Missing db_id or question for pruned mode.")
                return "No pruned schema available."

        else:
            max_chars = 8512
            db = SQLDatabase.from_uri(db_path) # type: ignore
            try:
                full_schema = db.get_table_info()
                if self.truncate == "llm":
                    prompt = DDL_SCHEMA_GROUNDING_PROMPT.invoke( # type: ignore
                        {
                            "question": rollout_question,
                            "schema": full_schema
                        }
                    )
                    linked_schema = self.schema_filter.invoke(prompt).content 
                    return linked_schema
                if len(full_schema) > self.db_schema_truncate:
                    if self.truncate == "truncate":
                        truncated_schema = full_schema[: self.db_schema_truncate] + "\n... (truncated)"
                        return truncated_schema
                    elif self.truncate == "full":
                        return full_schema[:max_chars] + "\n... (truncated)"
            except Exception as e:
                logger.error(f"Failed to get table info: {e}")
                if db_schema:
                    if self.truncate == "llm":
                        prompt = DDL_SCHEMA_GROUNDING_PROMPT.invoke(
                            {
                                "question": rollout_question,
                                "schema": db_schema
                            }
                        )
                        linked_schema = self.schema_filter.invoke(prompt).content
                        return linked_schema
                    if len(db_schema) > self.db_schema_truncate:
                        if self.truncate == "truncate":
                            return db_schema[: self.db_schema_truncate] + "\n... (truncated)"
                        elif self.truncate == "full":
                            return db_schema[:max_chars] + "\n... (truncated)"
                return "No schema available."

    # ------------------------------------------------------------------
    # Trajectory logging (writes to dedicated file, not the main log)
    # ------------------------------------------------------------------

    def _log_trajectory(
        self,
        rollout_id: str,
        mode: str,
        question: str,
        db_id: str,
        ground_truth: str,
        result: Dict[str, Any],
        reward: float | None,
    ) -> None:
        """Extract per-turn details from the agent result and persist them.

        The resulting JSONL record contains, for every ReAct turn:
        - ``input_char_length``: total character length of all messages fed to the LLM
        - ``output``: the LLM's full text response
        - ``output_char_length``: character length of the response
        - ``observation``: the SQL execution result returned to the agent (if any)

        This method is wrapped in a try/except so it never breaks training.
        """
        try:
            messages: list = result.get("messages", [])
            num_turns: int = result.get("num_turns", 0)
            final_query: str = result.get("query", "")

            turns: list[Dict[str, Any]] = []
            # Message layout: [system, user, ai_0, obs_0, ai_1, obs_1, …, ai_n]
            idx = 2  # skip system + user prompt
            turn_num = 0
            while idx < len(messages):
                ai_msg = messages[idx]
                output_text = str(getattr(ai_msg, "content", ai_msg))

                # Compute cumulative input length (all messages before this AI turn)
                input_char_len = sum(
                    len(str(getattr(m, "content", m))) for m in messages[:idx]
                )

                # Extract finish_reason from AIMessage.response_metadata
                finish_reason = getattr(ai_msg, 'response_metadata', {}).get('finish_reason', None)

                turn_info: Dict[str, Any] = {
                    "turn": turn_num + 1,
                    "input_char_length": input_char_len,
                    "output": output_text,
                    "output_char_length": len(output_text),
                    "finish_reason": finish_reason,
                }

                # Observation from the next message (if the loop continued)
                if idx + 1 < len(messages):
                    obs_msg = messages[idx + 1]
                    obs_text = str(getattr(obs_msg, "content", obs_msg))
                    turn_info["observation"] = obs_text
                    # Detect SQL execution errors in the observation
                    obs_lower = obs_text.lower()
                    turn_info["is_sql_error"] = any(
                        kw in obs_lower
                        for kw in ("error", "operationalerror", "no such", "syntax error", "ambiguous")
                    )

                turns.append(turn_info)
                idx += 2
                turn_num += 1

            # Compute rollout-level SQL execution success rate
            turns_with_obs = [t for t in turns if "is_sql_error" in t]
            sql_exec_count = len(turns_with_obs)
            sql_error_count = sum(1 for t in turns_with_obs if t["is_sql_error"])
            sql_success_count = sql_exec_count - sql_error_count

            truncated_turns = sum(1 for t in turns if t.get("finish_reason") == "length")

            record: Dict[str, Any] = {
                "rollout_id": rollout_id,
                "mode": mode,
                "question": question,
                "db_id": db_id,
                "ground_truth": ground_truth,
                "num_turns": num_turns,
                "final_query": final_query,
                "reward": reward,
                "truncated_turns": truncated_turns,
                "sql_exec_count": sql_exec_count,
                "sql_success_count": sql_success_count,
                "sql_error_count": sql_error_count,
                "sql_success_rate": round(sql_success_count / sql_exec_count, 4) if sql_exec_count > 0 else None,
                "turns": turns,
            }
            self._traj_logger.log(record)
        except Exception as exc:
            logger.warning(f"[Trajectory] Failed to log rollout {rollout_id}: {exc}")

    def _log_pass_at_n_trajectories(
        self,
        rollout_id: str,
        question: str,
        db_id: str,
        ground_truth: str,
        valid_results: list[tuple[dict, str | None]],
        pass_at_n_rewards: list[float],
        final_reward: float,
    ) -> None:
        """Log ALL pass@n candidates' full trajectories in a single JSONL record.

        Each candidate includes the complete per-turn details (input/output/observation),
        just like a single-sampling trajectory, so that every candidate's reasoning
        process can be fully reconstructed from the log.
        """
        try:
            candidates_trajectories = []
            for idx, (r, _) in enumerate(valid_results):
                messages: list = r.get("messages", [])
                num_turns: int = r.get("num_turns", 0)
                final_query: str = r.get("query", "")

                turns: list[Dict[str, Any]] = []
                msg_idx = 2  # skip system + user prompt
                turn_num = 0
                while msg_idx < len(messages):
                    ai_msg = messages[msg_idx]
                    output_text = str(getattr(ai_msg, "content", ai_msg))

                    input_char_len = sum(
                        len(str(getattr(m, "content", m))) for m in messages[:msg_idx]
                    )

                    finish_reason = getattr(ai_msg, 'response_metadata', {}).get('finish_reason', None)

                    turn_info: Dict[str, Any] = {
                        "turn": turn_num + 1,
                        "input_char_length": input_char_len,
                        "output": output_text,
                        "output_char_length": len(output_text),
                        "finish_reason": finish_reason,
                    }

                    if msg_idx + 1 < len(messages):
                        obs_msg = messages[msg_idx + 1]
                        obs_text = str(getattr(obs_msg, "content", obs_msg))
                        turn_info["observation"] = obs_text
                        obs_lower = obs_text.lower()
                        turn_info["is_sql_error"] = any(
                            kw in obs_lower
                            for kw in ("error", "operationalerror", "no such", "syntax error", "ambiguous")
                        )

                    turns.append(turn_info)
                    msg_idx += 2
                    turn_num += 1

                turns_with_obs = [t for t in turns if "is_sql_error" in t]
                sql_exec_count = len(turns_with_obs)
                sql_error_count = sum(1 for t in turns_with_obs if t["is_sql_error"])

                candidates_trajectories.append({
                    "candidate_index": idx,
                    "final_query": final_query,
                    "reward": pass_at_n_rewards[idx] if idx < len(pass_at_n_rewards) else None,
                    "num_turns": num_turns,
                    "sql_exec_count": sql_exec_count,
                    "sql_error_count": sql_error_count,
                    "turns": turns,
                })

            record: Dict[str, Any] = {
                "rollout_id": rollout_id,
                "mode": "val",
                "question": question,
                "db_id": db_id,
                "ground_truth": ground_truth,
                "eval_mode": "pass_at_n",
                "n": len(valid_results),
                "final_reward": final_reward,
                "candidates": candidates_trajectories,
            }
            self._traj_logger.log(record)
        except Exception as exc:
            logger.warning(f"[Trajectory] Failed to log pass@n rollout {rollout_id}: {exc}")

    def _log_failed_trajectory(
        self,
        rollout_id: str,
        question: str,
        db_id: str,
        ground_truth: str,
        turn_history: list[Dict[str, Any]],
        error_type: str,
        error_msg: str,
        elapsed: float,
    ) -> None:
        """Log a FAILED rollout to the trajectory file for post-hoc analysis."""
        try:
            record: Dict[str, Any] = {
                "rollout_id": rollout_id,
                "mode": "train",
                "question": question,
                "db_id": db_id,
                "ground_truth": ground_truth,
                "num_turns": len(turn_history),
                "final_query": "",
                "reward": None,
                "status": "FAILED",
                "error_type": error_type,
                "error_msg": error_msg[:500],
                "elapsed": round(elapsed, 2),
                "turns": turn_history,
            }
            self._traj_logger.log(record)
        except Exception as exc:
            logger.warning(f"[Trajectory] Failed to log failed rollout {rollout_id}: {exc}")

    def training_rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        start_time = time.time()
        
        question = task["question"]
        db_id = task["db_id"]
        ground_truth = task["query"]
        evidence = task.get("evidence", "")  # BIRD-specific field
        
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        rollout_id = rollout.rollout_id

        if self.debug:
            print(f"[DEBUG][{rollout_id}] >>> training_rollout START | db_id={db_id} | question={question[:60]}...")
        
        # choose the database path by dataset (local disk, no copy, read directly)
        if self.dataset == "bird":
            db_path = Path("/root/local_bird_db/train_databases") / f"{db_id}/{db_id}.sqlite"
        else:
            db_path = Path(self.spider_dir) / f"database/{db_id}/{db_id}.sqlite"

        if not db_path.exists():
            logger.error(f"Database {db_path} does not exist. Skipping.")
            return None

        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 1: DB at {db_path} (direct read, no copy)")

        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        t0 = time.time()
        db_schema = self.get_table_info(
            is_extend=False,
            mode=rollout.mode, # type: ignore
            db_path="sqlite:///" + db_path.as_posix(),
            rollout_db_id=db_id,
            db_schema="",
            rollout_question=question,
            light_schema={}
        )
        t1 = time.time()
        logger.info(f"[Rollout {rollout_id}] [TIMING] get_table_info: {t1-t0:.3f}s")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 3: get_table_info done ({t1-t0:.3f}s) | schema_len={len(db_schema)}")

        # Run the ReAct agent
        prompt_template = REACT_SQL_BIRD_PROMPT if self.dataset == "bird" else REACT_SQL_PROMPT
        endpoint_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)  # type: ignore
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 4: Building agent graph | endpoint={endpoint_url} | model={llm.model}")

        agent_instance = Agent(
            "sqlite:///" + db_path.as_posix(),
            db_schema=db_schema,
            max_turns=self.max_turns,
            execution_truncate=self.execution_truncate,
            endpoint=endpoint_url,
            verl_replacement={"model": llm.model, **llm.sampling_parameters},
            prompt_template=prompt_template,
            evidence=evidence,
        )
        agent = agent_instance.graph()
        t2 = time.time()
        logger.info(f"[Rollout {rollout_id}] [TIMING] agent_init+graph: {t2-t1:.3f}s")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 5: Agent graph built ({t2-t1:.3f}s) | Starting invoke...")

        try:
            handler = self.tracer.get_langchain_handler()
            result = agent.invoke(  # type: ignore
                {"question": question},  # type: ignore
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            elapsed = time.time() - start_time
            # print the full trajectory to stdout (appears in the wandb output.log)
            import traceback as tb_mod
            turn_history_str = ""
            for th in agent_instance._turn_history:
                turn_history_str += (
                    f"    Turn {th['turn']}: input_chars={th['input_chars']} "
                    f"est_tokens={th['estimated_tokens']} "
                    f"status={th['status']}\n"
                )
                if th['status'] == 'ok':
                    turn_history_str += f"      output_chars={th['output_chars']} preview={th['output_preview'][:150]}\n"
                else:
                    turn_history_str += f"      output={th.get('output', '')}\n"
            if not turn_history_str:
                turn_history_str = "    (no turns completed)\n"
            print(
                f"\n{'='*60}\n"
                f"[ROLLOUT FAILED] {rollout_id}\n"
                f"  db_id: {db_id}\n"
                f"  question: {question[:120]}\n"
                f"  elapsed: {elapsed:.1f}s\n"
                f"  error: {type(e).__name__}: {e}\n"
                f"  turn_history ({len(agent_instance._turn_history)} turns completed):\n"
                f"{turn_history_str}"
                f"  traceback:\n{textwrap.indent(tb_mod.format_exc(), '    ')}\n"
                f"{'='*60}"
            )
            logger.error(
                f"[Rollout {rollout_id}] agent.invoke FAILED after {elapsed:.1f}s | "
                f"{type(e).__name__}: {e}"
            )
            # also record the failed rollout to the trajectory file for post-hoc analysis
            self._log_failed_trajectory(
                rollout_id, question, db_id, ground_truth,
                agent_instance._turn_history, type(e).__name__, str(e), elapsed
            )
            return
        t3 = time.time()
        logger.info(f"[Rollout {rollout_id}] [TIMING] agent_invoke: {t3-t2:.3f}s")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 6: agent.invoke done ({t3-t2:.3f}s) | query={result.get('query', '')[:80]}")

        # Reward calculation (read the local DB directly, no copy)
        if self.reward_mode == "binary":
            _reward_fn = binary_reward_bird if self.dataset == "bird" else binary_reward
            reward = _reward_fn(
                result["query"], ground_truth, db_path.as_posix(), raise_on_error=False,
            )
        elif self.dataset == "bird":
            reward = composite_reward_dynamic_bird(
                result["query"], ground_truth, db_path.as_posix(), raise_on_error=False,
            )
        else:
            reward = composite_reward_dynamic(
                result["query"], ground_truth, db_path.as_posix(), raise_on_error=False,
            )

        total_time = time.time() - start_time

        logger.info(f"[Rollout {rollout_id}] [TIMING] reward: {time.time()-t3:.3f}s")
        logger.info(f"[Rollout {rollout_id}] [TIMING] total: {total_time:.3f}s")
        logger.info(f"[Rollout {rollout_id}] Final reward: {reward}")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 7: reward={reward} | total={total_time:.2f}s | DONE")

        # Persist the full ReAct trajectory to the dedicated trajectory file
        self._log_trajectory(rollout_id, "train", question, db_id, ground_truth, result, reward)

        # Emit tool call stats as annotation span for daemon-side aggregation
        _sql_exec = result.get("sql_exec_count", 0)
        _sql_err = result.get("sql_error_count", 0)
        try:
            agl.emit_annotation({
                "tool_stats.sql_execute.calls": _sql_exec,
                "tool_stats.sql_execute.errors": _sql_err,
                "tool_stats.sql_execute.successes": _sql_exec - _sql_err,
            })
        except Exception as exc:
            logger.warning(f"[Rollout {rollout_id}] Failed to emit tool stats annotation: {exc}")

        return reward
    
    def validation_rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = task["question"]
        db_id = task["db_id"]
        ground_truth = task.get("query") or task.get("SQL", "")
        # BIRD-specific field
        evidence = task.get("evidence", "")
        
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        
        # choose the database path by dataset (local disk, read directly, no copy)
        if self.dataset == "bird":
            db_path = Path("/root/local_bird_db/dev_databases") / f"{db_id}/{db_id}.sqlite"
        else:
            db_path = Path(self.spider_dir) / f"test_database/{db_id}/{db_id}.sqlite"

        if not db_path.exists():
            logger.error(f"Database {db_path} does not exist. Skipping.")
            return None

        rollout_id = rollout.rollout_id
        
        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")
        
        def single_sampling(endpoint_url: str | None = None) -> tuple[dict | None, str | None]:
            """A single ReAct inference; returns (result, db_schema) or (None, None)."""
            try:
                db_schema = self.get_table_info(
                    is_extend=False,
                    mode=rollout.mode, # type: ignore
                    db_path="sqlite:///" + db_path.as_posix(),
                    rollout_db_id=db_id,
                    db_schema="",
                    rollout_question=question,
                    light_schema={}
                )

                # load balancing: prefer the passed-in endpoint_url, else fall back to llm.get_base_url
                _ep = endpoint_url
                if _ep is None:
                    _ep = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)  # type: ignore

                prompt_template = REACT_SQL_BIRD_PROMPT if self.dataset == "bird" else REACT_SQL_PROMPT
                agent = Agent(
                    "sqlite:///" + db_path.as_posix(),
                    db_schema=db_schema,
                    max_turns=self.max_turns,
                    execution_truncate=self.execution_truncate,
                    endpoint=_ep,
                    verl_replacement=(
                        {
                            "model": llm.model,
                            # force deterministic decoding (0.0) for val, ignoring llm.sampling_parameters.
                            # reason: the framework daemon injects the train temperature (1.1) into
                            # sampling_parameters, so dict.get("temperature", val_temperature) always returns
                            # 1.1 and val_temperature becomes dead code.
                            "temperature": self.val_temperature,
                        }
                    ),
                    prompt_template=prompt_template,
                    evidence=evidence,
                ).graph()
                
                handler = self.tracer.get_langchain_handler()
                result = agent.invoke(  # type: ignore
                    {"question": question},  # type: ignore
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
                return result, db_schema
            except Exception as e:
                logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
                return None, None
        
        query = ""
        result: dict = {}

        if self.val_concurrent <= 1:
            # single inference
            t_start = time.time()
            result, _ = single_sampling()
            if result is not None:
                query = result["query"]
            else:
                query = ""
                result = {}
            logger.info(f"[Rollout {rollout_id}] [TIMING] single_sampling: {time.time()-t_start:.3f}s")
        else:
            # parallel inference (multi-path sampling)
            # load balancing: if there are multiple vLLM endpoints, spread the concurrent requests across instances
            endpoints = self.llm_endpoints if self.llm_endpoints else [llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)]  # type: ignore
            t_start = time.time()
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.val_concurrent) as executor:
                futures = [
                    executor.submit(single_sampling, endpoints[i % len(endpoints)])
                    for i in range(self.val_concurrent)
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            if self.eval_mode == "pass_at_n":
                # ===== Pass@N mode =====
                # verify each SQL one by one; as long as one is correct, reward = 1
                pass_at_n_rewards = []
                valid_results = [(r, s) for r, s in results if r is not None]
                _reward_fn = binary_reward_bird if self.dataset == "bird" else binary_reward
                for r, _ in valid_results:
                    individual_reward = _reward_fn(
                        r["query"], ground_truth, db_path.as_posix(), raise_on_error=False
                    )
                    pass_at_n_rewards.append(individual_reward)

                # any one correct -> 1, all wrong -> 0
                reward = 1.0 if any(r == 1.0 for r in pass_at_n_rewards) else 0.0
                query = valid_results[0][0]["query"] if valid_results else ""
                result = valid_results[0][0] if valid_results else {}

                # save all N candidate SQLs and their individual reward (with full trajectories)
                pass_at_n_candidates = []
                for idx, (r, _) in enumerate(valid_results):
                    pass_at_n_candidates.append({
                        "index": idx,
                        "query": r.get("query", ""),
                        "reward": pass_at_n_rewards[idx] if idx < len(pass_at_n_rewards) else None,
                        "num_turns": r.get("num_turns", 0),
                    })
                # attach to result, for export_rollouts to export
                result["pass_at_n_candidates"] = pass_at_n_candidates
                result["pass_at_n_final_reward"] = reward

                logger.info(
                    f"[Rollout {rollout_id}] Pass@{self.val_concurrent}: "
                    f"individual_rewards={pass_at_n_rewards} → final={reward}"
                )
                logger.info(f"[Rollout {rollout_id}] [TIMING] pass_at_n: {time.time()-t_start:.3f}s")

                # Persist ALL candidates' full trajectories (save the full trajectory of every candidate)
                self._log_pass_at_n_trajectories(
                    rollout_id, question, db_id, ground_truth,
                    valid_results, pass_at_n_rewards, reward
                )

                _sql_exec = result.get("sql_exec_count", 0)
                _sql_err = result.get("sql_error_count", 0)
                try:
                    agl.emit_annotation({
                        "tool_stats.sql_execute.calls": _sql_exec,
                        "tool_stats.sql_execute.errors": _sql_err,
                        "tool_stats.sql_execute.successes": _sql_exec - _sql_err,
                        "pass_at_n.candidates": json.dumps(pass_at_n_candidates),
                        "pass_at_n.n": self.val_concurrent,
                        "pass_at_n.reward": reward,
                    })
                except Exception as exc:
                    logger.warning(f"[Rollout {rollout_id}] Failed to emit tool stats annotation: {exc}")
                return reward

            else:
                # ===== Best of N mode (original logic) =====
                # group by execution result to reduce context length
                execution_groups: dict[str, list[tuple[dict, str]]] = {}
                for r, db_schema in results:
                    if r is None:
                        continue
                    # truncate the execution result to 2048 chars, aligned with elsewhere
                    exec_str = str(r.get("execution", ""))
                    if len(exec_str) > self.execution_truncate:
                        exec_str = exec_str[:self.execution_truncate] + "\n... (truncated)"
                    if exec_str not in execution_groups:
                        execution_groups[exec_str] = []
                    execution_groups[exec_str].append((r, db_schema))

                candidate_blocks = ""
                group_idx = 1
                for exec_key, group in execution_groups.items():
                    # keep only one execution-result sample per group
                    first_result, first_schema = group[0]
                    sql_list = [r["query"] for r, _ in group]
                    candidate_blocks += f"""
                    Candidate Group {group_idx}:
                    Schema: {first_schema}
                    SQL(s) producing this result ({len(sql_list)} total):
                    {chr(10).join(f'  - {sql}' for sql in sql_list)}
                    Result: {exec_key}\n\n
                    """
                    group_idx += 1

                if candidate_blocks:
                    prompt = SUMMARIZE_PROMPT.invoke(
                        {
                            "question": question,
                            "candidate_blocks": candidate_blocks
                        }
                    )
                    query = self.summarizer_llm.invoke(prompt).content
                else:
                    query = ""
                    logger.warning(f"[Rollout {rollout_id}] All concurrent samples failed, no candidates for summarization")

                # use the first result of the first group for the trajectory log
                if execution_groups:
                    first_group = next(iter(execution_groups.values()))
                    result = first_group[0][0]
                else:
                    result = {}

                logger.info(f"[Rollout {rollout_id}] [TIMING] concurrent_sampling+summarize: {time.time()-t_start:.3f}s")

        logger.info(f"[Rollout {rollout_id}] Generated Query: {query}")

        # Log finish_reason for each LLM call
        if result:
            finish_reasons = [
                getattr(msg, 'response_metadata', {}).get('finish_reason')
                for msg in result.get("messages", [])
                if getattr(msg, 'response_metadata', {}).get('finish_reason') is not None
            ]
            truncated_count = sum(1 for fr in finish_reasons if fr == "length")
            logger.info(f"[Rollout {rollout_id}] finish_reasons: {finish_reasons}")
            if truncated_count > 0:
                logger.warning(
                    f"[Rollout {rollout_id}] {truncated_count}/{len(finish_reasons)} "
                    f"LLM calls truncated (finish_reason=length)"
                )
            
        t_start = time.time()

        # read the DB directly to compute reward, no copy (SELECT-only is safe)
        _reward_fn = binary_reward_bird if self.dataset == "bird" else binary_reward
        reward = _reward_fn(query, ground_truth, db_path.as_posix(), raise_on_error=False)

        logger.info("[Rollout %s] Reward: %s", rollout_id, reward)
        logger.info("[Rollout %s] Time taken for reward calculation: %.2f seconds", rollout_id, time.time() - t_start)

        # Persist the full ReAct trajectory to the dedicated trajectory file
        self._log_trajectory(rollout_id, "val", question, db_id, ground_truth, result, reward)

        # Emit tool call stats as annotation span for daemon-side aggregation
        _sql_exec = result.get("sql_exec_count", 0)
        _sql_err = result.get("sql_error_count", 0)
        try:
            agl.emit_annotation({
                "tool_stats.sql_execute.calls": _sql_exec,
                "tool_stats.sql_execute.errors": _sql_err,
                "tool_stats.sql_execute.successes": _sql_exec - _sql_err,
            })
        except Exception as exc:
            logger.warning(f"[Rollout {rollout_id}] Failed to emit tool stats annotation: {exc}")

        return reward