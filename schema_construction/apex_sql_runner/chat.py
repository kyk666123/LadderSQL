import os
import time
import random
import threading
from openai import OpenAI

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY",  "") or os.environ.get("JUDGE_API_KEY", "")
# Model name is provider-specific: SiliconFlow -> zai-org/GLM-5.2 ; DashScope -> glm-5.2.
# Honour JUDGE_API_BASE (the single knob the rest of the pipeline uses) before falling back.
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") or os.environ.get("JUDGE_API_BASE") or "https://api.siliconflow.cn/v1"
DEFAULT_MODEL   = os.environ.get("GLM_MODEL", "glm-5.2")

# global rate-limit counter (cross-thread), aggregated by the runner
RATE_LIMIT_COUNTER = 0
_rl_lock = threading.Lock()


def _bump_rate_limit():
    global RATE_LIMIT_COUNTER
    with _rl_lock:
        RATE_LIMIT_COUNTER += 1


class RateLimitError(Exception):
    """限流专用异常，消息带 ##RATELIMIT## 前缀，供 runner 识别。"""
    pass


def _is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    keys = ["429", "rate limit", "ratelimit", "too many requests", "throttl",
            "flow control", "requests per", "quota", "limit exceeded",
            "tpm", "rpm", "concurrency limit", "allocated", "服务限流", "限流"]
    return any(k in m for k in keys)


class GPTChat:
    """多轮对话客户端，接口对齐 schema_linking.py 的用法：
       init_messages() / get_model_response_txt(prompt, tag) / .messages
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2,
                 max_tokens: int = 8192, api_key: str = OPENAI_API_KEY,
                 api_base: str = OPENAI_API_BASE, timeout: int = 180):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.messages = []
        # max_retries=0：把重试完全交给本类，便于识别限流
        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout, max_retries=0)

    def init_messages(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_model_response_txt(self, prompt, tag: str = ""):
        self.messages.append({"role": "user", "content": prompt})
        content = self._call_api()
        self.messages.append({"role": "assistant", "content": content})
        return content

    # 兼容可能存在的别名调用
    def get_model_response(self, prompt, tag: str = ""):
        return self.get_model_response_txt(prompt, tag)

    def _call_api(self, max_retries: int = 4):
        delay = 2.0
        last_err = None
        for attempt in range(max_retries):
            try:
                _t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"enable_thinking": False},   # 关闭 GLM 思考（关键）
                )
                try:
                    import llm_usage
                    llm_usage.record("schema_link", getattr(resp, "usage", None), time.time() - _t0, model=self.model)
                except Exception:  # noqa: BLE001  统计绝不影响主流程
                    pass
                content = resp.choices[0].message.content
                if content is None or content.strip() == "":
                    raise ValueError("empty content from model")
                return content
            except Exception as e:
                last_err = e
                msg = str(e)
                if _is_rate_limit(msg):
                    _bump_rate_limit()
                    # 限流：本地短退避几次；仍失败则抛 RateLimitError 交 runner 统一重跑
                    if attempt < max_retries - 1:
                        time.sleep(delay + random.uniform(0, 1.5))
                        delay = min(delay * 2, 30)
                        continue
                    raise RateLimitError("##RATELIMIT## " + msg)
                # 其它瞬时错误：少量重试
                if attempt < max_retries - 1:
                    time.sleep(1.0 + random.uniform(0, 1.0))
                    continue
                raise
        raise last_err
