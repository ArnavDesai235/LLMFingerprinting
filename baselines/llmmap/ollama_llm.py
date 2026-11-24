"""Ollama backend with the interface LLMmap's PromptConf expects
(llm_name, make_prompt(system, user), generate(prompt, gen_kargs) -> [str]).

Matches LLMmap's own settings: max_new_tokens = 100, sampling params from the prompt conf.
Reasoning models: LLMmap's pretrained embedder was trained on plain chat replies, and 100 tokens of
<think> text would hide the answer, so by default we ask Ollama for think=False (falls back to a
plain call if the model rejects the flag) and strip any <think>...</think> block from the reply.
Use think='raw' to keep whatever Ollama returns (second arm for reasoning models)."""
import os, re, time
import requests

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
MAX_NEW_TOKENS = 100
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


class LLM_ollama:
    def __init__(self, label, tag, reasoning=False, think="off", timeout=600, retries=3):
        self.llm_name, self.tag, self.reasoning, self.think = label, tag, reasoning, think
        self.timeout, self.retries, self.is_hf = timeout, retries, False
        self._think_ok = True  # flips to False if the server rejects the think flag

    def make_prompt(self, system, user):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return msgs

    def generate(self, prompt, gen_kargs, max_new_tokens=MAX_NEW_TOKENS):
        opts = {"num_predict": max_new_tokens}
        t = gen_kargs.get("temperature", 0.7)
        opts["temperature"] = t if gen_kargs.get("do_sample", True) else 0.0
        if not gen_kargs.get("do_sample", True):
            opts["seed"] = 0
        body = {"model": self.tag, "messages": prompt, "stream": False, "options": opts}
        use_think_flag = self.reasoning and self.think == "off" and self._think_ok
        if use_think_flag:
            body["think"] = False
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{HOST}/api/chat", json=body, timeout=self.timeout)
                if r.status_code == 400 and use_think_flag and "think" in r.text.lower():
                    self._think_ok = False          # model does not support the flag
                    body.pop("think"); use_think_flag = False
                    continue
                r.raise_for_status()
                msg = r.json()["message"]
                text = msg.get("content") or ""
                if self.think != "raw":
                    text = _THINK.sub("", text).strip()
                if not text.strip() and msg.get("thinking"):
                    # model ignored think=False and spent its 100 tokens reasoning: keep that text, as LLMmap
                    # would see the first 100 tokens of whatever the endpoint returns
                    text = msg["thinking"]
                return [text]
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
