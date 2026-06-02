import json
import os
import re
from typing import List, Optional

from .utils import parse_json_relaxed, stable_hash_float


class BaseLLMClient:
    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        raise NotImplementedError

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return self.complete(system_prompt + "\n\n" + user_prompt, temperature, max_tokens)


class OpenAICompatibleLLMClient(BaseLLMClient):
    """OpenAI-compatible chat client, usable with vLLM/TGI gateways."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")}
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["timeout"] = timeout
        self.client = OpenAI(**kwargs)
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class LocalTransformersLLMClient(BaseLLMClient):
    """Local Hugging Face Transformers chat client for quick GPU tests."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_context_tokens: int = 8192,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }.get(dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        load_kwargs = {
            "dtype": torch_dtype,
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = {"": self.device}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        self.max_context_tokens = max_context_tokens

    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        return self._generate([{"role": "user", "content": prompt}], temperature, max_tokens)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._generate(messages, temperature, max_tokens)

    def _generate(self, messages: list, temperature: float, max_tokens: int) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        ).to(self.device)
        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.8
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return text


class MockLLMClient(BaseLLMClient):
    """Deterministic local stand-in for smoke tests.

    It does not reproduce model quality; it only exercises the same algorithmic
    interface when Llama 3.3 70B / 3.1 8B are not locally available.
    """

    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        if "Define numerical features" in prompt:
            return json.dumps(self._features(prompt))
        if "Text objects to rate:" in prompt and "Return ONLY the JSON object" in prompt:
            return json.dumps(self._extract(prompt))
        if "Generate exactly" in prompt and "JSON array" in prompt:
            return json.dumps(self._prompt_array(prompt))
        if "Target feature vector:" in prompt:
            return self._generate_from_target(prompt)
        if "Feature gap analysis" in prompt:
            return self._refine(prompt)
        return "Reason carefully, follow the requested output format, and provide only the final answer."

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        # This is only used if the benchmark evaluator is accidentally paired
        # with the mock target.
        if "True or False" in user_prompt:
            return "True"
        if "Yes or No" in user_prompt:
            return "Yes"
        if "valid or invalid" in user_prompt:
            return "valid"
        if "letter choice" in user_prompt:
            return "(A)"
        return "0"

    def _features(self, prompt: str) -> List[dict]:
        task = prompt.lower()
        features = [
            {
                "name": "format_discipline",
                "description": (
                    "Measures how strongly the prompt enforces the exact answer format required "
                    "by the task. 0 means no output-format control; 1 means it clearly requires only "
                    "the final answer in the requested form, which directly affects accuracy under "
                    "regex-based evaluation."
                ),
            },
            {
                "name": "explicit_reasoning",
                "description": (
                    "Measures whether the prompt asks the assistant to reason through task-relevant "
                    "steps before answering. 0 means direct guessing; 1 means clear internal or "
                    "stepwise analysis appropriate for the task."
                ),
            },
        ]
        if any(word in task for word in ["sarcastic", "pronoun", "table", "objects", "boolean", "causal", "fallacies", "adjective", "math"]):
            features.append(
                {
                    "name": "task_specific_cues",
                    "description": (
                        "Measures whether the prompt names cues specific to this task, such as "
                        "logical operators, causal counterfactuals, ambiguity resolution, table "
                        "parsing, object tracking, sarcasm cues, adjective order, or arithmetic "
                        "consistency. 0 means generic advice; 1 means concrete task-specific cues."
                    ),
                }
            )
        return features

    def _extract(self, prompt: str) -> dict:
        feature_names = re.findall(r"^- ([A-Za-z0-9_]+):", prompt, flags=re.MULTILINE)
        objects = re.findall(
            r'-- Text Object ID: "([^"]+)" --\n(.*?)(?=\n-- Text Object ID: "|\Z)',
            prompt,
            flags=re.DOTALL,
        )
        out = {}
        for tid, text in objects:
            lower = text.lower()
            row = {}
            for name in feature_names:
                if name == "format_discipline":
                    value = self._score_terms(
                        lower,
                        ["only", "just", "final", "format", "letter", "true", "false", "yes", "no", "valid"],
                    )
                elif name == "explicit_reasoning":
                    value = self._score_terms(
                        lower,
                        ["reason", "step", "careful", "analyze", "track", "parse", "compute", "check"],
                    )
                elif name == "task_specific_cues":
                    value = self._score_terms(
                        lower,
                        [
                            "sarcasm",
                            "literal",
                            "figurative",
                            "logical",
                            "operator",
                            "causal",
                            "pronoun",
                            "table",
                            "swap",
                            "adjective",
                            "arithmetic",
                        ],
                    )
                else:
                    value = stable_hash_float(name + text, 0.2, 0.8)
                row[name] = round(value, 3)
            out[tid] = row
        return out

    def _score_terms(self, text: str, terms: List[str]) -> float:
        hits = sum(1 for term in terms if term in text)
        length_penalty = 0.15 if len(text.split()) > 130 else 0.0
        return max(0.0, min(1.0, 0.15 + 0.16 * hits - length_penalty))

    def _prompt_array(self, prompt: str) -> List[str]:
        match = re.search(r"Generate exactly (\d+)", prompt)
        q = int(match.group(1)) if match else 5
        bases = [
            "Solve the task carefully. Analyze the input step by step, then provide only the requested final answer.",
            "Focus on the task-specific cues, check the answer format, and output only the final choice or value.",
            "Use concise reasoning to identify the correct answer. Do not include extra explanation in the final response.",
            "Parse the problem precisely, avoid assumptions, and return the answer in exactly the required format.",
            "Think through the relevant evidence internally, verify consistency, and give only the final answer.",
        ]
        return [bases[i % len(bases)] for i in range(q)]

    def _generate_from_target(self, prompt: str) -> str:
        target = self._target_dict(prompt)
        parts = ["You are a precise assistant for this task."]
        if target.get("explicit_reasoning", 0.5) >= 0.45:
            parts.append("Reason through the relevant steps carefully before deciding.")
        if target.get("task_specific_cues", 0.5) >= 0.45:
            parts.append(
                "Attend to task-specific cues such as exact wording, logical structure, ambiguity, tables, swaps, tone, or arithmetic consistency."
            )
        if target.get("format_discipline", 0.5) >= 0.35:
            parts.append("Return only the final answer in the requested format, with no extra text.")
        return " ".join(parts)

    def _refine(self, prompt: str) -> str:
        match = re.search(r"Current system prompt:\n(.*?)\n\nFeature gap analysis", prompt, re.DOTALL)
        current = match.group(1).strip() if match else "Solve the task carefully."
        gaps_match = re.search(r"Feature gap analysis.*?(\[\s*\{.*?\}\s*\])", prompt, re.DOTALL)
        if not gaps_match:
            return current
        try:
            gaps = parse_json_relaxed(gaps_match.group(1))
        except Exception:
            return current
        additions = []
        for gap in gaps[:2]:
            if gap.get("direction") != "increase":
                continue
            name = gap.get("feature_name", "")
            if name == "format_discipline":
                additions.append("Use exactly the requested answer format.")
            elif name == "explicit_reasoning":
                additions.append("Check the reasoning path before answering.")
            elif name == "task_specific_cues":
                additions.append("Use concrete task-specific cues instead of generic guessing.")
        return (current + " " + " ".join(additions)).strip()

    def _target_dict(self, prompt: str) -> dict:
        match = re.search(r"Target feature vector:\n(\{.*?\})", prompt, re.DOTALL)
        if not match:
            return {}
        try:
            return parse_json_relaxed(match.group(1))
        except Exception:
            return {}
