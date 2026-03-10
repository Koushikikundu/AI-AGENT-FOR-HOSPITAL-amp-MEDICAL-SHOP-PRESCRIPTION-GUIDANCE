from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from backend.config import GENERATION_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE


class HospitalAnswerGenerator:
    def __init__(self, model_name: str = GENERATION_MODEL_NAME) -> None:
        self.model_name = model_name
        self.task = "text-generation"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception:
            # Safe fallback if a large instruct model cannot be loaded on the host.
            self.task = "text2text-generation"
            self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
You are a hospital medicine assistant.
Use only the provided context.
If context is not sufficient, say what is missing.

Context:
{context}

Question:
{question}

Answer clearly with:
1) direct answer
2) dosage guidance from context (if present)
3) alternative medicine (if present)
"""
        kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        if self.task == "text-generation":
            kwargs.update(
                {
                    "temperature": TEMPERATURE,
                    "do_sample": True,
                    "return_full_text": False,
                }
            )
        out = self.generator(prompt, **kwargs)[0]["generated_text"]
        return out.strip()
