# src/utils/response_utils.py

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ResponseGenerator:
    """
    Multilingual response generator using Qwen2.5-1.5B-Instruct.

    Modes:
      - lang_code != "en": reply ONLY in that language (no translation)
      - lang_code == "en": pure English translation of the input text
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        print(f"[ResponseGenerator] Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model (dtype fix)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            trust_remote_code=True,
            dtype=torch.float32
        )

        # Ensure pad token exists
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Language name map (for same-language reply)
        self.lang_map = {
            "en": "English",
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "it": "Italian",
            "pt": "Portuguese",
            "ca": "Catalan",
            "hi": "Hindi",
            "zh": "Chinese",
            "ar": "Arabic",
            "ru": "Russian",
        }

    def _language_name(self, lang_code):
        if lang_code is None:
            return None
        return self.lang_map.get(lang_code, lang_code)

    # ===============================================================
    #                 *** MAIN GENERATION METHOD ***
    # ===============================================================
    def generate(self, text: str, lang_code: str | None):
        """
        If lang_code == "en": return ONLY an English translation of `text`.
        Otherwise: return a short reply in the target language, with NO translation.
        """

        # -----------------------------------------------------------
        # SPECIAL CASE — PURE ENGLISH TRANSLATION
        # -----------------------------------------------------------
        if lang_code == "en":
            prompt = (
                "Translate the following text into clear, natural English.\n"
                "Return ONLY the English translation.\n"
                "Do NOT include the original language.\n"
                "Do NOT add explanations, notes, or labels.\n"
                "Return one clean English paragraph.\n"
                "End your answer with ###.\n\n"
                f"Text:\n{text}\n\n"
                "English translation:"
            )

        else:
            # Normal chatbot behavior: reply in same language as user
            lang_full = self._language_name(lang_code)

            if lang_full:
                prompt = (
                    f"You are a multilingual AI assistant.\n"
                    f"Reply ONLY in {lang_full}.\n"
                    f"Write clearly and naturally.\n"
                    f"Give a short, helpful answer (1–3 sentences).\n"
                    f"Do NOT provide translations.\n"
                    f"Do NOT include any English.\n"
                    f"Do NOT add extra commentary.\n"
                    f"End your reply with ###.\n\n"
                    f"User message:\n{text}\n\n"
                    f"Assistant reply in {lang_full}:"
                )
            else:
                # Language unknown -> reply in same language
                prompt = (
                    "You are a multilingual AI assistant.\n"
                    "Reply ONLY in the same language as the user.\n"
                    "Write clearly and naturally.\n"
                    "Give a short, helpful answer (1–3 sentences).\n"
                    "Do NOT provide translations.\n"
                    "Do NOT include English.\n"
                    "End your reply with ###.\n\n"
                    f"User message:\n{text}\n\n"
                    "Assistant reply:"
                )

        # ===========================================================
        # TOKENIZATION
        # ===========================================================
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # ===========================================================
        # GENERATION SETTINGS
        # ===========================================================
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.3,
                top_p=0.7,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
            )

        # Decode
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip prompt
        reply = full_output.replace(prompt, "").strip()

        # Remove trailing ###
        reply = reply.split("###")[0].strip()

        # Clean unwanted artifacts
        for bad in ["Human:", "Assistant:", "User:", "Q:", "A:"]:
            if bad in reply:
                reply = reply.split(bad)[0].strip()

        # Extra cleanup for translation mode
        reply = reply.replace("English translation:", "").replace("Translation:", "").strip()
        reply = reply.replace("EN:", "").strip()

        gc.collect()
        return reply
