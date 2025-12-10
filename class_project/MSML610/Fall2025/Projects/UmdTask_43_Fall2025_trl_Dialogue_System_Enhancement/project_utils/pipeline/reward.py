import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util


class RewardFunction:
    def __init__(self, device=None):

        # Auto detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ---- 1. Sentiment Model ----
        self.sent_tok = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        ).to(device)

        # ---- 2. Toxicity Model ----
        self.tox_tok = AutoTokenizer.from_pretrained(
            "unitary/unbiased-toxic-roberta"
        )
        self.tox_model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/unbiased-toxic-roberta"
        ).to(device)

        # ---- 3. Coherence Encoder ----
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2", device=device
        )

        # ---- 4. Harmful Keywords ----
        self.harmful_keywords = [
            "weed", "cocaine", "heroin", "acid", "dope",
            "ganja", "pot", "stoned", "drug", "overdose",
            "kill yourself", "suicide",
        ]

        # ---- 5. Reward Weights ----
        self.w_sent = 1.0
        self.w_coh  = 1.0
        self.w_tox  = 4.0
        self.w_harm = 2.0
        self.w_len  = 0.3
        self.w_div  = 0.1
        self.w_rep  = 1.0

    # ------------------------------------------
    # Components
    # ------------------------------------------

    @torch.no_grad()
    def _sentiment(self, texts):
        enc = self.sent_tok(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        logits = self.sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        neg, neu, pos = probs[:, 0], probs[:, 1], probs[:, 2]
        return pos - neg

    @torch.no_grad()
    def _toxicity(self, texts):
        enc = self.tox_tok(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        logits = self.tox_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]  # toxic probability

    @torch.no_grad()
    def _coherence(self, prompts, responses):
        emb_p = self.embedder.encode(prompts, convert_to_tensor=True)
        emb_r = self.embedder.encode(responses, convert_to_tensor=True)
        sims = util.cos_sim(emb_p, emb_r).diagonal()
        return sims.clamp(0, 1)

    def _length_reward(self, responses, tokenizer):
        true_lengths = []
        for r in responses:
            toks = tokenizer.encode(r, add_special_tokens=False)
            true_lengths.append(len(toks))

        lengths = torch.tensor(true_lengths, dtype=torch.float32, device=self.device)
        good = (lengths >= 6) & (lengths <= 30)

        pos_val = torch.tensor(0.2, device=self.device)
        neg_val = torch.tensor(-0.2, device=self.device)
        return torch.where(good, pos_val, neg_val)

    def _distinct2(self, responses):
        scores = []
        for r in responses:
            toks = r.split()
            if len(toks) < 2:
                scores.append(0.0)
                continue
            bigrams = list(zip(toks, toks[1:]))
            scores.append(len(set(bigrams)) / len(bigrams))
        return torch.tensor(scores, device=self.device)

    def _repetition_penalty(self, prompts, responses):
        penalties = []
        for p, r in zip(prompts, responses):
            p_low = p.lower().strip()
            r_low = r.lower().strip()

            # Too similar
            if r_low.startswith(p_low[:40]):
                penalties.append(-1.0)
            elif p_low in r_low[:len(p_low)]:
                penalties.append(-0.5)
            else:
                penalties.append(0.0)

        return torch.tensor(penalties, device=self.device)

    def _harmful_keyword_penalty(self, responses):
        penalties = []
        for r in responses:
            r_low = r.lower()
            if any(k in r_low for k in self.harmful_keywords):
                penalties.append(-1.0)
            else:
                penalties.append(0.0)
        return torch.tensor(penalties, device=self.device)

    # ------------------------------------------
    # MASTER REWARD
    # ------------------------------------------

    @torch.no_grad()
    def __call__(self, prompts, responses, gen_tokenizer=None):

        assert gen_tokenizer is not None, \
            "Pass the generation tokenizer into RewardFunction.__call__"

        # Batch everything to avoid repeated model calls
        s_sent = self._sentiment(responses)
        s_tox  = self._toxicity(responses)
        s_coh  = self._coherence(prompts, responses)
        s_len  = self._length_reward(responses, gen_tokenizer)
        s_div  = self._distinct2(responses)
        s_rep  = self._repetition_penalty(prompts, responses)
        s_harm = self._harmful_keyword_penalty(responses)

        final = (
            self.w_sent * s_sent +
            self.w_coh  * s_coh  -
            self.w_tox  * s_tox  +
            self.w_len  * s_len  +
            self.w_div  * s_div  -
            self.w_rep  * s_rep  -
            self.w_harm * s_harm
        )

        return final.cpu().tolist()
