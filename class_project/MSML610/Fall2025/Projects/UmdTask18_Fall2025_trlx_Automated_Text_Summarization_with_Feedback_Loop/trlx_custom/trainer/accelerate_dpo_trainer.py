import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

from trlx.data.configs import TRLConfig
from trlx.data.method_configs import MethodConfig, register_method
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer

# Ensure pipeline registration happens when this module is imported
from trlx_custom.pipeline import dpo_pipeline as _  # noqa: F401


@register_method
class DPOConfig(MethodConfig):
    """
    Minimal DPO method config for seq2seq models.
    """

    beta: float = 0.1
    gen_kwargs: dict = None

    def __init__(self, name="dpo", beta=0.1, gen_kwargs=None):
        super().__init__(name=name)
        self.beta = beta
        self.gen_kwargs = gen_kwargs or {"max_new_tokens": 64}

    @classmethod
    def from_dict(cls, config):
        return cls(**config)


@register_trainer
class AccelerateDPOTrainer(AccelerateRLTrainer):
    """
    Lightweight DPO trainer for encoder-decoder models (e.g., T5).
    """

    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.generate_kwargs = dict(
            self.config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def get_arch(self, config):
        return AutoModelForSeq2SeqLM.from_pretrained(config.model.model_path)

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size, shuffle=True)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size, shuffle=False)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, train_dataloader, eval_dataloader)

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def loss(self, batch):
        beta = getattr(self.config.method, "beta", 0.1)

        def seq_logps(labels):
            outputs = self.model(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                labels=labels,
            )
            logits = outputs.logits
            vocab = logits.size(-1)
            labels_flat = labels.view(-1)
            logits_flat = logits.view(-1, vocab)
            loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            loss = loss_flat.view(labels.shape)
            mask = (labels != -100).float()
            token_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return -token_loss

        logp_chosen = seq_logps(batch["chosen_labels"])
        logp_rejected = seq_logps(batch["rejected_labels"])

        dpo_loss = -F.logsigmoid(beta * (logp_chosen - logp_rejected))
        loss = dpo_loss.mean()

        stats = {
            "loss": loss,
            "logp_chosen": logp_chosen.mean(),
            "logp_rejected": logp_rejected.mean(),
            "margin": (logp_chosen - logp_rejected).mean(),
        }
        return loss, stats

    def evaluate(self):
        self.model.eval()
        all_decoded = []
        for batch in self.eval_dataloader:
            input_ids = batch.get("prompt_input_ids")
            if input_ids is None:
                input_ids = batch.get("input_ids")

            attention_mask = batch.get("prompt_attention_mask")
            if attention_mask is None:
                attention_mask = batch.get("attention_mask")

            with torch.inference_mode():
                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **self.generate_kwargs,
                )
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            all_decoded.extend(decoded)
        return {"eval/samples": len(all_decoded)}
