import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config import get_ppo_config
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLMWithValueHead.from_pretrained("microsoft/DialoGPT-small")

print("Loading PPO config...")
ppo_config = get_ppo_config()

print("Building PPO trainer...")
trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer
)



print("\n🎉 PPO Trainer created successfully!")
