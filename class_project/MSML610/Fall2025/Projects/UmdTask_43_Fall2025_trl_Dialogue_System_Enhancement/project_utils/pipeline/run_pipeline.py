"""
pipeline.py

Runs the full end-to-end workflow:
1. Load base model
2. Load RLHF fine-tuned model
3. Run pre-evaluation on base model
4. Run post-evaluation on fine-tuned model
5. Save outputs
"""

from pipeline.base_model import load_base_model
from pipeline.ppo_model import load_finetuned_model
from pipeline.evaluation import evaluation
from pipeline.user_feedback import generate_feedback



def main():

    print("\n========== STEP 1: Load Base Model ==========\n")
    base_tokenizer, base_model = load_base_model()

    print("\n========== STEP 2: Load Fine-Tuned Model ==========\n")
    ft_tokenizer, ft_model = load_finetuned_model()

    print("\n========== STEP 3: Pre-Evaluation on Base Model ==========\n")
    pre_results = run_pre_evaluation(base_model, base_tokenizer)

    print("\n========== STEP 4: Evaluation on Fine-Tuned Model ==========\n")
    post_results = run_post_evaluation(ppo_model, base_tokenizer, )

    print("\n========== STEP 5: Runing Feedback script ==========\n")
    generate_feedback(pre_results, post_results)

    print("\n==========STEP 6:  Supervised _finetuning model Evaluation========\n")
    sft_results=run_sft_evaluation(sft_finetuned,) base_tokenizer


    print("\n🎉 Pipeline Completed Successfully!\n")


if __name__ == "__main__":
    main()
