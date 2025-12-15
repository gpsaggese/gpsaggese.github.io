def evaluation(model, tokenizer):
    """
    Run pre-evaluation using:
    - Coherence
    - Sentiment
    - BLEU
    - ROUGE
    - BERTScore
    - Distinct-2
    """

    print("\n🔍 Running PRE-EVALUATION...\n")

    # ===========================
    # 1. Imports
    # ===========================
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, util
    import evaluate
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification



    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===========================
    # 2. Load Dataset (TEST SPLIT)
    # ===========================
    ds = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates", split="test")

    N = 200   # evaluate on first 200 samples
    prompts = ds["first"][:N]
    ground_truth = ds["second"][:N]

    # ===========================
    # 3. Generate responses from BASE model
    # ===========================
    def generate_response(prompt):
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inp,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    generated = [generate_response(p) for p in prompts]

    # ===========================
    # 4. Embedding Model (Coherence)
    # ===========================
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def coherence_score(p, r):
        emb_p = embedder.encode(p, convert_to_tensor=True)
        emb_r = embedder.encode(r, convert_to_tensor=True)
        return util.cos_sim(emb_p, emb_r).item()

    coherences = [coherence_score(p, g) for p, g in zip(prompts, generated)]
    coherence_mean = float(np.mean(coherences))
    print("Coherence:", coherence_mean)

    # ===========================
    # 5. Sentiment Score
    # ===========================
    sent_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    ).to(device)

    def sentiment_score(text):
        enc = sent_tok(text, return_tensors="pt", truncation=True).to(device)
        logits = sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        neg, neu, pos = probs[0]
        return (pos - neg).item()

    sentiments = [sentiment_score(g) for g in generated]
    sentiment_mean = float(np.mean(sentiments))
    print("Sentiment:", sentiment_mean)

    # ===========================
    # 7. BLEU Score
    # ===========================
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=generated, references=[[gt] for gt in ground_truth])
    bleu_score = bleu_results["bleu"]
    print("BLEU:", bleu_score)

    # ===========================
    # 8. ROUGE Score
    # ===========================
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=generated, references=ground_truth)
    rouge_l = rouge_results["rougeL"]
    print("ROUGE-L:", rouge_l)

    # ===========================
    # 9. BERTScore
    # ===========================
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(
        predictions=generated, references=ground_truth, lang="en"
    )
    bert_f1 = float(np.mean(bertscore_results["f1"]))
    print("BERTScore (F1):", bert_f1)

    # ===========================
    # 10. Diversity (Distinct-2)
    # ===========================
    def distinct_2(text):
        tokens = text.split()
        if len(tokens) < 2:
            return 0
        pairs = list(zip(tokens, tokens[1:]))
        return len(set(pairs)) / len(pairs)

    distinct_scores = [distinct_2(g) for g in generated]
    distinct_mean = float(np.mean(distinct_scores))
    print("Distinct-2:", distinct_mean)

    # ===========================
    # 11. Print some examples
    # ===========================
    print("\n================ EXAMPLES ================\n")
    for i in range(10):
        print(f"Prompt:        {prompts[i]}")
        print(f"Generated:     {generated[i]}")
        print("----------------------------------------\n")

    # ===========================
    # Return evaluation results
    # ===========================
    return {
        "coherence": coherence_mean,
        "sentiment": sentiment_mean,
        "bleu": bleu_score,
        "rougeL": rouge_l,
        "bertscore_f1": bert_f1,
        "distinct2": distinct_mean,
        "samples": list(zip(prompts[:10], generated[:10], ground_truth[:10]))
    }


"""
