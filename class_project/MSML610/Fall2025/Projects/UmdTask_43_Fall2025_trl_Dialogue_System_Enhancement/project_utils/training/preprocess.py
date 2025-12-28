from datasets import load_dataset

def load_dataset_for_rl():
    """
    Load the dataset and return it directly (no duplication),
    since it already contains first -> second pairs.
    """

    dataset = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates")
    train = dataset["train"]

    # Optional: filter broken rows on the fly
    cleaned = train.filter(
        lambda x: isinstance(x["first"], str)
                  and isinstance(x["second"], str)
                  and len(x["first"]) > 3
                  and len(x["second"]) > 3
    )

    # -------------------------------
    # DEBUGGING SECTION
    # -------------------------------
    print("\n=== DEBUG: preprocess.py ===")
    
    # 1. Print type of returned dataset
    print("Type(cleaned):", type(cleaned))

    # 2. Print example row
    try:
        example = cleaned[0]
        print("cleaned[0] type:", type(example))
        print("cleaned[0] value:", example)
    except Exception as e:
        print("Error reading cleaned[0]:", e)

    # 3. Print type of batch slice
    try:
        batch = cleaned[0:5]
        print("cleaned[0:5] type:", type(batch))
        print("cleaned[0:5] value:", batch)
    except Exception as e:
        print("Error slicing cleaned[0:5]:", e)

    print("=== END DEBUG ===\n")
    # -------------------------------

    return cleaned



