# train_resnet.py

import os
import math
import dotenv
import wandb
import tensorflow
import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.applications
import tensorflow.keras.optimizers
import wandb.integration.keras
import dataset_operations  # your data prep script with main_prep()

# --- Dynamic GPU/CPU Selection ---
gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth for GPUs: {e}")
else:
    print("No GPUs detected. Using CPU only.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- W&B Setup ---
dotenv.load_dotenv()
key = os.environ.get("WANDB_API_KEY")

if key:
    print("Logging into Weights & Biases...")
    wandb.login(key=key)
else:
    print("WARNING: WANDB_API_KEY not found in .env. Logging in anonymously.")
    wandb.login()

wandb.init(project="animal-faces-classification")

# --- Training Configuration ---
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

def build_and_train_model(train_generator, val_generator, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """
    Builds a ResNet50 Transfer Learning model, compiles it, and trains it
    while logging metrics to W&B.
    """
    print("\n--- Initializing Transfer Learning Pipeline (ResNet50) ---")

    IMG_HEIGHT, IMG_WIDTH = train_generator.target_size
    num_classes = len(train_generator.class_indices)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    print(f"Model Input Shape: {input_shape}")
    print(f"Number of Classes: {num_classes}")

    base_model = tensorflow.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    # Functional API to avoid graph errors
    inputs = tensorflow.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tensorflow.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tensorflow.keras.models.Model(inputs=inputs, outputs=outputs, name="ResNet50_Transfer")

    print("ResNet50 Base Model loaded and frozen. Top layers added.")

    wandb.config.update({
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_generator.batch_size,
        "model_type": "ResNet50_Transfer",
        "dataset": "AFHQ"
    })

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)

    print("\nStarting model training...")

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[
            wandb.integration.keras.WandbMetricsLogger(log_freq="epoch"),
            wandb.integration.keras.WandbModelCheckpoint("model_checkpoint.keras", save_weights_only=False)
        ]
    )

    print("\n--- Final Model Evaluation ---")
    final_loss, final_acc = model.evaluate(val_generator, steps=val_steps)
    print(f"Validation Loss: {final_loss:.4f}")
    print(f"Validation Accuracy: {final_acc:.4f}")

    wandb.log({"final_val_loss": final_loss, "final_val_accuracy": final_acc})
    return model

if __name__ == "__main__":
    try:
        print("--- Calling Data Preparation Pipeline from dataset_operations.py ---")
        train_gen, val_gen = dataset_operations.main_prep()

        print("\n--- Starting Model Training ---")
        final_model = build_and_train_model(train_gen, val_gen)

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")
    finally:
        if wandb.run:
            wandb.finish()
