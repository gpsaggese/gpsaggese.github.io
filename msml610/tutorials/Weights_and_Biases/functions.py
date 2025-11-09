import math
import numpy as np
import wandb
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.applications
import tensorflow.keras.optimizers
import wandb.integration.keras


def build_model(architecture, input_shape, num_classes, trainable_layers=50):
    """
    Builds a transfer learning model based on the specified architecture.
    Supported architectures: 'ResNet50', 'EfficientNetB0', 'MobileNetV2'
    Only last `trainable_layers` are trainable; the rest are frozen.
    """
    architecture = architecture.lower()
    
    if architecture == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif architecture == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif architecture == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Freeze all but last `trainable_layers`
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Build top layers
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_transfer")
    
    return model


def train_model(train_generator, val_generator, architecture="ResNet50", epochs=10, lr=0.0001, trainable_layers=20):
    """
    Trains a single model (homogeneous or heterogeneous) with W&B logging.
    """
    print(f"\n--- Training {architecture} model ---")
    
    IMG_HEIGHT, IMG_WIDTH = train_generator.target_size
    num_classes = len(train_generator.class_indices)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = build_model(architecture, input_shape, num_classes, trainable_layers=trainable_layers)

    # W&B configuration
    wandb.config.update({
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_generator.batch_size,
        "model_type": architecture,
        "dataset": "AFHQ"
    })

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)

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

    final_loss, final_acc = model.evaluate(val_generator, steps=val_steps)
    print(f"{architecture} Validation Accuracy: {final_acc:.4f}")
    wandb.log({f"{architecture}_val_accuracy": final_acc})

    return model


def ensemble_models(models, val_generator):
    """
    Evaluates an ensemble of models on the validation set (homogeneous or heterogeneous).
    Uses soft voting (averaging predictions) and logs W&B metrics.
    """
    val_steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
    print("\n--- Evaluating Ensemble ---")

    # Collect predictions
    preds_list = [model.predict(val_generator, steps=val_steps) for model in models]
    ensemble_preds = np.mean(preds_list, axis=0)
    ensemble_labels = val_generator.classes

    # Compute accuracy
    ensemble_acc = np.mean(np.argmax(ensemble_preds, axis=1) == ensemble_labels)
    print(f"Ensemble Validation Accuracy: {ensemble_acc:.4f}")
    wandb.log({"ensemble_val_accuracy": ensemble_acc})

    return ensemble_preds, ensemble_acc


def homogeneous_ensemble(train_generator, val_generator, num_models=3, epochs=10, lr=0.0001, trainable_layers=50):
    """
    Trains multiple homogeneous ResNet50 models and returns the ensemble results.
    """
    models = []
    for i in range(num_models):
        print(f"\n--- Training homogeneous model {i+1}/{num_models} ---")
        model = train_model(train_generator, val_generator, architecture="ResNet50", epochs=epochs, lr=lr, trainable_layers=trainable_layers)
        models.append(model)

    ensemble_preds, ensemble_acc = ensemble_models(models, val_generator)
    return models, ensemble_preds, ensemble_acc


def heterogeneous_ensemble(train_generator, val_generator, architectures, epochs=10, lr=0.0001, trainable_layers=50):
    """
    Trains multiple heterogeneous models and returns the ensemble results.
    `architectures` is a list of strings, e.g., ["ResNet50", "EfficientNetB0", "MobileNetV2"]
    """
    models = []
    for arch in architectures:
        model = train_model(train_generator, val_generator, architecture=arch, epochs=epochs, lr=lr, trainable_layers=trainable_layers)
        models.append(model)

    ensemble_preds, ensemble_acc = ensemble_models(models, val_generator)
    print(f"\nHeterogeneous Ensemble Accuracy: {ensemble_acc:.4f}")
    wandb.log({"hetero_ensemble_val_accuracy": ensemble_acc})

    return models, ensemble_preds, ensemble_acc
