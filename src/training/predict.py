import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Expected logits with shape (n_samples, n_classes >= 2).")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite.")
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def predict_dataset(dataset,trainer) :
    """dataset must be tokenized"""
    if not len(dataset):
        raise ValueError("Cannot predict an empty dataset.")
    logits,labels,_ = trainer.predict(dataset)
  
    logits = np.asarray(logits)
    probabilities = softmax(logits)
    labels = np.asarray(labels)

    if labels.shape != (len(dataset),) or logits.shape[0] != len(dataset):
        raise ValueError("Predictions and labels dont align with dataset rows.")

    values = {
    "id": np.asarray(dataset["id"]), #if "id" in dataset.column_names else np.arange(len(dataset)),
    "prediction": logits.argmax(axis=1), 
    "label": labels, 
    "logits": logits, 
    "probabilities": probabilities,
    "gender_label": np.asarray(dataset["gender_label"]),
    "text": np.asarray(dataset["text"]),
    
    }

    return values