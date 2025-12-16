import haiku as hk
from sentiment_utils import SentimentModel, clean_text

class SentimentService(hk.Module):
    """
    Haiku-based service wrapper for real-time sentiment prediction.
    """

    def __init__(self, model: SentimentModel):
        super().__init__()
        self.model = model

    def __call__(self, review_text: str) -> str:
        """
        Predict sentiment for a single review string.
        """
        cleaned = clean_text(review_text)
        prediction = self.model.predict([cleaned])[0]
        return prediction
