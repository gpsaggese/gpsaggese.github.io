from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


# custom transformer for amenities feature engineering
class AmenitiesEncoder(BaseEstimator, TransformerMixin):
    """
    custom transformer for encoding pipe delimited amenities text field
    
    converts amenities lists into one hot encoded binary columns
    enables amenities to be treated as categorical features
    """
 
    # initialize with MultiLabelBinarizer for handling amenities tokens
    def __init__(self):
        self.binarizer = MultiLabelBinarizer()

    # learn the set of amenities that occur in the training data
    def fit(self, X, y=None):
        amenities_lists = self._split(X)
        self.binarizer.fit(amenities_lists)
        return self

    # produce a binary matrix indicating if each amenity exists or not
    def transform(self, X):
        amenities_lists = self._split(X)
        return self.binarizer.transform(amenities_lists)

    # provide readable column names for amenity features
    def get_feature_names_out(self, input_features=None):
        return [f"amenity__{a}" for a in self.binarizer.classes_]

    #split comma separated amenities into token lists + handle empty cells
    @staticmethod
    def _split(series):
        return series.fillna("").apply(
            lambda x: [token.strip() for token in x.split(",") if token.strip()]
        )