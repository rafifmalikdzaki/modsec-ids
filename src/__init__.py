"""
IDS Inference Package
"""

from .ids_inference import IDSInference, load_inference_engine, batch_predict_from_csv
from .model_architecture import HybridIDSModel, TextVectorizer

__version__ = '1.0.0'
__all__ = [
    'IDSInference',
    'load_inference_engine',
    'batch_predict_from_csv',
    'HybridIDSModel',
    'TextVectorizer'
]