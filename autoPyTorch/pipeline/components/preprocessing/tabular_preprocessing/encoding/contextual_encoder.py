
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, Any, Optional, Union
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder import BaseEncoder
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
class ContextualEncoder(BaseEncoder):
    model = None
    """
    Perform encoding on text features using a pretrained BERT model
    """
    def __init__(self,bert_model_path : str, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.max_length = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ContextualEncoder.model is None:
            # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # ContextualEncoder.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
            ContextualEncoder.model = BertForSequenceClassification.from_pretrained(bert_model_path,num_labels=2).to(self.device)
        self.model = ContextualEncoder.model
        self.model.eval()

    def get_text_representation(self, text: str):
        """
        Function to get text representation using the fine-tuned BERT model.
        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: The BERT text representation.
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            text_representations = outputs.last_hidden_state[:, 0, :].cpu()

        return text_representations

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        texts = X['sentence'].astype(str).tolist()
        encoded_texts = self.get_text_representation(texts)
        
        X.update({'encoder': {'categorical': encoded_texts}})
        return X
    
    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:
        """
        Since BERT is already pretrained, we don't need to fit it to our data.
        The fit function simply checks the requirements and returns the instance of self.
        """
        return self
    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ContextualEncoder',
            'name': 'Contextual Encoder',
            'handles_sparse': True
        }
