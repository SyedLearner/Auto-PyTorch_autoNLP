import torch
from transformers import BertTokenizer, BertModel
from typing import Dict, Any, Optional, Union
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder import BaseEncoder

class BertEncoder(BaseEncoder):
    """
    Perform encoding on text features using a pretrained BERT model
    """
    def __init__(self, pari: str, max_length: int = 128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pari)
        self.model = BertModel.from_pretrained(pari)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def get_text_representation(self, text: str):
        """
        Function to get text representation using the fine-tuned BERT model.
        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: The BERT text representation.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        # Move the inputs to the device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get the BERT text representations
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            text_representations = outputs.last_hidden_state[:, 0, :].cpu()

        return text_representations

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEncoder:
        """
        Since BERT is already pretrained, we don't need to fit it to our data.
        The fit function simply checks the requirements and returns the instance of self.
        """
        self
    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'contextual_encoder',
            'name': 'contextual encoder',
            'handles_sparse': True
        }
   
