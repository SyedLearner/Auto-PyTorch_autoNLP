# Import necessary modules and classes
from typing import Any, Dict, List
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import autoPyTorchTabularPreprocessingComponent
from autoPyTorch.utils.common import FitRequirement

# Define a class named BaseEncoder that extends the autoPyTorchTabularPreprocessingComponent class
class BaseEncoder(autoPyTorchTabularPreprocessingComponent):
    """
    Base class for encoder
    """

    # Constructor method
    def __init__(self) -> None:
        # Call the constructor of the parent class
        super().__init__()

        # Define fit requirements for the encoder
        self.add_fit_requirements([
            FitRequirement('categorical_columns', (List,), user_defined=True, dataset_property=True),
            FitRequirement('categories', (List,), user_defined=True, dataset_property=True)
        ])

    # Method to transform the input dictionary
    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the self into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """

        # Check if the encoder has been fitted by verifying if the 'numerical' and 'categorical' attributes are None
        if self.preprocessor['numerical'] is None and self.preprocessor['categorical'] is None:
            # Raise a ValueError if the encoder has not been fitted
            raise ValueError(f"Cannot call transform on {self.__class__.__name__} without fitting first.")

        # Update the 'X' dictionary by adding the encoder to it
        X.update({'encoder': self.preprocessor})
        
        # Return the updated 'X' dictionary
        return X
