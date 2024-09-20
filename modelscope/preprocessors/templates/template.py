from abc import abstractmethod
from typing import Dict, Any


class Template:

    TEMPLATE_REGEX = {}

    @staticmethod
    def load_by_model_id(model_id: str) -> 'Template':
        """Load a template by model-id

        Args:
            model_id: The model-id used to load the proper template

        Returns:
            The template instance
        """
        pass

    @staticmethod
    def load_by_template_name(template_name: str) -> 'Template':
        """Load a template by model-id

        Args:
            template_name: The template name used to load the proper template

        Returns:
            The template instance
        """
        pass

    @abstractmethod
    def encode(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Encode an input example

        Args:
            example: The input example

        Returns:
            The output example encoded
        """
        pass

