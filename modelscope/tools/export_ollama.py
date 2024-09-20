

class OllamaExporter:

    def to_ollama(self, model_id: str, template_name: str = None, gguf_file: str = None) -> str:
        """Export to ollama ModelFile

        Args:
            model_id: The model-id to use
            template_name: An extra template name to use
            gguf_file: An extra gguf_file path to use in the `FROM` field
        Returns:
            The ModelFile content
        """
        pass
