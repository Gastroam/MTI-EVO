from .base import BaseEngine, LLMResponse

class APIEngine(BaseEngine):
    """Engine for External APIs (OpenAI/Anthropic). Stub."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.backend_name = "api"
        self.api_key = config.get("api_key", "")

    def load_model(self):
        print(f"[APIEngine] ☁️  Ready to connect to API (Key: {self.api_key[:4]}...)")

    def infer(self, prompt: str, max_tokens: int = 1024, stop: list = None, **kwargs) -> LLMResponse:
        return LLMResponse("[APIEngine] Not Implemented Yet", 0, 0, 0.0)

    def embed(self, text: str):
        return []

    def unload(self):
        pass
