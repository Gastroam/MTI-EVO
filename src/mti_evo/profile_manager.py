
import json
import os
from typing import Dict, Any

class ProfileManager:
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = os.path.abspath(os.path.join(os.getcwd(), profiles_dir))
        
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Loads a profile JSON by name (e.g., 'gemma_12b_q2')."""
        if not profile_name.endswith(".json"):
            profile_name += ".json"
            
        path = os.path.join(self.profiles_dir, profile_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_profiles(self):
        """Returns available profiles."""
        if not os.path.exists(self.profiles_dir):
            return []
        return [f.replace(".json", "") for f in os.listdir(self.profiles_dir) if f.endswith(".json")]

    def get_model_config(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts LLMAdapter compatible config from profile."""
        return {
            "model_path": profile_data.get("model_path"),
            "n_ctx": profile_data.get("n_ctx", 2048),
            "n_gpu_layers": profile_data.get("n_gpu_layers", -1),
            "temperature": profile_data.get("temperature", 0.7),
            "top_k": profile_data.get("top_k", 40),
            "top_p": profile_data.get("top_p", 0.9),
            "stop": profile_data.get("stop", ["<|eot_id|>", "<start_of_turn>", "<end_of_turn>"])
        }
