import time
import torch
import structlog
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from dataclasses import dataclass, field

logger = structlog.get_logger()

# Model name mapping
MODEL_REGISTRY = {
    "small": "gpt2",
    "large": "gpt2-medium",
}


@dataclass
class LoadedModel:
    model_type: str
    model_name: str
    model: object
    tokenizer: object
    device: str
    load_time_s: float
    param_count: int = 0


class ModelManager:
    """
    Loads and caches models in memory.
    Supports hot-reload and lazy loading.
    """

    def __init__(self):
        self._models: dict[str, LoadedModel] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("model_manager_init", device=self.device)

    def load_model(self, model_type: str) -> LoadedModel:
        if model_type in self._models:
            return self._models[model_type]

        model_name = MODEL_REGISTRY.get(model_type)
        if not model_name:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info("model_loading", model_type=model_type, model_name=model_name)
        start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        model.to(self.device)
        model.eval()

        elapsed = time.perf_counter() - start
        param_count = sum(p.numel() for p in model.parameters())

        loaded = LoadedModel(
            model_type=model_type,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            load_time_s=round(elapsed, 2),
            param_count=param_count,
        )
        self._models[model_type] = loaded

        logger.info(
            "model_loaded",
            model_type=model_type,
            model_name=model_name,
            device=self.device,
            load_time_s=round(elapsed, 2),
            params_M=round(param_count / 1e6, 1),
        )
        return loaded

    def get_model(self, model_type: str) -> Optional[LoadedModel]:
        return self._models.get(model_type)

    def hot_reload(self, model_type: str) -> LoadedModel:
        """Unload and reload a model without restarting the server."""
        if model_type in self._models:
            logger.info("model_hot_reload_start", model_type=model_type)
            old = self._models.pop(model_type)
            del old.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return self.load_model(model_type)

    def get_gpu_memory_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"available": False}
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 3),
        }

    def list_loaded(self) -> list:
        return [
            {
                "model_type": m.model_type,
                "model_name": m.model_name,
                "device": m.device,
                "load_time_s": m.load_time_s,
                "params_M": round(m.param_count / 1e6, 1),
            }
            for m in self._models.values()
        ]
