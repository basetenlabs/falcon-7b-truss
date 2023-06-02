import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
from typing import Dict, List

CHECKPOINT = "tiiuae/falcon-40b"
DEFAULT_MAX_LENGTH = 128
DEFAULT_TOP_P = 0.95

class Model:
    def __init__(self, data_dir: str, config: Dict, secrets: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=CHECKPOINT,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )


    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                prompt = request.pop("prompt")
                max_length = request.pop("max_length", DEFAULT_MAX_LENGTH)
                top_p = request.pop("top_p", DEFAULT_TOP_P)
                encoded_prompt = self._tokenizer(prompt, return_tensors="pt").input_ids

                encoded_output = self._model.generate(
                    encoded_prompt,
                    max_length=max_length,
                    top_p=top_p,
                    **request
                )[0]
                decoded_output = self._tokenizer.decode(
                    encoded_output, skip_special_tokens=True
                )
                instance_response = {
                    "completion": decoded_output,
                    "prompt": prompt,
                }

                return {"status": "success", "data": instance_response, "message": None}
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}