"""Pinned MLX vision-language checkpoints used for the equine radiograph comparison."""
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    model_id: str
    revision: str
    weights: dict[str, str]

    def verify(self, folder: Path):
        if {p.name for p in folder.glob("*.safetensors")} != set(self.weights):
            raise ValueError(f"Unexpected weight files for {self.model_id}")
        for filename, expected in self.weights.items():
            with (folder / filename).open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError(f"SHA-256 mismatch for {filename}")


CHECKPOINTS = {
    "2b": Checkpoint("mlx-community/Qwen3-VL-2B-Instruct-4bit", "9c4f5209e57b31f4b9dfba735de3fb983739c9cc", {
        "model.safetensors": "4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb",
    }),
    "7b": Checkpoint("mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b", {
        "model-00001-of-00002.safetensors": "f80072ac0e82a2ace0740eab995f6f9ae2623adb55e6906e21c2f8edca537803",
        "model-00002-of-00002.safetensors": "016bd9d3596e9b824f5a527875e5830414552c950eee794e98efb44c1cd8dc6a",
    }),
    "8b": Checkpoint("mlx-community/Qwen3-VL-8B-Instruct-4bit", "defcdea7cc7a4b0858fea563cbbce171d328e457", {
        "model-00001-of-00002.safetensors": "7c637158b2203e321d83596d3661f33b7b98a72beddfaaaa0eddc512acbdd1fb",
        "model-00002-of-00002.safetensors": "77190cd1dcf244522869bf923558340112b26d7db2ef3692f88407dd9b33c25d",
    }),
    "mimo-7b": Checkpoint("mlx-community/MiMo-VL-7B-RL-4bit", "c118e2b92c54ddee5bf1f381a8b22b31f5faf3c2", {
        "model-00001-of-00002.safetensors": "d3109927170fa75ac031a774f0364c66638f2afd746fdb89847a70a7e13628ab",
        "model-00002-of-00002.safetensors": "ea3d8b0833f52cd94cb8ccdc0aea486c17cd1944f94ef254adb52ec3e18920ba",
    }),
    "qwen35-9b": Checkpoint("mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53", {
        "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
        "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
    }),
}
