"""逐步落盘 JSONL，异常时也保留已经完成的轨迹。"""
import json
from pathlib import Path
from utils.io import write_json


class ExperimentRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_step(self, record: dict) -> None:
        with (self.output_dir / "trajectory.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def save_result(self, result: dict) -> None:
        write_json(result, self.output_dir / "result.json")

    def record_model_call(self, record: dict) -> None:
        """每次调用立即落盘，保留 SPRING 中间节点及预算耗尽前的证据。"""
        with (self.output_dir / "model_calls.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
