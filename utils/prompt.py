"""提示词文件读取与 $变量 填充。"""
from pathlib import Path
from string import Template


def load_prompt(path: str | Path, variables: dict | None = None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return Template(text).substitute(variables) if variables is not None else text
