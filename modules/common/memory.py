"""可选的公用记忆组件；算法可使用自定义记忆。"""


class Memory:
    def reset(self) -> None:
        raise NotImplementedError("记忆清理尚未实现")

    def add(self, record: dict) -> None:
        raise NotImplementedError("记忆写入尚未实现")

    def retrieve(self, query: str) -> list[dict]:
        raise NotImplementedError("记忆检索尚未实现")
