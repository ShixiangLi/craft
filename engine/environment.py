"""Crafter 1.8.3 适配：局部语义地图 + 背包，而非像素观测。

私有字段访问集中在这里（reset 不返回 info）。语义地图只截取 LocalView
范围，但不模拟夜晚遮挡，属于结构化状态辅助观测。
"""
import crafter


class CrafterEnvironment:
    def __init__(self, config: dict):
        self.params = dict(config.get("params", {}))
        self.env = None
        self.steps = 0

    def reset(self, *, seed: int) -> dict:
        self.close()
        # 每次重新创建，使同一个 seed 不受 Crafter 内部回合计数影响。
        self.env = crafter.Env(**self.params, seed=seed, length=0)
        self.env.reset()
        self.steps = 0
        semantic = self.env._sem_view
        self.labels = {value: name or "boundary" for name, value in semantic._mat_ids.items()}
        self.labels.update({value: cls.__name__.lower() for cls, value in semantic._obj_ids.items()})
        return self._observation()

    def _observation(self) -> dict:
        player = self.env._player
        grid = self.env._sem_view()
        width, height = map(int, self.env._local_view._grid)
        px, py = map(int, player.pos)
        rows = []
        for dy in range(-(height // 2), height - height // 2):
            row = []
            for dx in range(-(width // 2), width - width // 2):
                x, y = px + dx, py + dy
                label = self.labels[int(grid[x, y])] if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] else "boundary"
                row.append(label)
            rows.append(row)
        return {
            "step": self.steps, "local_map": rows,
            "facing": list(map(int, player.facing)),
            "inventory": {k: int(v) for k, v in player.inventory.items()},
            "achievements": {k: int(v) for k, v in player.achievements.items()},
            "sleeping": bool(player.sleeping), "actions": list(self.env.action_names),
        }

    def step(self, action: int) -> dict:
        if type(action) is not int or not 0 <= action < len(self.env.action_names):
            raise ValueError(f"非法环境动作: {action!r}")
        _, reward, done, _ = self.env.step(action)
        self.steps += 1
        return {"observation": self._observation(), "reward": float(reward),
                "terminated": bool(done), "truncated": False}

    def close(self) -> None:
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if close:
                close()
            self.env = None


def create_environment(config: dict) -> CrafterEnvironment:
    if config.get("name") != "crafter":
        raise ValueError("第一版只支持 crafter")
    return CrafterEnvironment(config)
