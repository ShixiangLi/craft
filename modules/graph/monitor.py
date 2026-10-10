"""代码检测生存事件，提供去重、优先级、恢复阈值与可追溯证据。"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .evaluator import valid_grid


DEFAULT_THRESHOLDS = {key: {"trigger": 3, "recover": 6} for key in ("health", "drink", "food", "energy")}
PRIORITY = {"threat": 0, "health": 1, "drink": 2, "food": 3, "energy": 4}


@dataclass
class EmergencyEvent:
    id: str
    kind: str
    priority: int
    detected_at: int
    evidence: dict

    def to_dict(self):
        return asdict(self)


class EventMonitor:
    def __init__(self, thresholds=None, *, threat_radius=5, clear_steps=2):
        self.thresholds = deepcopy(DEFAULT_THRESHOLDS)
        if thresholds is not None:
            if not isinstance(thresholds, dict) or not set(thresholds) <= set(self.thresholds):
                raise ValueError("thresholds 仅支持 health、drink、food、energy")
            self.thresholds.update(deepcopy(thresholds))
        for key, values in self.thresholds.items():
            if (not isinstance(values, dict) or set(values) != {"trigger", "recover"}
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values())
                    or not 0 <= values["trigger"] < values["recover"] <= 9):
                raise ValueError(f"{key} 阈值必须满足 0 <= trigger < recover <= 9")
        if type(threat_radius) is not int or threat_radius < 1 or type(clear_steps) is not int or clear_steps < 1:
            raise ValueError("threat_radius 和 clear_steps 必须是正整数")
        self.threat_radius, self.clear_steps = threat_radius, clear_steps
        self.active = {}
        self.previous_health = self.last_step = None
        self.counter = self.quiet_steps = 0
        self.safety = {"no_immediate_threat": None}
        self.changes = []

    def update(self, observation: dict) -> list[EmergencyEvent]:
        """每个新 step 处理一次；重复观测不推进解除计数，也不重复产生事件。"""
        step = observation["step"]
        self.changes = []
        if self.last_step == step:
            return self.events()
        if self.last_step is not None and step < self.last_step:
            raise ValueError("不接受倒序观测；新回合请重新创建 EventMonitor")
        self.last_step = step
        inventory = observation.get("inventory", {})
        health = inventory.get("health")
        damaged = (type(health) in (int, float) and self.previous_health is not None
                   and health < self.previous_health)
        self.previous_health = health if type(health) in (int, float) else None
        grid = observation.get("local_map")
        nearby = []
        if valid_grid(grid):
            cx, cy = len(grid[0]) // 2, len(grid) // 2
            nearby = [{"label": cell, "dx": x-cx, "dy": y-cy}
                      for y, row in enumerate(grid) for x, cell in enumerate(row)
                      if (cell in ("zombie", "skeleton", "arrow") or cell.startswith("arrow-"))
                      and abs(x-cx) + abs(y-cy) <= self.threat_radius]
        # 只有警戒范围内可见的敌人/投射物触发威胁；掉血仅作为证据，
        # 不推断攻击来源。低生命及食物、饮水、精力不足由各自阈值处理。
        threat = bool(nearby)
        self.quiet_steps = 0 if threat or not valid_grid(grid) else self.quiet_steps + 1
        if threat:
            self._activate("threat", step, {"nearby": nearby, "damage_observed": damaged})
        elif valid_grid(grid) and self.quiet_steps >= self.clear_steps:
            self._resolve("threat", step)
        self.safety["no_immediate_threat"] = (None if not valid_grid(grid) else not (threat or "threat" in self.active))
        for key, threshold in self.thresholds.items():
            value = inventory.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                continue
            if value <= threshold["trigger"]:
                self._activate(key, step, {"value": value, **threshold})
            elif value >= threshold["recover"]:
                self._resolve(key, step)
        return self.events()

    def _activate(self, kind, step, evidence):
        if kind not in self.active:
            self.counter += 1
            self.active[kind] = EmergencyEvent(f"event_{self.counter:04d}", kind, PRIORITY[kind], step, evidence)
            self.changes.append({"type": "trigger", "step": step, "event": self.active[kind].to_dict()})
        else:
            self.active[kind].evidence = evidence

    def _resolve(self, kind, step):
        if kind in self.active:
            event = self.active.pop(kind)
            self.changes.append({"type": "resolved", "step": step, "event": event.to_dict()})

    def events(self):
        return sorted((deepcopy(event) for event in self.active.values()),
                      key=lambda event: (event.priority, event.detected_at, event.id))
