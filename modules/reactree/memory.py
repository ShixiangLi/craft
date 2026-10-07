"""ReAcTree 的局部观测黑板和冻结的子目标经验检索。

Crafter 未公开世界坐标；黑板从相邻公开局部地图保守配准回合内坐标段，
不确定时保留历史观测而不假定移动成功。经验库只读；检索预算用完整
JSON 示例的字符数，而不是原论文的 tokenizer token 数。
"""

from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")
    return value


class WorkingMemory:
    """只利用公开地图维护地点；不确定的移动开始独立坐标段。

    spatial 配准只接受相邻帧中唯一一致的静态地形位移，不将动作意图
    当成实际移动。坐标从每个 segment 的玩家位置 (0, 0) 开始，无法
    对齐的 segment 不自动合并。last_seen 保留初版标签最后观测语义。
    """

    # Crafter 1.8.3：非 move 动作不改变玩家位置；move 可能被障碍或睡眠阻止。
    _MOVES = {"move_left": (-1, 0), "move_right": (1, 0),
              "move_up": (0, -1), "move_down": (0, 1)}
    _TERRAIN = frozenset({"water", "grass", "stone", "path", "sand", "tree",
                          "lava", "coal", "iron", "diamond", "table", "furnace"})
    _STATIONARY = _TERRAIN | {"plant", "fence"}
    # 箭可以在 move 期间摧毁 table/furnace，动态对象可以遮挡底层地形。
    _ALIGNMENT_LABELS = (_TERRAIN - {"table", "furnace"}) | {"boundary"}

    def __init__(self, mode="spatial", max_locations=8):
        if not isinstance(mode, str) or mode not in {"spatial", "last_seen"}:
            raise ValueError("working memory mode 必须是 spatial 或 last_seen")
        self.mode = mode
        self.max_locations = _integer(max_locations, "max_locations", minimum=1)
        self.reset()

    def reset(self):
        self._records = {}
        self._latest_step = None
        self._grid = None
        self._landmarks = {}
        self._last_seen = {}
        self._segment = 0
        self._position = (0, 0)
        self._localization = None

    def update(self, observation, action=None):
        if not isinstance(observation, dict):
            raise ValueError("working memory observation 必须是字典")
        if action is not None and (not isinstance(action, str) or not action):
            raise ValueError("working memory action 必须是非空字符串或 None")
        step = _integer(observation.get("step"), "observation.step")
        if self._latest_step is not None and step < self._latest_step:
            raise ValueError("working memory observation.step 不能倒退")
        grid = observation.get("local_map")
        if (not isinstance(grid, list) or not grid
                or not isinstance(grid[0], list) or not grid[0]):
            raise ValueError("observation.local_map 必须是非空的矩形标签列表")
        width = len(grid[0])
        if any(not isinstance(row, list) or len(row) != width for row in grid):
            raise ValueError("observation.local_map 必须是矩形标签列表")
        if any(not isinstance(label, str) or not label.strip()
               for row in grid for label in row):
            raise ValueError("observation.local_map 标签必须是非空字符串")
        grid = tuple(tuple(row) for row in grid)
        if step == self._latest_step:
            if grid != self._grid:
                raise ValueError("同一 observation.step 不能包含不同 local_map")
            return  # observe 后 act 会重复输入同一观测，不能再累计位移。
        if self.mode == "spatial":
            self._update_position(step, grid, action)
            self._update_landmarks(step, grid)
        cx, cy = width // 2, len(grid) // 2
        seen = {}
        for y, row in enumerate(grid):
            for x, label in enumerate(row):
                if label in {"player", "boundary"}:
                    continue
                seen.setdefault(label, []).append({"dx": x - cx, "dy": y - cy})
        for label, positions in seen.items():
            self._last_seen[label] = step
            if self.mode == "last_seen" or label not in self._STATIONARY:
                self._records[label] = {
                    "observed_step": step,
                    "positions_relative_to_player_then": positions,
                }
        self._latest_step = step
        self._grid = grid

    def _update_position(self, step, grid, action):
        reason, candidates = "initial_observation", []
        displacement = None
        if self._latest_step is not None:
            if step != self._latest_step + 1:
                reason = "missing_intermediate_observations"
            elif len(grid) != len(self._grid) or len(grid[0]) != len(self._grid[0]):
                reason = "view_shape_changed"
            elif action is None:
                reason = "missing_executed_action"
            elif action in self._MOVES:
                candidates = [self._alignment(grid, delta)
                              for delta in ((0, 0), self._MOVES[action])]
                compatible = [item for item in candidates if item["mismatches"] == 0]
                if len(compatible) == 1 and compatible[0]["matches"] > 0:
                    displacement = (compatible[0]["dx"], compatible[0]["dy"])
                    reason = "unique_static_map_alignment"
                else:
                    reason = "ambiguous_static_map_alignment" if compatible else "inconsistent_static_maps"
            elif not action.startswith("move_"):
                displacement = (0, 0)
                reason = "stationary_action"
            else:
                reason = "unknown_move_action"
            if displacement is None:
                self._segment += 1
                self._position = (0, 0)
            else:
                self._position = tuple(a + b for a, b in zip(self._position, displacement))
        self._localization = {
            "step": step, "segment": self._segment,
            "position_in_segment": {"x": self._position[0], "y": self._position[1]},
            "status": ("initialized" if self._latest_step is None else
                       "uncertain" if displacement is None else "tracked"),
            "reason": reason,
            "previous_segments_aligned": self._segment == 0,
            "candidate_displacements": candidates,
        }

    def _alignment(self, grid, displacement):
        dx, dy = displacement
        height, width = len(grid), len(grid[0])
        matches = mismatches = 0
        for y, row in enumerate(grid):
            for x, label in enumerate(row):
                # 同一个地块：当前相对坐标 + 玩家位移 = 上一帧相对坐标。
                ox, oy = x + dx, y + dy
                if not (0 <= ox < width and 0 <= oy < height):
                    continue
                previous = self._grid[oy][ox]
                if label not in self._ALIGNMENT_LABELS or previous not in self._ALIGNMENT_LABELS:
                    continue
                if previous == label:
                    matches += 1
                else:
                    mismatches += 1
        return {"dx": dx, "dy": dy, "matches": matches, "mismatches": mismatches}

    def _update_landmarks(self, step, grid):
        cx, cy = len(grid[0]) // 2, len(grid) // 2
        for y, row in enumerate(grid):
            for x, label in enumerate(row):
                px, py = self._position[0] + x - cx, self._position[1] + y - cy
                key = (self._segment, px, py)
                if label == "boundary":
                    self._landmarks.pop(key, None)
                if label not in self._STATIONARY:
                    # 动态对象（含 player）遮挡时不能断言原地标消失。
                    continue
                previous = self._landmarks.get(key)
                first_seen = (previous["first_seen_step"]
                              if previous and previous["label"] == label else step)
                self._landmarks[key] = {
                    "label": label, "segment": self._segment,
                    "position_in_segment": {"x": px, "y": py},
                    "first_seen_step": first_seen, "observed_step": step,
                    "position_relative_to_player_then": {"dx": x - cx, "dy": y - cy},
                }

    def targets(self):
        # 曾见资源已消失时仍可查询，返回 no_known_locations 而非让模型猜测查询非法。
        return sorted(set(self._last_seen) | set(self._records)
                      | {record["label"] for record in self._landmarks.values()})

    def recall(self, target, current_step):
        if not isinstance(target, str) or not target.strip():
            raise ValueError("recall target 必须是非空字符串")
        current_step = _integer(current_step, "current_step")
        if self.mode == "spatial":
            return self._recall_spatial(target, current_step)
        record = self._records.get(target)
        if record is None:
            return {"target": target, "status": "not_seen", "observed_step": None,
                    "age_steps": None, "positions_relative_to_player_then": []}
        if current_step < record["observed_step"]:
            raise ValueError("current_step 不能早于最后观测时间")
        return {"target": target, "status": "last_seen",
                "observed_step": record["observed_step"],
                "age_steps": current_step - record["observed_step"],
                "positions_relative_to_player_then":
                    deepcopy(record["positions_relative_to_player_then"])}

    def _recall_spatial(self, target, current_step):
        if self._latest_step is not None and current_step < self._latest_step:
            raise ValueError("current_step 不能早于最后观测时间")
        result = {"target": target, "status": "not_seen", "locations": [],
                  "localization": deepcopy(self._localization),
                  "total_known_locations": 0, "returned_count": 0, "truncated": False}
        if target not in self._STATIONARY:
            record = self._records.get(target)
            if record is not None:
                result.update(deepcopy(record), status="last_seen", kind="dynamic",
                              age_steps=current_step - record["observed_step"],
                              positions_relative_to_player_now=None,
                              uncertainty="Dynamic objects can move; historical sightings are not fixed landmarks.")
                positions = result["positions_relative_to_player_then"]
                result.update(total_known_locations=len(positions),
                              returned_count=min(len(positions), self.max_locations),
                              truncated=len(positions) > self.max_locations)
                result["positions_relative_to_player_then"] = positions[:self.max_locations]
            return result
        for record in self._landmarks.values():
            if record["label"] != target:
                continue
            location = deepcopy(record)
            location.pop("label")
            location["age_steps"] = current_step - record["observed_step"]
            location["visible_now"] = record["observed_step"] == current_step
            relative = None
            if record["segment"] != self._segment:
                uncertainty = "Different unaligned coordinate segment; current relative position is unknown."
            elif current_step != self._latest_step:
                uncertainty = "Player position has not been observed at the requested step."
            else:
                position = record["position_in_segment"]
                relative = {"dx": position["x"] - self._position[0],
                            "dy": position["y"] - self._position[1]}
                uncertainty = (None if location["visible_now"] else
                               "Location is aligned, but the landmark has not been observed this step.")
            location["position_relative_to_player_now"] = relative
            location["uncertainty"] = uncertainty
            result["locations"].append(location)
        def proximity(item):
            relative = item["position_relative_to_player_now"]
            # 未定位的旧坐标段只有时间排序，没有假造的“距离”。
            distance = abs(relative["dx"]) + abs(relative["dy"]) if relative is not None else 0
            return (relative is None, distance, -item["observed_step"],
                    item["segment"], item["position_in_segment"]["y"], item["position_in_segment"]["x"])

        result["locations"].sort(key=proximity)
        if result["locations"]:
            result["status"] = ("located" if any(item["position_relative_to_player_now"] is not None
                                                for item in result["locations"]) else "unlocalized")
            count = len(result["locations"])
            result.update(total_known_locations=count,
                          returned_count=min(count, self.max_locations),
                          truncated=count > self.max_locations)
            result["locations"] = result["locations"][:self.max_locations]
        elif target in self._last_seen:
            result.update(status="no_known_locations", observed_step=self._last_seen[target],
                          age_steps=current_step - self._last_seen[target])
        return result

    def snapshot(self):
        if self.mode == "last_seen":
            return deepcopy(self._records)
        return {"mode": self.mode, "localization": deepcopy(self._localization),
                "landmarks": deepcopy(list(self._landmarks.values())),
                "dynamic_last_seen": deepcopy(self._records)}


class EpisodicMemory:
    """读取成功回合导出的子目标经验；评估期间不增加或修改经验。"""

    STATES = ("expand", "success", "failure", "interrupted")

    def __init__(self, path=None, *, enabled=True,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 embedding_device="cpu", max_examples=3,
                 max_example_chars=20000, encoder=None):
        if not isinstance(enabled, bool):
            raise ValueError("episodic memory enabled 必须是布尔值")
        for name, value in (("embedding_model", embedding_model),
                            ("embedding_device", embedding_device)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 必须是非空字符串")
        self.enabled = enabled
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.max_examples = _integer(max_examples, "max_examples", minimum=1)
        self.max_example_chars = _integer(
            max_example_chars, "max_example_chars", minimum=1)
        self.path = Path(path) if path is not None else None
        self._encoder = encoder
        self._embeddings = None
        self.source_sha256 = None
        # 显式配置的路径即使禁用也校验，避免把拼写错误当作有效配置。
        self._records = tuple(self._load(self.path)) if self.path is not None else ()

    def __len__(self):
        return len(self._records)

    def metadata(self):
        return {"enabled": self.enabled, "records": len(self),
                "path": str(self.path) if self.path is not None else None,
                "sha256": self.source_sha256,
                "embedding_model": self.embedding_model,
                "embedding_device": self.embedding_device}

    def validate_evaluation_seeds(self, seeds):
        """在首个模型调用前拒绝未知训练 seed 或地图种子泄漏。"""
        if not self.enabled or not self._records:
            return
        evaluation = set(seeds)
        training = set()
        for index, record in enumerate(self._records, 1):
            source = record.get("source")
            seed = source.get("seed") if isinstance(source, dict) else None
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError(f"经验记录 {index} 缺少有效 source.seed，无法核对训练/评估隔离")
            training.add(seed)
        overlap = sorted(evaluation & training)
        if overlap:
            raise ValueError(f"ReAcTree 经验库训练种子与评估种子重叠: {overlap}；请选择独立种子")

    def _load(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"episodic memory JSONL 文件不存在: {path}")
        # 摘要和解析使用同一份字节，记录实际冻结的输入，而非后续可能变化的文件。
        raw = path.read_bytes()
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        records = []
        with StringIO(raw.decode("utf-8")) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    def reject_nonfinite(value):
                        raise ValueError(f"JSON 中不允许 {value}")

                    record = json.loads(line, parse_constant=reject_nonfinite)
                    if not isinstance(record, dict):
                        raise ValueError("每行必须是 JSON 对象")
                    for field in ("goal", "trajectory"):
                        if (not isinstance(record.get(field), str)
                                or not record[field].strip()):
                            raise ValueError(f"{field} 必须是非空字符串")
                    if record.get("state") not in self.STATES:
                        raise ValueError("state 必须是 success/failure/expand/interrupted")
                    if record.get("episode_success") is not True:
                        raise ValueError("episode_success 必须是 true（真实环境成功筛选标记）")
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"episodic memory {path} 第 {line_number} 行: {exc}") from exc
                records.append(record)
        return records

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "启用非空 episodic memory 需要 sentence-transformers；请执行 "
                    ".venv/bin/python -m pip install sentence-transformers。"
                    "首次使用 embedding_model 可能下载其权重。") from exc
            self._encoder = SentenceTransformer(
                self.embedding_model, device=self.embedding_device)
        return self._encoder

    @staticmethod
    def _encode(encoder, goals):
        vectors = np.asarray(encoder.encode(goals), dtype=np.float64)
        if (vectors.ndim != 2 or vectors.shape[0] != len(goals)
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
            raise ValueError("episodic memory encoder.encode 必须返回有限的二维向量矩阵")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

    def retrieve(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("retrieve goal 必须是非空字符串")
        if not self.enabled or not self._records:
            return []
        encoder = self._get_encoder()
        if self._embeddings is None:
            self._embeddings = self._encode(encoder, [r["goal"] for r in self._records])
        query = self._encode(encoder, [goal])
        if query.shape[1] != self._embeddings.shape[1]:
            raise ValueError("episodic memory 查询和经验向量维度不一致")
        similarities = np.clip(self._embeddings @ query[0], -1.0, 1.0)
        by_similarity = {}
        for index, score in enumerate(similarities):
            by_similarity.setdefault(float(score), []).append(index)
        ranked = []
        # 完全同分时按状态轮转；各状态内部保留库中的顺序，可复现实验。
        for score in sorted(by_similarity, reverse=True):
            groups = {state: [] for state in self.STATES}
            for index in by_similarity[score]:
                groups[self._records[index]["state"]].append(index)
            depth = 0
            while any(depth < len(group) for group in groups.values()):
                for state in self.STATES:
                    if depth < len(groups[state]):
                        ranked.append((score, groups[state][depth]))
                depth += 1
        selected = []
        for score, index in ranked:
            record = deepcopy(self._records[index])
            record["similarity"] = score
            candidate = selected + [record]
            if len(json.dumps(candidate, ensure_ascii=False, sort_keys=True)) > self.max_example_chars:
                continue
            selected.append(record)
            if len(selected) == self.max_examples:
                break
        return selected
