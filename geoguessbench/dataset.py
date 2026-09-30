"""Validated, content-addressed datasets with graph/sequence/split leakage checks."""
import hashlib
import itertools
import json
import math
from pathlib import Path

from PIL import Image

from .geo import EARTH_RADIUS_KM, haversine
from .schema import Manifest


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("image paths must be relative, without traversal")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("image path escapes the dataset root")
    return target


class Dataset:
    def __init__(self, path: str | Path, *, verify_images: bool = True,
                 allow_fixture: bool = False, min_split_distance_km: float = 0.1):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.manifest = Manifest.model_validate_json(self.path.read_text(encoding="utf-8"))
        if self.manifest.fixture and not allow_fixture:
            raise ValueError("Synthetic test fixture refused; use real imagery for benchmark runs")
        self.nodes = {n.id: n for n in self.manifest.nodes}
        self.rounds = {r.id: r for r in self.manifest.rounds}
        if len(self.nodes) != len(self.manifest.nodes) or len(self.rounds) != len(self.manifest.rounds):
            raise ValueError("duplicate node or round ID")
        # Union graph neighbors, capture sequences AND duplicate image bytes.
        parent = {key: key for key in self.nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        sequences, images = {}, {}
        for node in self.nodes.values():
            safe_path(self.root, node.image)
            if node.panorama and node.heading is None:
                raise ValueError("panorama requires a known camera heading")
            if len(set(node.neighbors)) != len(node.neighbors):
                raise ValueError("duplicate graph edge")
            for other in node.neighbors:
                if other not in self.nodes or other == node.id:
                    raise ValueError("missing or self-referential graph edge")
                union(node.id, other)
            if node.sequence:
                if node.sequence in sequences:
                    union(node.id, sequences[node.sequence])
                sequences[node.sequence] = node.id
            if node.sha256 in images:
                union(node.id, images[node.sha256])
            images[node.sha256] = node.id
            if verify_images:
                image_path = safe_path(self.root, node.image)
                raw = image_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != node.sha256:
                    raise ValueError(f"image hash mismatch: {node.id}")
                with Image.open(image_path) as image:
                    if node.panorama and abs(image.width / image.height - 2) > 0.03:
                        raise ValueError("panoramas must be 2:1 equirectangular images")
                    image.verify()
        groups, components = {}, {}
        for rnd in self.rounds.values():
            if rnd.start not in self.nodes:
                raise ValueError("round refers to missing start node")
            component = find(rnd.start)
            for lookup, key in ((groups, rnd.group), (components, component)):
                if key in lookup and lookup[key] != rnd.split:
                    raise ValueError("split leakage: shared group, graph, sequence or image")
                lookup[key] = rnd.split
        self.component = {key: find(key) for key in self.nodes}
        # Stable cluster IDs for bootstrap: join group labels too, never split correlated rounds.
        group_node = {}
        for rnd in self.rounds.values():
            if rnd.group in group_node:
                union(rnd.start, group_node[rnd.group])
            group_node[rnd.group] = rnd.start
        self.cluster = {r.id: find(r.start) for r in self.rounds.values()}
        self._check_spatial_splits(components, min_split_distance_km)
        self.fingerprint = digest_json(self.manifest.model_dump())

    def _check_spatial_splits(self, components, distance):
        if distance <= 0 or len(set(components.values())) < 2:
            return
        # Hash unit-sphere XYZ cells; neighboring cells suffice within this chord radius.
        cell = 2 * math.sin(min(distance / EARTH_RADIUS_KM, math.pi) / 2)
        grid = {}
        for node in self.nodes.values():
            split = components.get(self.component[node.id])
            if split is None:
                continue
            p, q = math.radians(node.lat), math.radians(node.lon)
            xyz = (math.cos(p) * math.cos(q), math.cos(p) * math.sin(q), math.sin(p))
            key = tuple(math.floor(v / cell) for v in xyz)
            for offset in itertools.product((-1, 0, 1), repeat=3):
                nearby = tuple(x + y for x, y in zip(key, offset))
                for other, other_split in grid.get(nearby, []):
                    if split != other_split and haversine((node.lat, node.lon), (other.lat, other.lon)) < distance:
                        raise ValueError("split leakage: geographic buffer violated")
            grid.setdefault(key, []).append((node, split))

    def path_for(self, node):
        return safe_path(self.root, node.image)

    def validate_track(self, track: str, split: str):
        rounds = [r for r in self.rounds.values() if r.split == split]
        if not rounds:
            raise ValueError(f"no rounds in split {split}")
        if track == "nmpz":
            return
        reachable = set()
        for rnd in rounds:
            todo = [rnd.start]
            while todo:
                key = todo.pop()
                if key in reachable:
                    continue
                reachable.add(key)
                if track == "moving":
                    todo.extend(self.nodes[key].neighbors)
            if track == "moving" and not self.nodes[rnd.start].neighbors:
                raise ValueError("moving track requires a real connected route for every start")
        if any(not self.nodes[key].panorama for key in reachable):
            raise ValueError("NM/moving require real 360 panoramas; flat photos cannot be rotated")
