"""Real-image ingestion. Nothing here fabricates images, labels or fallback scores."""
import csv
import hashlib
import io
import random
import re
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from PIL import Image, ImageOps

from .dataset import Dataset, safe_path
from .geo import haversine
from .schema import Manifest, Node, Round


def write_image(raw: bytes, root: Path) -> tuple[str, str]:
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        output = io.BytesIO()
        # Re-encode to lossless PNG once; evaluation views are fixed-quality JPEG.
        Image.frombytes("RGB", image.size, image.tobytes()).save(output, "PNG")
    content = output.getvalue()
    sha = hashlib.sha256(content).hexdigest()
    path = root / "images" / f"{sha}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.relative_to(root).as_posix(), sha


def save_manifest(root: Path, manifest: Manifest):
    path = root / "manifest.json"
    if path.exists():
        raise ValueError("manifest already exists; choose a new output directory/version")
    temporary = root / "manifest.pending.json"
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    Dataset(temporary)  # Fail closed on bad labels, missing images, broken graphs, split leakage.
    temporary.replace(path)
    return path


def import_csv(csv_path, image_root, out, *, name, license_name):
    """CSV columns: id,image,lat,lon; optional panorama,heading,sequence,neighbors,split,country."""
    root, image_root = Path(out), Path(image_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("output manifest already exists")
    nodes, rounds = [], []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image, sha = write_image(safe_path(image_root, row["image"]).read_bytes(), root)
            sequence = row.get("sequence", "")
            pano = row.get("panorama", "false").lower()
            if pano not in {"true", "false", "1", "0"}:
                raise ValueError("panorama must be true/false/1/0")
            nodes.append(Node(id=row["id"], image=image, sha256=sha, lat=float(row["lat"]),
                              lon=float(row["lon"]), panorama=pano in {"true", "1"},
                              heading=float(row["heading"]) if row.get("heading") else None, sequence=sequence,
                              neighbors=[x for x in row.get("neighbors", "").split(";") if x],
                              attribution=row.get("attribution", ""), source_url=row.get("source_url", "")))
            if row.get("start", "true").lower() not in {"false", "0"}:
                rounds.append(Round(id=row["id"], start=row["id"], split=row.get("split") or "test",
                                    group=row.get("group") or sequence or row["id"],
                                    country=row.get("country") or None, region=row.get("region") or None))
    return save_manifest(root, Manifest(name=name, version="1", source="local CSV import",
                         license=license_name, nodes=nodes, rounds=rounds,
                         provenance={"csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()}))


class RangeReader(io.RawIOBase):
    """Seekable HTTP ZIP reader. Requires byte ranges; never silently downloads a huge shard."""
    def __init__(self, client, url):
        self.client, self.url, self.position = client, url, 0
        initial = self._range(0, 0)
        self.length = initial[1]

    def _range(self, start, end):
        with self.client.stream("GET", self.url, headers={"Range": f"bytes={start}-{end}",
                                                        "Accept-Encoding": "identity"}) as response:
            if response.status_code != 206:
                raise ValueError("image host must support HTTP byte ranges (206); full-shard fallback disabled")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
            if not match or int(match[1]) != start or int(match[2]) != end:
                raise ValueError("invalid Content-Range")
            chunks, total = [], 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > end - start + 1:
                    raise ValueError("range response exceeded requested size")
                chunks.append(chunk)
            if total != end - start + 1:
                raise ValueError("truncated range response")
            return b"".join(chunks), int(match[3])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.length if whence == 2 else -1)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return self.position

    def read(self, size=-1):
        if self.position >= self.length or size == 0:
            return b""
        end = self.length - 1 if size < 0 else min(self.length - 1, self.position + size - 1)
        if end - self.position + 1 > 64 * 1024 * 1024:
            raise ValueError("single range read exceeds 64 MiB")
        content, _ = self._range(self.position, end)
        self.position = end + 1
        return content


def fetch_osv(out, *, limit=100, seed=42, revision="main"):
    """Seeded sample from the official TEST CSV; only selected ZIP members are transferred."""
    if limit < 1:
        raise ValueError("limit must be positive")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ValueError('install data support: pip install -e ".[data]"') from exc
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("output manifest already exists")
    pinned = HfApi().dataset_info("osv5m/osv5m", revision=revision).sha
    metadata = Path(hf_hub_download("osv5m/osv5m", "test.csv", repo_type="dataset", revision=pinned))
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"id", "latitude", "longitude"}.issubset(rows[0]):
        raise ValueError("upstream OSV-5M metadata schema changed")
    if limit > len(rows):
        raise ValueError("limit exceeds the test split size")
    chosen = random.Random(seed).sample(sorted(rows, key=lambda r: r["id"]), limit)
    missing = {r["id"]: r for r in chosen}
    if len(missing) != limit:
        raise ValueError("upstream test IDs are not unique")
    nodes, rounds = [], []
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for shard in range(5):
            if not missing:
                break
            url = f"https://huggingface.co/datasets/osv5m/osv5m/resolve/{pinned}/images/test/{shard:02d}.zip"
            with RangeReader(client, url) as remote, zipfile.ZipFile(remote) as archive:
                for member in archive.infolist():
                    image_id = Path(member.filename).stem
                    if image_id not in missing or member.is_dir():
                        continue
                    if member.file_size > 32 * 1024 * 1024:
                        raise ValueError("unexpectedly large source image")
                    row = missing.pop(image_id)
                    image, sha = write_image(archive.read(member), root)
                    nodes.append(Node(id=image_id, image=image, sha256=sha,
                                      lat=float(row["latitude"]), lon=float(row["longitude"]),
                                      attribution="OpenStreetView-5M / Mapillary contributors",
                                      source_url=f"https://www.mapillary.com/app/?pKey={image_id}"))
                    rounds.append(Round(id=image_id, start=image_id,
                                        group=row.get("sequence_id") or image_id,
                                        country=row.get("country") or None, region=row.get("region") or None))
    if missing:
        raise ValueError(f"{len(missing)} selected test images are missing upstream; no benchmark was created")
    return save_manifest(root, Manifest(name=f"OSV5M-test-{limit}-seed{seed}", version=pinned,
                         source="osv5m/osv5m official test split (subset)", license="CC-BY-SA-4.0 imagery",
                         nodes=sorted(nodes, key=lambda n: n.id), rounds=sorted(rounds, key=lambda r: r.id),
                         provenance={"hf_revision": pinned, "seed": seed, "selection": "uniform_without_replacement",
                                     "test_csv_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                                     "official_full_test": limit == len(rows)}))


def fetch_mapillary(out, sequences, token, *, max_frames=40, max_gap_m=75):
    """Build bounded sequence routes from real panoramic frames, without fictitious graph edges."""
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN is required")
    if not sequences or max_frames < 2 or max_gap_m <= 0:
        raise ValueError("provide sequence IDs, at least two frames and a positive gap limit")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("output manifest already exists")
    fields = "id,captured_at,computed_geometry,geometry,is_pano,compass_angle,computed_compass_angle,thumb_2048_url,creator"
    nodes, rounds = [], []
    with httpx.Client(base_url="https://graph.mapillary.com", headers={"Authorization": f"OAuth {token}"},
                      timeout=60) as api, httpx.Client(timeout=90, follow_redirects=True) as images:
        for sequence in dict.fromkeys(sequences):
            ids, after = [], None
            while len(ids) < max_frames:
                params = {"sequence_id": sequence, "limit": min(100, max_frames - len(ids))}
                if after:
                    params["after"] = after
                response = api.get("/image_ids", params=params)
                response.raise_for_status()
                body = response.json()
                page = [str(item["id"]) for item in body.get("data", [])]
                if not page:
                    break
                ids.extend(page)
                next_url = body.get("paging", {}).get("next")
                if not next_url:
                    break
                parsed = urlparse(next_url)
                if parsed.hostname != "graph.mapillary.com":
                    raise ValueError("unexpected pagination host")
                next_after = parse_qs(parsed.query).get("after", [None])[0]
                if not next_after or next_after == after:
                    break
                after = next_after
            entries = []
            for image_id in dict.fromkeys(ids[:max_frames]):
                response = api.get(f"/{image_id}", params={"fields": fields})
                response.raise_for_status()
                item = response.json()
                if not item.get("is_pano"):
                    continue
                geometry = item.get("computed_geometry") or item.get("geometry")
                angle = item.get("computed_compass_angle")
                if angle is None:
                    angle = item.get("compass_angle")
                if not geometry or angle is None:
                    raise ValueError("panorama lacks coordinates or camera heading")
                lon, lat = geometry["coordinates"][:2]
                response = images.get(item["thumb_2048_url"])
                response.raise_for_status()
                image, sha = write_image(response.content, root)
                creator = item.get("creator") or {}
                node = Node(id=image_id, image=image, sha256=sha, lat=lat, lon=lon, panorama=True,
                            heading=float(angle) % 360, sequence=sequence,
                            attribution=f"{creator.get('username', 'Mapillary contributor')} / Mapillary / CC-BY-SA",
                            source_url=f"https://www.mapillary.com/app/?pKey={image_id}")
                entries.append((int(item["captured_at"]), node))
            entries.sort(key=lambda entry: (entry[0], entry[1].id))
            route = [node for _, node in entries]
            for a, b in zip(route, route[1:]):
                if 0.1 <= haversine((a.lat, a.lon), (b.lat, b.lon)) * 1000 <= max_gap_m:
                    a.neighbors.append(b.id)
                    b.neighbors.append(a.id)
            starts = [node for node in route if node.neighbors]
            if not starts:
                raise ValueError("sequence has no connected panoramic route within the requested gap limit")
            start = starts[len(starts) // 2]
            rounds.append(Round(id=f"route-{len(rounds):05d}", start=start.id, group=sequence))
            nodes.extend(route)
    return save_manifest(root, Manifest(name="Mapillary-sequence-routes", version="1",
                         source="Mapillary Graph API panoramic sequences", license="CC-BY-SA imagery",
                         nodes=nodes, rounds=rounds,
                         provenance={"sequences": list(dict.fromkeys(sequences)), "max_frames": max_frames,
                                     "max_gap_m": max_gap_m, "navigation": "bounded same-sequence adjacent frames"}))
