# Dataset preparation

## Sources

`fetch-osv` samples the official OSV-5M `test.csv` and resolves the Hugging Face revision to an immutable commit SHA. Required columns are `id`, `latitude`, `longitude`; country/region/sequence labels are retained when available. It reads the official `images/test/00.zip` through `04.zip` indexes using byte-range requests and extracts only selected members. Every selected ID must be found. Failure never silently changes the sampled test set. Static images are NMPZ inputs, not synthetic panoramas.

`fetch-mapillary` requires the user's own access token and actual panoramic sequence IDs. It gets frame IDs, fetches each frame's camera/location metadata, retains panoramas, orders by capture timestamp, and connects spatially close consecutive frames from the same sequence. The default first-40-frame cap is a bounded acquisition policy, **not globally random Mapillary sampling**. One connected start is selected per sequence. It does not infer paths between unrelated streets or reconstruct Street View's entire road graph. API availability, schemas, licenses and access policies can change; failures are explicit. Automated tests mock this interface, so a successful live acquisition remains a deployment prerequisite.

The local PNG copies have stripped metadata and content-addressed filenames. Manifests preserve provenance, source URLs and contributor attribution. Agent-facing views are re-encoded as JPEG without metadata. Check source licensing and attribution requirements before downloading, hosting or redistributing images; the software MIT license does not relicense them. Google Street View/GeoGuessr scraping and credential workarounds are not part of this project.

## CSV import

Required columns:

| Column | Meaning |
|---|---|
| id | Unique source-local node ID |
| image | Relative filename under `--images`; absolute paths/traversal forbidden |
| lat | Actual latitude, finite, -90 to 90 |
| lon | Actual longitude, finite, -180 to 180 |

Optional columns:

| Column | Meaning |
|---|---|
| panorama | `true`/`false` or `1`/`0`; default false |
| heading | Geographic heading in [0,360) at the **image center**; required for panoramas; flat images may omit |
| sequence | Capture sequence ID; correlated frames stay in one split |
| neighbors | Semicolon-separated real neighboring node IDs |
| start | Default true; false/0 means navigation-only node, not a round start |
| split | train/dev/test; default test |
| group | Correlation group; default sequence, otherwise node ID |
| country / region | Optional ground-truth strata, never model inputs |
| attribution / source_url | Source author and provenance |

Example **schema only**, not a usable dataset:

```csv
id,image,lat,lon,panorama,heading,sequence,neighbors,start,split,group,country,attribution,source_url
```

Supply real, verified coordinates and imagery. A panorama must be a genuine 2:1 equirectangular capture; changing a flat image's dimensions does not make it a valid panorama. The importer validates geometry/format but cannot establish that a curator's claimed GPS labels are true. Verify the provider's camera-heading convention and manually audit some rendered bearings before publishing a benchmark.

## Manifest

Generated `manifest.json` has:

- `schema_version=1`, dataset name/version/source/license, immutable provenance and explicit `fixture` flag.
- `nodes`: relative image file, SHA-256, true coordinates, panorama/heading/sequence, actual graph neighbors and attribution.
- `rounds`: a start node, split, correlation group, optional country/region, optional initial camera heading.

Run `geoguessbench validate --data <manifest> --track <track>` after preparing data. The normal CLI rejects `fixture:true`. Images, labels and raw source data belong on the **evaluator side**. Do not publish the hidden test manifest, serve raw image filenames, mount the dataset into an untrusted agent, or include coordinate-bearing IDs in prompts.

Validation rejects duplicate IDs, bad/missing hashes, absolute/traversal/symlink-escaping paths, missing/self/duplicate edges, broken starts, invalid coordinates, non-panoramic NM/Moving input, and disconnected Moving starts. It prevents graph/sequence/duplicate-image/group sharing across splits and enforces a default 100m cross-split spatial buffer (including across the antimeridian). The buffer is only a minimum safeguard: larger route/geographic holdouts may be needed to evaluate generalization. The API-level Dataset constructor supports a larger buffer for curator workflows; the standard CLI uses 100m.

Image duplicates merge into the same bootstrap cluster, even within a split; the validator does not silently delete images and alter sample counts. Very similar crops or visually overlapping sequences are not exhaustively detected. Verify perceptual near-duplicates, geographic distribution, urban/rural mix, capture dates and coverage externally. Freeze the actual chosen manifest before comparing models. New data versions or seeds must receive distinct results, not overwrite earlier benchmarks.

## Storage and outputs

The repository ignores `data/`, `private/`, `results/`, API secrets and SQLite files by default. Importers refuse to replace an existing manifest. Incomplete acquisitions may leave images or a pending manifest; inspect the error and use a new output directory or explicitly clean the incomplete one. Automatic fixture fallback is never performed.

For a public release, provide dataset cards and selection code, fixed versions and approved attribution/license material; expose held-out images only through the evaluator as allowed by their source terms. A live periodically refreshed benchmark is a different experiment and is **not** implemented as an automatic scheduler here.
