"""Explicit, author-linked validation-only acquisition (never imported by inference)."""
import hashlib
import json
import shutil
import stat
import urllib.parse
import urllib.request
import zipfile
import argparse
import math
from collections import Counter
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MIRROR_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip"
EXPECTED_BYTES = 81_638_851
RELEASE_HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
FILE_ID = "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
SOURCE = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"
PAGES = {
    "author_readme": "https://raw.githubusercontent.com/VisDrone/VisDrone-Dataset/master/README.md",
    "terms": "https://aiskyeye.com/data-protection/",
    "version_context": "https://aiskyeye.com/download/",
}


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120)


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields = None, {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
        if tag == "input" and attrs.get("type") == "hidden" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = set()
        images, annotations = set(), set()
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            unsafe_component = any(part.endswith((" ", ".")) or "\x00" in part
                                   or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
                                   for part in name.parts)
            if (name.is_absolute() or ".." in name.parts or "\\" in info.filename
                    or ":" in info.filename or stat.S_ISLNK(info.external_attr >> 16)
                    or unsafe_component or info.filename.casefold() in names):
                raise ValueError("Unsafe or duplicate ZIP member")
            names.add(info.filename.casefold())
            if name.parts[0] != "VisDrone2019-DET-val":
                raise ValueError("Unexpected archive root")
            if len(name.parts) == 3 and name.parts[1] == "images" and name.suffix.lower() == ".jpg":
                images.add(name.stem)
            elif len(name.parts) == 3 and name.parts[1] == "annotations" and name.suffix == ".txt":
                annotations.add(name.stem)
            elif not info.is_dir() and name.name != ".DS_Store":
                raise ValueError(f"Unexpected archive member: {info.filename}")
        if len(images) != 548 or images != annotations:
            raise ValueError("Expected exactly 548 image/annotation pairs; refusing extraction")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        return sum(i.file_size for i in archive.infolist())


def author_google_drive():
    destination = ROOT / "datasets/stage6a"
    downloads = destination / "downloads"
    if (destination / "VisDrone2019-DET-val").exists() or (downloads / "VisDrone2019-DET-val.zip").exists():
        raise SystemExit("Archive or extraction already exists; inspect rather than overwrite")
    if shutil.disk_usage(ROOT).free < 2_500_000_000:
        raise SystemExit("Insufficient free disk headroom")
    downloads.mkdir(parents=True, exist_ok=True)
    provenance = {"source_url": SOURCE, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "pages": {}, "license_ambiguity": "Terms copyright names VisDrone2021; official download page identifies DET as the same VisDrone2019 dataset. Academic noncommercial evaluation only; no redistribution of images."}
    for label, url in PAGES.items():
        with open_url(url) as response:
            content = response.read()
        page_path = downloads / f"{label}.txt"
        if page_path.exists() and page_path.read_bytes() != content:
            raise SystemExit("Previously retrieved official page changed; inspect before proceeding")
        if not page_path.exists():
            page_path.write_bytes(content)
        provenance["pages"][label] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    readme = (downloads / "author_readme.txt").read_text(encoding="utf-8")
    terms = (downloads / "terms.txt").read_text(encoding="utf-8")
    context = (downloads / "version_context.txt").read_text(encoding="utf-8")
    if FILE_ID not in readme or "academic use only" not in terms or "NonCommercial-ShareAlike" not in terms or "same as VisDrone2019" not in context:
        raise SystemExit("Official source/terms changed; manual provenance review required")
    archive_path = downloads / "VisDrone2019-DET-val.zip"
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    response = open_url(url)
    if "text/html" in response.headers.get("Content-Type", ""):
        form = DownloadForm()
        form.feed(response.read().decode("utf-8"))
        response.close()
        if not form.action or urllib.parse.urlparse(form.action).hostname != "drive.usercontent.google.com" or form.fields.get("id") != FILE_ID:
            raise SystemExit("Unexpected Google Drive confirmation; no alternate mirror used")
        url = form.action + "?" + urllib.parse.urlencode(form.fields)
        response = open_url(url)
    with response:
        provenance["resolved_url"] = response.url
        provenance["response_content_type"] = response.headers.get("Content-Type")
        with archive_path.open("xb") as output:
            shutil.copyfileobj(response, output)
    provenance["archive_sha256"] = sha256(archive_path)
    provenance["archive_bytes"] = archive_path.stat().st_size
    if not zipfile.is_zipfile(archive_path):
        provenance["status"] = "failed_non_zip_response"
        provenance["zip_crc_validated"] = False
        provenance["archive_sha256"] = None
        provenance["response_sha256"] = sha256(archive_path)
        (downloads / "failed_acquisition.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        raise SystemExit("Official source returned a non-ZIP response; no extraction or inference permitted")
    provenance["extracted_bytes"] = validate_zip(archive_path)
    provenance["zip_crc_validated"] = True
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    provenance["manifest"] = [
        {"path": p.relative_to(destination).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted((destination / "VisDrone2019-DET-val").rglob("*")) if p.is_file()
    ]
    (downloads / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items() if k != "manifest"}, indent=2))


def validate_mirror_url(url, initial=False):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in RELEASE_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or (initial and url != MIRROR_URL)):
        raise ValueError("Unapproved validation-asset URL or redirect")
    if parsed.hostname == "github.com" and url != MIRROR_URL:
        raise ValueError("Unexpected GitHub asset path")


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_mirror_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def raw_manifest(root, expected_pairs=548):
    """Validate original files; converted labels and OS metadata are not raw data."""
    import cv2
    root = Path(root)
    images = sorted((root / "images").glob("*.jpg"))
    annotations = sorted((root / "annotations").glob("*.txt"))
    if (len(images) != expected_pairs or len(annotations) != expected_pairs
            or {p.stem for p in images} != {p.stem for p in annotations}):
        raise ValueError("Expected exactly 548 unique raw image/annotation pairs")
    for folder, suffix in (("images", ".jpg"), ("annotations", ".txt")):
        for path in (root / folder).iterdir():
            if path.name != ".DS_Store" and (not path.is_file() or path.suffix != suffix or path.is_symlink()):
                raise ValueError("Unexpected raw data member")
    files, categories = [], Counter()
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None or frame.size == 0:
            raise ValueError(f"Unreadable image: {image.name}")
        height, width = frame.shape[:2]
        annotation = root / "annotations" / (image.stem + ".txt")
        rows = []
        for number, line in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), 1):
            fields = line.strip().rstrip(",").split(",")
            if len(fields) != 8:
                raise ValueError(f"Malformed annotation: {annotation.name}:{number}")
            values = [float(v) for v in fields]
            if not all(math.isfinite(v) and v == int(v) for v in values):
                raise ValueError("Noninteger/nonfinite annotation")
            x, y, w, h, score, category, truncation, occlusion = map(int, values)
            if (w <= 0 or h <= 0 or score not in (0, 1) or category not in range(12)
                    or truncation not in (0, 1, 2) or occlusion not in (0, 1, 2)
                    or min(x + w, width) <= max(x, 0) or min(y + h, height) <= max(y, 0)):
                raise ValueError(f"Invalid annotation geometry/category: {annotation.name}:{number}")
            categories[category] += 1
            rows.append(values)
        files.extend([
            {"path": image.relative_to(root).as_posix(), "bytes": image.stat().st_size,
             "sha256": sha256(image), "width": width, "height": height},
            {"path": annotation.relative_to(root).as_posix(), "bytes": annotation.stat().st_size,
             "sha256": sha256(annotation), "rows": len(rows)},
        ])
    files.sort(key=lambda entry: entry["path"])
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"pairs": len(images), "files": files, "category_counts": dict(sorted(categories.items())),
            "manifest_sha256": digest}


def mirror_acquisition():
    destination = ROOT / "datasets/stage6a/mirror"
    archive_path = destination / "VisDrone2019-DET-val.zip"
    if destination.exists():
        raise SystemExit("Mirror destination exists; refusing overwrite")
    destination.mkdir(parents=True)
    validate_mirror_url(MIRROR_URL, initial=True)
    provenance = {"source": "ultralytics-assets-mirror", "source_url": MIRROR_URL,
                  "description": "Ultralytics-maintained GitHub release mirror, not the authors' original host",
                  "repository": "ultralytics/yolov5", "release": "v1.0", "asset_id": 35550792,
                  "retrieval_started_utc": datetime.now(timezone.utc).isoformat(),
                  "installed_package": "8.2.48; VisDrone.yaml literal URL; ASSETS_URL is absent"}
    opener = urllib.request.build_opener(ReleaseRedirect())
    with opener.open(urllib.request.Request(MIRROR_URL, headers={"User-Agent": "DALEEL-academic-evaluation"}), timeout=120) as response:
        validate_mirror_url(response.url)
        if response.status != 200 or "html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Download returned an HTTP error or HTML")
        provenance["resolved_host"] = urllib.parse.urlsplit(response.url).hostname
        with archive_path.open("xb") as output:
            shutil.copyfileobj(response, output)
    provenance["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["archive_bytes"] = archive_path.stat().st_size
    provenance["archive_sha256"] = sha256(archive_path)
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if provenance["archive_bytes"] != EXPECTED_BYTES:
        raise ValueError("Archive size differs from approved 81,638,851 bytes; stopping")
    with archive_path.open("rb") as stream:
        if stream.read(4) != b"PK\x03\x04":
            raise ValueError("Invalid ZIP signature")
    provenance["extracted_bytes"] = validate_zip(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Extraction escaped destination")
        archive.extractall(destination)
    fresh = raw_manifest(destination / "VisDrone2019-DET-val")
    existing = raw_manifest(ROOT / "datasets/VisDrone2019-DET-val")
    provenance["zip_crc_validated"] = True
    provenance["manifest"] = fresh
    provenance["existing_manifest_sha256"] = existing["manifest_sha256"]
    left, right = ({f["path"]: f for f in m["files"]} for m in (fresh, existing))
    differences = [{"path": path, "mirror": left.get(path), "existing": right.get(path)}
                   for path in sorted(left.keys() | right.keys()) if left.get(path) != right.get(path)]
    provenance["comparison"] = {"identical": fresh == existing, "differences": differences,
                                "scope": "all raw images and annotations; excludes converted labels and OS metadata"}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items() if k != "manifest"}, indent=2))
    if fresh != existing:
        raise ValueError("Mirror and existing raw copy differ; no inference permitted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["author-google-drive", "ultralytics-assets-mirror", "validated-existing-copy"])
    args = parser.parse_args()
    if args.source == "author-google-drive":
        raise SystemExit("Author-host retries disabled; quota diagnostic is preserved")
    if args.source == "validated-existing-copy":
        print(json.dumps(existing_copy_provenance(ROOT / "datasets/VisDrone2019-DET-val"), indent=2))
    else:
        mirror_acquisition()


def existing_copy_provenance(root):
    return {"source": "validated-existing-copy", "archive_sha256": None,
            "provenance_limitation": "Previously obtained local copy; original archive checksum unavailable; no author-host verification",
            "manifest": raw_manifest(root)}


if __name__ == "__main__":
    main()
