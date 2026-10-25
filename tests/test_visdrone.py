"""Safety tests for the VisDrone validation-data acquisition script."""
import zipfile

import pytest

from scripts.acquire_visdrone_val import DownloadForm, validate_zip
from scripts.acquire_visdrone_val import MIRROR_URL, ReleaseRedirect, validate_mirror_url, raw_manifest
import urllib.request


def make_archive(path, count=548, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(count):
            # Structure-only fixture, never supplied to a decoder or model.
            archive.writestr(f"VisDrone2019-DET-val/images/{index:04}.jpg", b"fixture")
            archive.writestr(f"VisDrone2019-DET-val/annotations/{index:04}.txt", b"")
        if extra:
            archive.writestr(extra, b"unsafe")


def test_expected_548_pairs(tmp_path):
    path = tmp_path / "val.zip"
    make_archive(path)
    assert validate_zip(path) == 548 * len(b"fixture")


def test_missing_pair_rejected(tmp_path):
    path = tmp_path / "val.zip"
    make_archive(path, count=547)
    with pytest.raises(ValueError, match="548"):
        validate_zip(path)


@pytest.mark.parametrize("extra", ["../escape", "C:/escape", "other/images/a.jpg"])
def test_unsafe_or_unexpected_member_rejected(tmp_path, extra):
    path = tmp_path / "val.zip"
    make_archive(path, extra=extra)
    with pytest.raises(ValueError):
        validate_zip(path)


def test_quota_html_not_an_archive(tmp_path):
    path = tmp_path / "val.zip"
    path.write_text("<html>Google Drive - Quota exceeded</html>")
    with pytest.raises(zipfile.BadZipFile):
        validate_zip(path)


def test_confirmation_form_is_parsed_not_executed():
    form = DownloadForm()
    form.feed('<form id="download-form" action="https://drive.usercontent.google.com/download">'
              '<input type="hidden" name="id" value="example"></form><script>throw 1</script>')
    assert form.fields == {"id": "example"}
    assert form.action == "https://drive.usercontent.google.com/download"


@pytest.mark.parametrize("url", [MIRROR_URL.replace("val.zip", "train.zip"), "http://github.com/anything", "https://evil.example/file", "https://github.com@evil.example/file", "https://release-assets.githubusercontent.com.evil.example/file"])
def test_mirror_url_restriction(url):
    with pytest.raises(ValueError):
        validate_mirror_url(url, initial=True)


def test_redirect_hosts_are_checked_before_request():
    request = urllib.request.Request(MIRROR_URL)
    with pytest.raises(ValueError):
        ReleaseRedirect().redirect_request(request, None, 302, "Found", {}, "https://evil.example/file")
    validate_mirror_url("https://release-assets.githubusercontent.com/asset?signature=example")
    validate_mirror_url(MIRROR_URL, initial=True)


def test_corrupt_zip_rejected(tmp_path):
    path = tmp_path / "broken.zip"
    make_archive(path)
    content = bytearray(path.read_bytes())
    position = content.find(b"fixture")
    content[position] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="CRC"):
        validate_zip(path)


def test_raw_manifest_checks_pixels_and_rows(tmp_path):
    import cv2
    import numpy as np
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    cv2.imwrite(str(tmp_path / "images/a.jpg"), np.zeros((12, 15, 3), dtype=np.uint8))
    (tmp_path / "annotations/a.txt").write_text("0,0,10,10,1,4,0,0\n")
    result = raw_manifest(tmp_path, expected_pairs=1)
    assert result["category_counts"] == {4: 1}
    assert result["files"][1]["width"] == 15
    assert result == raw_manifest(tmp_path, expected_pairs=1)


def test_existing_copy_provenance_does_not_claim_original_checksum(monkeypatch):
    from scripts import acquire_visdrone_val as acquisition
    monkeypatch.setattr(acquisition, "raw_manifest", lambda _: {"pairs": 548, "manifest_sha256": "example"})
    evidence = acquisition.existing_copy_provenance("unused")
    assert evidence["source"] == "validated-existing-copy"
    assert evidence["archive_sha256"] is None
    assert "no author-host verification" in evidence["provenance_limitation"]


@pytest.mark.parametrize("extra", ["VisDrone2019-DET-val/images/CON.jpg", "VisDrone2019-DET-val/images/a.jpg "])
def test_windows_unsafe_component_rejected(tmp_path, extra):
    path = tmp_path / "val.zip"
    make_archive(path, extra=extra)
    with pytest.raises(ValueError):
        validate_zip(path)
