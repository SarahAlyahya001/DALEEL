"""Measure NFR-01 (processing time) and NFR-02 (memory) for a 60-second 1080p video.

    .venv\\Scripts\\python.exe scripts\\measure_runtime.py

The script builds the 60-second 1080p test video from the demonstration flight (a ping-pong loop of
demo/desert_search.mp4 with a matching SRT file), starts DALEEL with the Waitress server in a temporary
instance folder (your own database is not used), signs in as a temporary analyst, uploads the video
through the web form, follows the status until the mission is Complete, and prints one JSON summary.

Measured: the time from submitting the upload form until the status reports Complete, the
processing time the application records for the mission, and the peak working set of this process,
which contains the server, the detection models and a small HTTP client. Not measured: the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMO = PROJECT_ROOT / "demo" / "desert_search.mp4"


def build_test_video(source: Path, out_mp4: Path, out_srt: Path, seconds: int = 60) -> int:
    """Ping-pong loop of the demonstration flight to the requested length, with one SRT cue per second."""
    import cv2
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    height, width = frames[0].shape[:2]
    order, index, step = [], 0, 1
    while len(order) < int(round(seconds * fps)):
        order.append(index)
        if not 0 <= index + step < len(frames):
            step = -step
        index += step
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in order:
        writer.write(frames[index])
    writer.release()

    def stamp(t):
        ms = int(round(t * 1000))
        hours, rest = divmod(ms, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        secs, ms = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    with open(out_srt, "w", encoding="utf-8") as handle:
        for i in range(seconds):
            lat, lon = 21.4235 + i * 0.000065, 45.0712 + i * 0.000030
            handle.write(f"{i + 1}\n{stamp(i)} --> {stamp(i + 1)}\n<font size=\"28\">SrtCnt : {i + 1}, "
                         f"DiffTime : 1000ms\n[latitude: {lat:.6f}] [longitude: {lon:.6f}] "
                         "[rel_alt: 80.000 abs_alt: 912.000] </font>\n\n")
    return len(order)


def peak_memory_mb() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return round(counters.PeakWorkingSetSize / 2 ** 20, 1)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10, 1)


def hardware() -> dict:
    info = {"processor": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
            "system": platform.platform(), "python": platform.python_version()}
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        info["ram_gb"] = round(status.ullTotalPhys / 2 ** 30, 1)
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                info["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return info


class Client:
    def __init__(self, base):
        self.base = base
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def get(self, path):
        with self.opener.open(self.base + path, timeout=60) as response:
            return response.geturl(), response.read().decode("utf-8")

    def post(self, path, fields, files=None):
        if files:
            boundary = uuid.uuid4().hex
            body = b""
            for name, value in fields.items():
                body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
            for name, path_ in files.items():
                body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                         f"filename=\"{Path(path_).name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
                body += Path(path_).read_bytes() + b"\r\n"
            body += f"--{boundary}--\r\n".encode()
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        else:
            body = urllib.parse.urlencode(fields).encode()
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
        request = urllib.request.Request(self.base + path, data=body, headers=headers, method="POST")
        with self.opener.open(request, timeout=600) as response:
            return response.geturl(), response.read().decode("utf-8")


def csrf(html):
    return re.search(r'name="_csrf_token" value="([^"]+)"', html).group(1)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def measure(video: Path, srt: Path, instance: Path) -> dict:
    from waitress.server import create_server
    from daleel import create_app, initialize_database
    from daleel.models import User, db
    from daleel.pipeline.runner import initialize_processor

    app = create_app(str(instance))
    password = "Runtime measurement " + secrets.token_hex(8)
    with app.app_context():
        initialize_database()
        user = User(username="runtime-check", full_name="Runtime Check", email="runtime@daleel.invalid",
                    role="analyst", account_status="Active", password_initialized=True,
                    must_change_password=False)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
    processor = initialize_processor(app)
    port = free_port()
    server = create_server(app, host="127.0.0.1", port=port)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        client = Client(f"http://127.0.0.1:{port}")
        _, page = client.get("/login")
        client.post("/login", {"_csrf_token": csrf(page), "username": "runtime-check", "password": password})
        _, page = client.get("/upload")
        started = time.perf_counter()
        location, _ = client.post("/upload", {"_csrf_token": csrf(page)}, {"video": video, "srt": srt})
        mission_id = int(location.rstrip("/").rsplit("/", 1)[1])
        while True:
            _, body = client.get(f"/api/status/{mission_id}")
            state = json.loads(body)
            if state["status"] != "Processing":
                break
            time.sleep(1)
        wall = time.perf_counter() - started
    finally:
        # The server thread is a daemon and ends with the script; closing its sockets while a
        # keep-alive connection is open only produces shutdown noise.
        processor.shutdown()
    with app.app_context():
        from daleel.models import Mission
        mission = db.session.get(Mission, mission_id)
        observations = len(mission.observations)
        db.session.remove()
        db.engine.dispose()
    frames = {key: value for key, value in state.items() if key.endswith("_frames")}
    return {"status": state["status"], "submit_to_complete_seconds": round(wall, 1),
            "app_processing_seconds": round(state["elapsed_seconds"], 1), "frames_per_model": frames,
            "observations": observations, "peak_working_set_mb": peak_memory_mb()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, help="An existing MP4 instead of the generated test video")
    parser.add_argument("--srt", type=Path, help="The SRT file for --video")
    parser.add_argument("--build-video", nargs=2, type=Path, metavar=("MP4", "SRT"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.build_video:
        build_test_video(DEMO, *args.build_video)
        return 0
    with tempfile.TemporaryDirectory(prefix="daleel-runtime-") as work:
        work = Path(work)
        video, srt = args.video, args.srt
        if video is None:
            video, srt = work / "test_60s_1080p.mp4", work / "test_60s_1080p.srt"
            # A separate process, so that the frames held while building the video are not part of
            # the memory measured for DALEEL.
            subprocess.run([sys.executable, __file__, "--build-video", str(video), str(srt)], check=True)
        digest = hashlib.sha256(Path(video).read_bytes()).hexdigest()
        result = measure(Path(video), Path(srt), work / "instance")
    result.update(video_sha256=digest, hardware=hardware(),
                  nfr01_limit_seconds=300, nfr02_limit_mb=4096)
    result["nfr01_pass"] = result["status"] == "Complete" and result["submit_to_complete_seconds"] <= 300
    result["nfr02_pass"] = result["peak_working_set_mb"] <= 4096
    print(json.dumps(result, indent=1))
    return 0 if result["nfr01_pass"] and result["nfr02_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
