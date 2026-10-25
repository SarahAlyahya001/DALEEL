<h1 align="center">DALEEL</h1>

<p align="center">Turning drone footage into reviewable evidence for desert search and rescue teams</p>

<p align="center">
  <img src="docs/images/results.png" width="900" alt="DALEEL mission results with observations and map">
</p>

DALEEL is a web application developed as a university graduation project to support search
and rescue teams in desert environments. A team uploads the video recorded by a drone together
with its SRT flight log. Two detection models, YOLOv8n and YOLOv8s, examine the footage for
people and vehicles. DALEEL shows every finding with its image, time, location and the models
that reported it, and lets the team confirm or reject each one before exporting a PDF report
and GPX map points.

## Preview

<details>
<summary>Show screens</summary>

| Sign in | Account management | New mission |
| --- | --- | --- |
| ![Sign in](docs/images/sign-in.png) | ![Account management](docs/images/accounts.png) | ![New mission](docs/images/new-mission.png) |

| Processing | Observation evidence | Mission history |
| --- | --- | --- |
| ![Processing](docs/images/processing.png) | ![Observation evidence](docs/images/evidence.png) | ![Mission history](docs/images/history.png) |

</details>

## Try DALEEL

Everything you need is in this repository, including a demonstration flight in
[`demo/`](demo/); the coordinates in its telemetry file are illustrative. Follow the steps in
order. The commands are for Windows PowerShell; notes for Linux and macOS are given where they
differ.

<details>
<summary><b>1. Install DALEEL</b></summary>

1. Install [Python 3.10](https://www.python.org/downloads/release/python-31011/). On Windows,
   tick **Add python.exe to PATH** in the installer.
2. On this page, choose **Code → Download ZIP** and extract the ZIP file (or clone the
   repository with Git).
3. Open a terminal in the project folder, the folder that contains `run.py` and
   `requirements.txt`. On Windows, right-click inside that folder and choose
   **Open in Terminal**.
4. Run these commands one at a time:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe scripts/setup_weights.py
   ```

   The second command downloads the required Python packages. The third downloads DALEEL's two
   trained detection models from the project's release page and checks each file. Together they
   take several minutes.

> On Linux or macOS, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

</details>

<details>
<summary><b>2. Start DALEEL</b></summary>

```powershell
.\.venv\Scripts\python.exe run.py
```

The first time only, the terminal shows a **temporary Administrator username and password**.
They are shown once, so copy them now. Then open **http://127.0.0.1:5000** in your browser.

Keep the terminal open while you use DALEEL; press **Ctrl+C** in it to stop.

**Starting DALEEL later:** open a terminal in the project folder and run the same command,
`.\.venv\Scripts\python.exe run.py`. Accounts, passwords and missions are kept.

</details>

<details>
<summary><b>3. Sign in as the Administrator</b></summary>

This account setup is needed only once.

1. Sign in with the temporary username and password from the terminal.
2. Choose a personal password of at least 15 characters, enter it twice, and choose
   **Save password**. The **Account management** page opens.

</details>

<details>
<summary><b>4. Create an analyst account</b></summary>

Analysts are the people who upload and review missions.

1. Under **Create account**, enter a full name, an email address and a username, and leave the
   role as **Analyst**. The email address is contact information only; DALEEL sends no email.
2. Choose **Create account**. The analyst's username and temporary password are shown together
   once; choose **Copy username and password** and give them to the analyst.
3. Sign out with the button at the bottom of the left sidebar.

</details>

<details>
<summary><b>5. Sign in as the analyst</b></summary>

1. Sign in with the analyst's username and temporary password.
2. Choose a personal password and **Save password**. The **New Mission** page opens.

From then on, analysts sign in at **http://127.0.0.1:5000** with their username and personal
password.

</details>

<details>
<summary><b>6. Upload the demonstration flight</b></summary>

1. Under **Upload video file**, select [`demo/desert_search.mp4`](demo/desert_search.mp4).
2. Under **Upload SRT file**, select [`demo/desert_search.srt`](demo/desert_search.srt).
3. Choose **Submit mission**.

The status page follows the analysis as it runs. On a laptop with an Intel Core i7-1355U processor, the demonstration flight took under one minute from submission to completion. When it shows **Complete**,
choose **View results**.

</details>

<details>
<summary><b>7. Review the observations</b></summary>

1. The list on the left shows every observation; the map shows where the drone was when each
   one was seen. Each observation is labeled **Both models** when YOLOv8n and YOLOv8s both
   reported the detection shown, or **One model** when one of them did, and shows each model's
   confidence (a dash means that model did not report it). Use the filter buttons to show one
   group. The label is information about the models; the decision is yours.
2. Select an observation to see its evidence image, time in the video and coordinates.
3. Choose **Confirm** or **Reject**. **Show on map** locates the observation on the map.

</details>

<details>
<summary><b>8. Export the results</b></summary>

- **Generate PDF** downloads a report of every observation and its review decision.
- **Export GPX** downloads the confirmed observations as map points that can be opened in
  mapping and navigation tools.

</details>

<details>
<summary><b>9. Return to earlier missions</b></summary>

**Mission History** in the sidebar lists your missions. Open one to see its results again, or
delete a completed mission after confirming.

</details>

To analyze your own flights, upload the MP4 video (up to 500 MB) together with the SRT file
recorded during the same flight. Many DJI drones can save this file next to each video when
video subtitles are enabled in the camera settings.

## Put DALEEL online

DALEEL is designed to run on a server under your own domain, served over HTTPS, so that teams
can use it from any browser. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) walks through it in six
steps with Docker.

## More information

| Topic | Document |
| --- | --- |
| Accounts, password recovery, data location and upgrading | [docs/ACCOUNT_MANAGEMENT.md](docs/ACCOUNT_MANAGEMENT.md) |
| Model files, settings and offline use | [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) |
| Training data, evaluation and results of the detection models | [docs/DATASET_EVALUATION.md](docs/DATASET_EVALUATION.md) |

To run the automated tests, install the test tools and run pytest:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

To run only the requirement-based unit and integration tests described in Chapter 6 of the
project report:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_requirements.py -v
```

To check the rendered interface in Microsoft Edge (layout at five screen widths, keyboard use,
accessibility, reduced motion and the results page), with Node.js installed:

```powershell
node tests/inspect_interface.mjs
```

To measure the processing time and memory for a 60-second 1080p video (NFR-01 and NFR-02):

```powershell
.\.venv\Scripts\python.exe scripts/measure_runtime.py
```

## Credits

- Object detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Maps: [Leaflet](https://leafletjs.com/) (BSD 2-Clause), with imagery from Esri World Imagery
  and OpenStreetMap.
- Training and evaluation data: the [HERIDAL](https://zenodo.org/records/5662351)
  search-and-rescue dataset and the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
  dataset, used to fine-tune and evaluate the two models; neither dataset is included in this
  repository. The splits are described in [docs/DATASET_EVALUATION.md](docs/DATASET_EVALUATION.md).
