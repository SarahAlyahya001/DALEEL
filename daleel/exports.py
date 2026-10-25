"""Canonical, in-memory exports. No inference or persisted export artifacts."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from io import BytesIO
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .mission_media import EvidenceUnavailable, load_evidence
from .mission_storage import display_basename
from .presentation import saudi_datetime, target_label

GPX_NAMESPACE = "http://www.topografix.com/GPX/1/1"
ET.register_namespace("", GPX_NAMESPACE)


def coordinate_pair(latitude, longitude):
    """Validate again at the presentation boundary without mutating the model."""
    try:
        if latitude is None or longitude is None or isinstance(latitude, bool) or isinstance(longitude, bool):
            return None
        lat, lon = Decimal(str(latitude)), Decimal(str(longitude))
        if not lat.is_finite() or not lon.is_finite():
            return None
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return None
        return lat, lon
    except (InvalidOperation, ValueError, TypeError):
        return None


def _decimal_text(value):
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if value == 0 else text


def format_video_time(value):
    """Render stored float milliseconds without binary floating-point noise."""
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            return "Unavailable"
        return f"{_decimal_text(number.quantize(Decimal('0.001')))} ms"
    except (InvalidOperation, ValueError, TypeError):
        return "Unavailable"


def format_score(value):
    """Render model confidence consistently to three decimal places."""
    if value is None:
        return "Not available"
    try:
        number = Decimal(str(value))
        return format(number, ".3f") if number.is_finite() else "Not available"
    except (InvalidOperation, ValueError, TypeError):
        return "Not available"


def format_video_clock(value, full=False):
    """Render elapsed video time; full mode is HH:MM:SS.mmm, never wall time."""
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            return "Unavailable"
        total_ms = int(number.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, TypeError):
        return "Unavailable"
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    if full:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    second_text = f"{seconds:02d}"
    if milliseconds:
        second_text += f".{milliseconds:03d}".rstrip("0")
    return (f"{hours:02d}:{minutes:02d}:{second_text}" if hours
            else f"{minutes:02d}:{second_text}")


def ordered(observations):
    return sorted(observations, key=lambda item: (item.video_time_ms, item.observation_id))


def build_gpx(mission, observations):
    root = ET.Element(f"{{{GPX_NAMESPACE}}}gpx", version="1.1", creator="DALEEL")
    for observation_number, observation in enumerate(ordered(observations), start=1):
        coords = coordinate_pair(observation.latitude, observation.longitude)
        if observation.review_status != "Confirmed" or coords is None:
            continue
        lat, lon = coords
        # GPX longitudeType excludes +180; the canonical database remains unchanged.
        if lon == Decimal("180"):
            lon = Decimal("-180")
        lon_text = _decimal_text(lon)
        if not Decimal("-180") <= Decimal(lon_text) < Decimal("180"):
            continue
        point = ET.SubElement(root, f"{{{GPX_NAMESPACE}}}wpt",
                              lat=_decimal_text(lat), lon=lon_text)
        ET.SubElement(point, f"{{{GPX_NAMESPACE}}}name").text = f"Observation {observation_number}"
        ET.SubElement(point, f"{{{GPX_NAMESPACE}}}desc").text = (
            f"Mission {mission.mission_id}; Observation {observation_number}; "
            f"{observation.target_class}; {observation.agreement}; Confirmed; "
            f"Drone-Position Approximation: UAV position at representative video timestamp "
            f"{format_video_time(observation.video_time_ms)}, not an exact ground-target location.")
    return BytesIO(ET.tostring(root, encoding="utf-8", xml_declaration=True))


# ---------------------------------------------------------------- PDF mission summary
# Colors follow the web interface (static/style.css) so the report matches the application.
INK = colors.HexColor("#1C1A17")
INK_SOFT = colors.HexColor("#5A544D")
MUTED = colors.HexColor("#7A746B")
SAND = colors.HexColor("#F2EADC")
PANEL = colors.HexColor("#FBF8F1")
LINE = colors.HexColor("#D9D2C2")
GOLD = colors.HexColor("#C9A96E")
TONES = {
    # Model agreement uses neutral ink so that it is never read as a review decision.
    "Both models": ("#1C1A17", "#F2EADC"), "One model": ("#5A544D", "#FBF8F1"),
    "Confirmed": ("#4F7A4A", "#E4ECDF"), "Rejected": ("#8B3A2E", "#F1DDD7"),
    "Unreviewed": ("#B5832E", "#F4E8CF"),
}
REVIEW_TEXT = {"Unreviewed": "Unreviewed", "Confirmed": "Confirmed", "Rejected": "Rejected"}


def _style(name, size=9.5, leading=None, color=INK, bold=False, align=0):
    return ParagraphStyle(name, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                          leading=leading or size * 1.35, textColor=color, alignment=align)


BODY = _style("body")
SMALL = _style("small", 8, color=MUTED)
LABEL = _style("label", 7.5, color=MUTED, bold=True)
VALUE = _style("value", 10, bold=True)
SECTION = _style("section", 13, color=INK, bold=True)


def _text(value):
    return escape(str(value))


def _duration(seconds):
    if seconds is None:
        return "Not recorded"
    seconds = int(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} seconds"


def _when(value):
    if not isinstance(value, datetime):
        return "Not recorded"
    local = saudi_datetime(value)
    return local.strftime("%d %b %Y, ") + local.strftime("%I:%M %p").lstrip("0") + " AST"


def _percent(score):
    if score is None:
        return "—"  # The model did not report this detection.
    text = format_score(score)
    return text if text == "Not available" else f"{Decimal(text) * 100:.0f}%"


def _chip(label, size=8):
    fg, bg = TONES.get(label, ("#5A544D", "#F2EADC"))
    shown = REVIEW_TEXT.get(label, label)
    chip = Table([[Paragraph(_text(shown), _style("chip", size, color=colors.HexColor(fg), bold=True, align=1))]],
                 colWidths=[30 * mm])
    chip.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                              ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(fg)),
                              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return chip


def _colored(label):
    fg = TONES.get(label, ("#1C1A17",))[0]
    return f'<font color="{fg}"><b>{_text(REVIEW_TEXT.get(label, label))}</b></font>'


def _location(observation):
    coords = coordinate_pair(observation.latitude, observation.longitude)
    if coords is None:
        return "Location unavailable"
    return f"{coords[0]:.6f}, {coords[1]:.6f}"


def _evidence(mission, observation, instance_path, upload_root, width, height):
    """Scaled-down copy of the representative frame, or None when it cannot be read."""
    try:
        data, _ = load_evidence(mission, observation, instance_path, upload_root)
        with PILImage.open(BytesIO(data)) as picture:
            picture = picture.convert("RGB")
            picture.thumbnail((1400, 1400))
            compact = BytesIO()
            picture.save(compact, "JPEG", quality=82)
        compact.seek(0)
        image = Image(compact)
        scale = min(width / image.imageWidth, height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        return image
    except (EvidenceUnavailable, OSError, ValueError):
        return None


def _page_canvas(mission_label):
    """Canvas that adds the header, footer and 'Page X of Y' after layout is known."""
    class PageCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            for page_number, state in enumerate(self._pages, start=1):
                self.__dict__.update(state)
                self._decorate(page_number, len(self._pages))
                super().showPage()
            super().save()

        def _decorate(self, number, total):
            width, height = A4
            self.setFillColor(INK)
            self.setFont("Helvetica-Bold", 11)
            self.drawString(18 * mm, height - 13 * mm, "DALEEL")
            self.setFont("Helvetica", 8.5)
            self.setFillColor(MUTED)
            self.drawRightString(width - 18 * mm, height - 13 * mm, f"Mission summary  ·  {mission_label}")
            self.setStrokeColor(GOLD)
            self.setLineWidth(1.2)
            self.line(18 * mm, height - 16 * mm, width - 18 * mm, height - 16 * mm)
            self.setStrokeColor(LINE)
            self.setLineWidth(0.6)
            self.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
            self.setFont("Helvetica", 7.5)
            self.drawString(18 * mm, 10 * mm, "Locations are drone positions at the time of each observation, "
                                              "not exact ground positions.")
            self.drawRightString(width - 18 * mm, 10 * mm, f"Page {number} of {total}")
    return PageCanvas


def build_pdf(mission, observations, instance_path, upload_root, names=None, analyst=None):
    """Readable mission summary: overview, review counts, table of observations, evidence cards."""
    output = BytesIO()
    items = ordered(observations)
    names = names or {}
    video_name = names.get("video_name") or display_basename(mission.video_filename)
    srt_name = names.get("srt_name") or display_basename(mission.srt_filename)
    label = f"M-{mission.mission_id:03d}"
    usable = A4[0] - 36 * mm
    story = []

    # Title banner
    banner = Table([[Paragraph("Mission Summary", _style("t", 22, color=colors.white, bold=True))],
                    [Paragraph(_text(f"Mission {mission.mission_id}  ·  {video_name}"),
                               _style("s", 10.5, color=SAND))]], colWidths=[usable])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INK),
                                ("LEFTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (0, 0), 14),
                                ("BOTTOMPADDING", (0, -1), (-1, -1), 14), ("LINEBELOW", (0, -1), (-1, -1), 3, GOLD)]))
    story += [banner, Spacer(1, 6 * mm)]

    # Mission details (two label/value pairs per row)
    details = [("Mission", label), ("Status", mission.status),
               ("Video file", video_name), ("Telemetry file", srt_name),
               ("Analyst", analyst or "Not recorded"), ("Submitted", _when(mission.submitted_at)),
               ("Processing time", _duration(mission.elapsed_seconds)),
               ("Frames analyzed", f"{mission.yolov8n_frames} sampled frames by YOLOv8n, "
                                   f"{mission.yolov8s_frames} by YOLOv8s")]
    rows = []
    for left, right in zip(details[0::2], details[1::2]):
        rows.append([Paragraph(_text(left[0]).upper(), LABEL), Paragraph(_text(left[1]), BODY),
                     Paragraph(_text(right[0]).upper(), LABEL), Paragraph(_text(right[1]), BODY)])
    grid = Table(rows, colWidths=[28 * mm, usable / 2 - 28 * mm, 30 * mm, usable / 2 - 30 * mm])
    grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
                              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [grid, Spacer(1, 6 * mm)]

    # Review summary tiles
    counts = [("Observations", len(items), INK),
              ("Confirmed", sum(o.review_status == "Confirmed" for o in items), colors.HexColor("#4F7A4A")),
              ("Rejected", sum(o.review_status == "Rejected" for o in items), colors.HexColor("#8B3A2E")),
              ("Unreviewed", sum(o.review_status == "Unreviewed" for o in items), colors.HexColor("#B5832E"))]
    tiles = Table([[Paragraph(f'<font size="18" color="{c.hexval().replace("0x", "#")}"><b>{n}</b></font>'
                              f'<br/><font size="7.5" color="#7A746B"><b>{_text(t.upper())}</b></font>',
                              _style("tile", 9, 22, align=1)) for t, n, c in counts]],
                  colWidths=[usable / 4] * 4)
    tiles.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                               ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                               ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [tiles, Spacer(1, 4 * mm)]
    note = Table([[Paragraph("<b>How to read this report.</b> Each observation is a person or vehicle found in "
                             "the video. <b>Both models</b> means that YOLOv8n and YOLOv8s both reported the "
                             "detection shown; <b>One model</b> means that one of them did. Model agreement is "
                             "supporting information; only the analyst's review confirms or rejects an "
                             "observation. The location is the drone's "
                             "position when the observation was seen (Drone-Position Approximation).",
                             _style("note", 8.5, color=INK_SOFT))]], colWidths=[usable])
    note.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SAND), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [note, Spacer(1, 7 * mm)]

    if not items:
        story.append(Paragraph("No observations were found in this mission.", SECTION))
    else:
        # Overview table
        story.append(Paragraph("Observations at a glance", SECTION))
        story.append(Spacer(1, 3 * mm))
        head = ["#", "Target", "Model agreement", "Video time", "Drone location", "Review"]
        table_rows = [[Paragraph(f"<b>{h}</b>", _style("h", 8, color=colors.white)) for h in head]]
        for number, item in enumerate(items, start=1):
            table_rows.append([Paragraph(str(number), BODY), Paragraph(_text(target_label(item.target_class)), BODY),
                               Paragraph(_colored(item.agreement), _style("c", 8.5)),
                               Paragraph(format_video_clock(item.video_time_ms, full=True), _style("m", 8.5)),
                               Paragraph(_text(_location(item)), _style("m2", 8.5)),
                               Paragraph(_colored(item.review_status), _style("c2", 8.5))])
        overview = Table(table_rows, colWidths=[10 * mm, 22 * mm, 30 * mm, 28 * mm, usable - 120 * mm, 30 * mm],
                         repeatRows=1)
        overview.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK),
                                      ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
                                      ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
                                      ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story += [overview, Spacer(1, 8 * mm), Paragraph("Observation details", SECTION), Spacer(1, 3 * mm)]

        # One card per observation
        for number, item in enumerate(items, start=1):
            title = Table([[Paragraph(_text(f"Observation {number}  ·  {target_label(item.target_class)}"),
                                      _style("ct", 11.5, bold=True)), _chip(item.review_status)]],
                          colWidths=[usable - 34 * mm, 34 * mm])
            title.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                       ("LINEBELOW", (0, 0), (-1, -1), 0.8, GOLD)]))
            facts = [("Review status", REVIEW_TEXT.get(item.review_status, item.review_status)),
                     ("Model agreement", item.agreement),
                     ("YOLOv8n confidence", _percent(item.yolov8n_score)),
                     ("YOLOv8s confidence", _percent(item.yolov8s_score)),
                     ("Video time", format_video_clock(item.video_time_ms, full=True)),
                     ("Seen in", f"{item.frame_count} sampled frame{'s' if item.frame_count != 1 else ''} "
                                 f"(frames {item.first_frame_index} to {item.last_frame_index})"),
                     ("Drone location", _location(item))]
            fact_table = Table([[Paragraph(_text(k), LABEL), Paragraph(_text(v), _style("fv", 9))] for k, v in facts],
                               colWidths=[31 * mm, usable - 76 * mm - 31 * mm - 12])
            fact_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            picture = _evidence(mission, item, instance_path, upload_root, 72 * mm, 58 * mm) or Paragraph(
                "Representative evidence unavailable (missing, invalid, or corrupt image).", SMALL)
            card = Table([[picture, fact_table]], colWidths=[76 * mm, usable - 76 * mm])
            card.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                                      ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                                      ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story.append(KeepTogether([title, Spacer(1, 2 * mm), card, Spacer(1, 7 * mm)]))

    generated = _when(datetime.now(timezone.utc).replace(tzinfo=None))
    story.append(Paragraph(_text(f"Generated by DALEEL on {generated}."), SMALL))
    SimpleDocTemplate(output, pagesize=A4, title=f"DALEEL Mission {mission.mission_id}", author="DALEEL",
                      leftMargin=18 * mm, rightMargin=18 * mm, topMargin=22 * mm, bottomMargin=20 * mm
                      ).build(story, canvasmaker=_page_canvas(label))
    output.seek(0)
    return output
