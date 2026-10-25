from flask import (Blueprint, abort, current_app, flash, jsonify, redirect,
                   render_template, request, url_for, send_file)
from io import BytesIO

from .auth import analyst_required, current_user, login_destination, login_required
from .models import Mission, Observation, db
from .exports import build_gpx, build_pdf, coordinate_pair, format_video_clock
from .mission_media import EvidenceUnavailable, load_evidence
from .mission_storage import archive_mission, original_names
from .submission import SubmissionStorageError, SubmissionValidationError, validated_submission
from .pipeline.runner import ProcessingUnavailable, start_processing

bp = Blueprint("main", __name__)


def _owned_mission_or_404(mission_id: int) -> Mission:
    """Analysts may access only their own mission records."""
    mission = db.session.get(Mission, mission_id)
    if mission is None or mission.analyst_id != current_user().user_id:
        abort(404)
    return mission


@bp.route("/")
@login_required
def index():
    return redirect(url_for(login_destination(current_user())))


@bp.route("/upload", methods=["GET", "POST"])
@analyst_required
def upload():
    if request.method == "POST":
        mission_id = None
        try:
            with validated_submission(
                request.files, current_app.config["UPLOAD_DIR"],
                current_app.config["VIDEO_MAX_BYTES"], current_app.config["MAX_CONTENT_LENGTH"],
            ) as staging:
                mission_id = start_processing(current_app._get_current_object(), current_user().user_id, staging)
        except SubmissionValidationError as exc:
            return render_template("upload.html", error=str(exc)), 400
        except ProcessingUnavailable as exc:
            return render_template("upload.html", error=str(exc)), 503
        except SubmissionStorageError as exc:
            if mission_id is not None:
                return render_template("upload.html", error="The mission was accepted, but temporary upload cleanup failed. Check mission history."), 500
            return render_template("upload.html", error=str(exc)), 500
        except Exception:
            current_app.logger.error("Mission admission failed; no processing success is claimed.")
            return render_template("upload.html", error="Mission submission failed. No mission was created."), 500
        return redirect(url_for("main.status", mission_id=mission_id), code=303)
    return render_template("upload.html")


@bp.route("/status/<int:mission_id>")
@analyst_required
def status(mission_id):
    mission = _owned_mission_or_404(mission_id)
    return render_template("status.html", mission=mission, state=_status_state(mission))


def _status_state(mission):
    state = dict(status=mission.status, yolov8n_frames=mission.yolov8n_frames,
                 yolov8s_frames=mission.yolov8s_frames, elapsed_seconds=mission.elapsed_seconds)
    processor = current_app.extensions.get("daleel_processor")
    if mission.status == "Processing" and processor is not None:
        state.update(processor.snapshot(mission.mission_id))
    return state


@bp.route("/api/status/<int:mission_id>")
@analyst_required
def api_status(mission_id):
    return jsonify(_status_state(_owned_mission_or_404(mission_id)))


@bp.route("/results/<int:mission_id>")
@analyst_required
def results(mission_id):
    mission = _owned_mission_or_404(mission_id)
    if mission.status != "Complete":
        return redirect(url_for("main.status", mission_id=mission_id))
    observations = _observations(mission)
    review_labels = {"Unreviewed": "Awaiting review", "Confirmed": "Confirmed",
                     "Rejected": "Rejected"}
    review_counts = {state: sum(item.review_status == state for item in observations)
                     for state in review_labels}
    markers = []
    numbered_observations = list(enumerate(observations, start=1))
    for observation_number, observation in numbered_observations:
        coords = coordinate_pair(observation.latitude, observation.longitude)
        if coords is not None:
            markers.append(dict(observation_number=observation_number,
                                anchor=f"observation-{observation.observation_id}",
                                latitude=float(coords[0]), longitude=float(coords[1]),
                                target_class=observation.target_class,
                                review_status=observation.review_status,
                                review_label=review_labels[observation.review_status]))
    return render_template("results.html", mission=mission, observations=observations,
                           numbered_observations=numbered_observations,
                           map_payload=markers, coordinate_pair=coordinate_pair,
                           format_video_clock=format_video_clock,
                           review_counts=review_counts, review_labels=review_labels)


@bp.route("/frame/<int:observation_id>")
@analyst_required
def frame(observation_id):
    observation = _owned_observation(observation_id)
    if observation.mission.status != "Complete":
        abort(404)
    try:
        data, mime = load_evidence(observation.mission, observation,
                                   current_app.instance_path, current_app.config["UPLOAD_DIR"])
    except EvidenceUnavailable:
        abort(404)
    return send_file(BytesIO(data), mimetype=mime)


@bp.route("/export/<int:mission_id>/gpx")
@analyst_required
def export_gpx(mission_id):
    mission = _owned_mission_or_404(mission_id)
    if mission.status != "Complete":
        return redirect(url_for("main.status", mission_id=mission_id))
    return send_file(build_gpx(mission, _observations(mission)), mimetype="application/gpx+xml",
                     as_attachment=True, download_name=f"mission-{mission_id}.gpx")


@bp.route("/export/<int:mission_id>/pdf")
@analyst_required
def export_pdf(mission_id):
    mission = _owned_mission_or_404(mission_id)
    if mission.status != "Complete":
        return redirect(url_for("main.status", mission_id=mission_id))
    return send_file(build_pdf(mission, _observations(mission), current_app.instance_path,
                               current_app.config["UPLOAD_DIR"], names=original_names(current_app, mission),
                               analyst=mission.analyst.display_name), mimetype="application/pdf",
                     as_attachment=True, download_name=f"mission-{mission_id}.pdf")


def _observations(mission):
    return (Observation.query.filter_by(mission_id=mission.mission_id)
            .order_by(Observation.video_time_ms.asc(), Observation.observation_id.asc()).all())


def _owned_observation(observation_id):
    observation = db.session.get(Observation, observation_id)
    if observation is None:
        abort(404)
    _owned_mission_or_404(observation.mission_id)
    return observation


@bp.route("/observations/<int:observation_id>/review", methods=["POST"])
@analyst_required
def review(observation_id):
    observation = _owned_observation(observation_id)
    if observation.mission.status != "Complete":
        abort(409, description="Only completed missions can be reviewed.")
    decisions = request.form.getlist("review_status")
    if (len(decisions) != 1 or decisions[0] not in {"Confirmed", "Rejected"}
            or set(request.form) - {"review_status", "_csrf_token"}):
        abort(400, description="Provide exactly one Confirmed or Rejected decision.")
    mission_id = observation.mission_id
    try:
        observation.review_status = decisions[0]
        db.session.commit()
    except Exception:
        db.session.rollback()
        abort(500, description="Review update failed; no success is claimed.")
    return redirect(url_for("main.results", mission_id=mission_id,
                            _anchor=f"observation-{observation_id}"), code=303)


@bp.after_request
def private_results(response):
    if request.endpoint in {"main.results", "main.frame", "main.review",
                            "main.export_pdf", "main.export_gpx", "main.delete_mission", "main.history"}:
        response.headers["Cache-Control"] = "private, no-store"
    return response


@bp.route("/history")
@analyst_required
def history():
    missions = (Mission.query.filter_by(analyst_id=current_user().user_id)
                .order_by(Mission.submitted_at.desc()).all())
    return render_template("history.html", missions=missions)


@bp.route("/missions/<int:mission_id>/delete", methods=["GET", "POST"])
@analyst_required
def delete_mission(mission_id):
    mission = _owned_mission_or_404(mission_id)
    if mission.status == "Processing":
        abort(409, description="A Processing mission cannot be deleted.")
    if request.method == "GET":
        return render_template("delete_mission.html", mission=mission)
    if (request.form.getlist("confirm") != ["delete"]
            or set(request.form) - {"confirm", "_csrf_token"}):
        abort(400, description="Confirm deletion of this mission.")
    try:
        with archive_mission(current_app, mission):
            for observation in list(mission.observations):
                db.session.delete(observation)
            db.session.delete(mission)
            db.session.commit()
    except ValueError:
        db.session.rollback()
        abort(409, description="Mission could not be deleted safely. Nothing was deleted.")
    except Exception:
        db.session.rollback()
        abort(500, description="Mission deletion failed. Nothing was deleted.")
    flash("Mission deleted.", "success")
    return redirect(url_for("main.history"), code=303)
