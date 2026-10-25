from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for
from sqlalchemy import or_, text
from sqlalchemy.exc import IntegrityError

from .auth import admin_required, current_user
from .account_security import normalize_email, generate_temporary_password
from .models import User, db

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _account_page(status=200):
    search = request.args.get("q", "").strip()[:150]
    query = User.query
    if search:
        needle = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        query = query.filter(or_(User.username.ilike(needle, escape="\\"),
                                 User.full_name.ilike(needle, escape="\\"),
                                 User.email.ilike(needle, escape="\\")))
    users = query.order_by(User.full_name, User.username).all()
    return render_template("admin.html", users=users, search=search), status


def _lock_admin():
    db.session.rollback()
    db.session.execute(text("BEGIN IMMEDIATE"))
    # A concurrently revoked administrator must not mutate accounts.
    actor = current_user()
    if actor is None or not actor.is_admin:
        abort(403)


def _account_values():
    for name in ("username", "full_name", "email", "role"):
        if len(request.form.getlist(name)) != 1:
            raise ValueError("Submit one full name, username, email address and role.")
    username = request.form["username"].strip()
    full_name = request.form["full_name"].strip()
    role = request.form["role"]
    if not username or len(username) > 100 or any(ord(c) < 32 for c in username):
        raise ValueError("Username must contain 1–100 characters.")
    if not full_name or len(full_name) > 150 or any(ord(c) < 32 for c in full_name):
        raise ValueError("Full name must contain 1–150 characters.")
    if role not in ("analyst", "administrator"):
        raise ValueError("Choose Analyst or Administrator.")
    return dict(username=username, full_name=full_name,
                email=normalize_email(request.form["email"]), role=role)


def _duplicate(values, exclude=None):
    query = User.query.filter(or_(User.username == values["username"], User.email == values["email"]))
    if exclude:
        query = query.filter(User.user_id != exclude)
    return query.first() is not None


@bp.route("/accounts", methods=["GET", "POST"])
@admin_required
def accounts():
    if request.method == "GET":
        return _account_page()
    _lock_admin()
    try:
        values = _account_values()
    except ValueError as exc:
        flash(str(exc), "error")
        return _account_page(400)
    if _duplicate(values):
        flash("That username or email address already exists.", "error")
        return _account_page(400)
    temporary = generate_temporary_password()
    user = User(**values, account_status="Active", password_initialized=True,
                must_change_password=True)
    user.set_password(temporary)
    db.session.add(user)
    try:
        db.session.flush()
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        flash("That username or email address already exists.", "error")
        return _account_page(409)
    return _credentials_page(user, temporary, created=True)


@bp.route("/accounts/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_account(user_id):
    user = db.get_or_404(User, user_id)
    if request.method == "GET":
        return _edit_page(user)
    _lock_admin()
    try:
        # Serialize changes, including concurrent attempts to remove the last admin.
        user = db.session.get(User, user_id)
        values = _account_values()
        statuses = request.form.getlist("account_status")
        if len(statuses) != 1 or statuses[0] not in ("Active", "Invited", "Suspended"):
            raise ValueError("Choose a valid account status.")
        status = statuses[0]
        if not user.password_initialized and status == "Active":
            raise ValueError("Reset this account’s password before activating it.")
        if user.password_initialized and status == "Invited":
            raise ValueError("Use Active or Suspended for an activated account.")
        if user.is_admin and user.account_status == "Active" and (
                values["role"] != "administrator" or status != "Active"):
            if User.query.filter_by(role="administrator", account_status="Active").count() <= 1:
                raise ValueError("The last active administrator cannot be demoted or suspended.")
        if _duplicate(values, user_id):
            raise ValueError("That username or email address already exists.")
        security_change = any(getattr(user, key) != value for key, value in values.items() if key != "full_name")
        security_change = security_change or user.account_status != status
        for key, value in values.items():
            setattr(user, key, value)
        user.account_status = status
        if security_change:
            user.session_version += 1
        db.session.commit()
    except (ValueError, IntegrityError) as exc:
        db.session.rollback()
        flash(str(exc) if isinstance(exc, ValueError) else "That username or email already exists.", "error")
        return _edit_page(db.session.get(User, user_id), 400)
    flash("Account updated." + (" Existing sessions were revoked." if security_change else ""), "ok")
    return redirect(url_for("admin.accounts"))


def _edit_page(user, status=200):
    return render_template("edit_account.html", account=user), status


def _credentials_page(user, temporary, created=False, signed_out=False):
    # Render directly after POST. Never place this value in a cookie, flash,
    # redirect, query string, database, delivery record or server-side receipt.
    return render_template("temporary_credentials.html", account=user,
                           temporary_password=temporary, created=created,
                           signed_out=signed_out)


@bp.post("/accounts/<int:user_id>/reset-password")
@admin_required
def reset_password(user_id):
    db.get_or_404(User, user_id)
    _lock_admin()
    user = db.session.get(User, user_id)
    actor_id = current_user().user_id
    temporary = generate_temporary_password()
    user.set_password(temporary)
    user.password_initialized = True
    user.must_change_password = True
    user.session_version += 1
    if user.account_status == "Invited":
        user.account_status = "Active"
    signed_out = user.user_id == actor_id
    db.session.commit()
    if signed_out:
        session.clear()
    return _credentials_page(user, temporary, signed_out=signed_out)
