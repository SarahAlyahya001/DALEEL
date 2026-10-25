"""Server deployment behavior: reverse-proxy trust and HTTPS headers."""
import pytest
from flask import request

from daleel import create_app, initialize_database
from daleel.models import db

FORWARDED = {"X-Forwarded-For": "203.0.113.7", "X-Forwarded-Proto": "https", "X-Forwarded-Host": "daleel.example.org"}


def make_app(tmp_path, **config):
    app = create_app(str(tmp_path / "instance"), {"TESTING": True, **config})
    with app.app_context():
        initialize_database()

    @app.get("/_test/client")
    def client_view():
        return {"address": request.remote_addr, "secure": request.is_secure, "host": request.host}
    return app


@pytest.fixture
def disposed():
    apps = []
    yield apps.append
    for app in apps:
        with app.app_context():
            db.session.remove()
            db.engine.dispose()


def test_forwarded_headers_are_ignored_without_a_trusted_proxy(tmp_path, disposed):
    app = make_app(tmp_path)
    disposed(app)
    response = app.test_client().get("/_test/client", headers=FORWARDED)
    assert response.json == {"address": "127.0.0.1", "secure": False, "host": "localhost"}
    assert "Strict-Transport-Security" not in response.headers


def test_one_trusted_proxy_supplies_client_address_scheme_and_host(tmp_path, disposed):
    app = make_app(tmp_path, TRUSTED_PROXIES="1")
    disposed(app)
    response = app.test_client().get("/_test/client", headers=FORWARDED)
    assert response.json == {"address": "203.0.113.7", "secure": True, "host": "daleel.example.org"}
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000"


def test_only_the_configured_number_of_proxies_is_trusted(tmp_path, disposed):
    app = make_app(tmp_path, TRUSTED_PROXIES="1")
    disposed(app)
    # A client-supplied address is followed by the proxy's own entry; only the last hop is trusted.
    response = app.test_client().get("/_test/client", headers={"X-Forwarded-For": "198.51.100.9, 203.0.113.7"})
    assert response.json["address"] == "203.0.113.7"


@pytest.mark.parametrize("value", ["-1", "one", "1.5"])
def test_invalid_proxy_count_is_rejected(tmp_path, value):
    with pytest.raises(RuntimeError, match="DALEEL_TRUSTED_PROXIES"):
        create_app(str(tmp_path / "instance"), {"TESTING": True, "TRUSTED_PROXIES": value})


def test_proxy_count_is_read_from_the_environment(tmp_path, monkeypatch, disposed):
    monkeypatch.setenv("DALEEL_TRUSTED_PROXIES", "1")
    app = make_app(tmp_path)
    disposed(app)
    response = app.test_client().get("/_test/client", headers=FORWARDED)
    assert response.json["address"] == "203.0.113.7"
