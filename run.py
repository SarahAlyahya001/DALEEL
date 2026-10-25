"""DALEEL launcher: python run.py serves the application; other commands are explicit."""
import argparse
import logging
import os
import sys

from sqlalchemy import inspect

from daleel import PROTECTED_DATABASE, create_app, initialize_database
from daleel.account_migration import upgrade_accounts
from daleel.account_runtime import (
    AccountFlowError, bootstrap_administrator, recover_administrator, recovery_available, verify_account_schema,
)
from daleel.legacy_archive import ArchiveError, archive_previous_installations
from daleel.models import db

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5000"
INSTANCE_ROOT = PROTECTED_DATABASE.parent


class LaunchError(RuntimeError):
    """Safe operator guidance about the server configuration."""


def server_address():
    """Read the listening address from DALEEL_HOST and DALEEL_PORT."""
    host = os.environ.get("DALEEL_HOST", DEFAULT_HOST).strip()
    port = os.environ.get("DALEEL_PORT", DEFAULT_PORT).strip()
    if not host:
        raise LaunchError("DALEEL_HOST must not be empty.")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise LaunchError("DALEEL_PORT must be a port number from 1 to 65535.")
    return host, int(port)


def prepare_accounts(output):
    """Create the schema in an empty database and the first Administrator if none exists."""
    if not inspect(db.engine).get_table_names():
        initialize_database()
    verify_account_schema()
    return bootstrap_administrator(output)


def serve(app, host, port):
    """Serve with Waitress; Flask's built-in server is for development only."""
    from waitress import serve as waitress_serve
    waitress_serve(app, host=host, port=port)


def main(argv=None, output=None):
    parser = argparse.ArgumentParser(description="DALEEL web application")
    parser.add_argument("command", nargs="?", choices=("serve", "init", "recover-admin", "migrate-accounts"),
                        default="serve",
                        help="serve (default) runs the website; init only creates the first Administrator")
    parser.add_argument("--username", help="Administrator username for explicit recovery")
    parser.add_argument("--check", action="store_true", help="Check recovery availability without changing credentials")
    args = parser.parse_args(argv)
    if (args.username or args.check) and args.command != "recover-admin":
        parser.error("--username and --check apply only to recover-admin")
    output = output or sys.stdout
    try:
        if args.command == "serve":
            host, port = server_address()
        if args.command in ("serve", "init"):
            # Data of the earlier version is copied and verified once; the originals stay in place.
            archive_previous_installations(INSTANCE_ROOT, output)
        app = create_app()
        with app.app_context():
            if args.command == "migrate-accounts":
                backup = upgrade_accounts()
                output.write("Account schema already current.\n" if backup is None
                             else "Migration verified. Recovery backup: " + str(backup) + "\n")
                return 0
            if args.command == "recover-admin":
                if args.check:
                    recovery_available(args.username)
                    output.write("Administrator recovery is available. No credentials were changed.\n")
                else:
                    recover_administrator(output, args.username)
                return 0
            created = prepare_accounts(output)
            if args.command == "init":
                if not created:
                    output.write("DALEEL is already initialized. No credentials were changed.\n")
                return 0
        from daleel.pipeline.runner import initialize_processor
        app.logger.setLevel(logging.INFO)
        processor = initialize_processor(app)
        output.write(f"DALEEL is running at http://{host}:{port} (press Ctrl+C to stop).\n")
        output.flush()
        try:
            serve(app, host, port)
        finally:
            processor.shutdown()
        return 0
    except (AccountFlowError, ArchiveError, LaunchError) as exc:
        # Only our own fixed, non-sensitive guidance is safe to display.
        sys.stderr.write(str(exc) + "\n")
        return 1
    except Exception:
        # Do not print exception values: a database/driver error can contain
        # parameter values. Diagnose using safe checks, not credential logs.
        sys.stderr.write("DALEEL could not complete this operation. Check setup, database permissions "
                         "and schema. Existing accounts require python run.py migrate-accounts "
                         "if the account column is missing. No credentials were logged.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
