# Accounts and recovery

## First Administrator

On the first start (`start.bat`, or `.\.venv\Scripts\python.exe run.py` from the project folder), DALEEL
creates one Administrator and displays its temporary username and password once in the
interactive terminal. On a server, `python run.py init` does the same without starting the
website, so that the website can then run as a service (see [Put DALEEL online](DEPLOYMENT.md)).
The credentials are never written to a file or log, and only salted password hashes are stored.
After signing in, the Administrator chooses a personal password and continues directly to
Account management. Later starts keep all accounts and passwords.

## Managing accounts

The Administrator creates accounts with a full name, unique username, email address and
role (Analyst or Administrator). DALEEL generates a temporary password and displays it once,
together with the username and a copy button, for private, offline sharing; no activation email
is needed. Until it is replaced, the account can only change its password or sign out. After the
personal password is saved, the user continues in a new session to New Mission (Analyst) or
Account management (Administrator); every other session of that account ends.

Passwords must have 15 to 256 characters; predictable passwords and passwords related to the
person's name or username are rejected. Unicode characters and spaces are allowed.

Administrators can edit account details, suspend or restore access, and reset passwords. A
reset issues a new temporary password and ends the account's existing sessions. The last
active Administrator cannot be demoted or suspended. Email addresses are contact information
only: DALEEL sends no email and has no public registration.

The account workflow includes CSRF protection, server-side role checks, salted password
hashes, limits on repeated failed sign-ins and session revocation. DALEEL is intended for
trusted local operators; it does not provide multi-factor authentication or email-based
password recovery.

## Forgotten Administrator password

Run these commands in an interactive terminal from the repository root (on Linux or macOS,
use `.venv/bin/python`):

```powershell
.\.venv\Scripts\python.exe run.py recover-admin --check
.\.venv\Scripts\python.exe run.py recover-admin
```

The first command confirms that recovery is possible without changing anything. The second
resets the Administrator's password and displays a temporary password once. If there is more
than one Administrator, add `--username USERNAME`. Do not redirect the output to a file.
Access to the computer's terminal and files is the authority for recovery.

## Data location and upgrades

DALEEL stores its database, secret key, uploaded files and evidence images under
`instance/v2/`. As a safeguard, a database located directly at `instance/daleel.db` is never
opened.

**Upgrading from the earlier version.** Version 2 analyzes every sampled frame with two
fine-tuned models and labels observations Both models or One model; observations made by the
earlier version (pretrained models, Dual-Model and Screening-Only labels) are therefore not
mixed into the new database. On the first start of version 2, any earlier data folder
(`instance/app/`, `instance/review/current/` or `instance/stage1/`) is copied into
`instance/archive/`. The database is copied with SQLite's backup function and checked
table by table; every other file is checked by SHA-256. The folder's `MANIFEST.json` lists every
file and its hash, and `RESTORE.md` explains how to open the data again with the earlier
version. The original folders are only read and stay where they are; DALEEL then starts with a
fresh database and the usual first Administrator setup. If a copy cannot be verified, DALEEL
stops without changing anything.

## Updating to a new release

DALEEL is not updated automatically, and the one-line installer never changes an existing
installation. To move to a new release while keeping all accounts, passwords, missions and
evidence:

1. Stop DALEEL by closing its window.
2. Download the new release as a ZIP file from the
   [releases page](https://github.com/SarahAlyahya001/DALEEL/releases) and extract it into a new
   folder. Do not extract it over the existing installation.
3. Copy the `instance` folder from the existing installation into the new folder, next to
   `start.bat`. Copy it; do not move it, so that the existing installation stays complete as a
   backup.
4. Start the new release with `start.bat` in the new folder. The first start sets up Python and
   the packages as described in the README; accounts and missions are those of the copied
   `instance` folder, so no new Administrator is created.
5. When the new release works as expected, the old folder can be deleted.

A release earlier than v2.0.1 has no `start.bat`; its data is updated the same way.

## Account migration

If startup reports that an account migration is required, run:

```powershell
.\.venv\Scripts\python.exe run.py migrate-accounts
```

The command creates and verifies a recoverable SQLite backup, adds the
`users.must_change_password` column with a false default, and verifies the existing records
before committing. Accounts, passwords and missions are preserved. New installations do not
need this command.
