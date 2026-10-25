"""Validation and one-time account credentials, independent of presentation."""
import re
import secrets
import unicodedata


COMMON_PASSWORDS = frozenset({
    "passwordpassword", "password123456789", "password1234567890",
    "qwerty1234567890", "123456789012345", "1234567890123456",
    "letmeinletmeinletmein", "adminadminadminadmin", "correcthorsebatterystaple",
})


def normalize_email(value):
    value = value.strip().casefold()
    if len(value) > 254 or not re.fullmatch(
        r"[a-z0-9.!#$%&'*+/=?^_{|}~-]+@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,63}", value
    ):
        raise ValueError("Enter a valid email address.")
    local, domain = value.rsplit("@", 1)
    if (len(local) > 64 or local.startswith(".") or local.endswith(".")
            or ".." in value or any(not label or len(label) > 63
                                   or label.startswith("-") or label.endswith("-")
                                   for label in domain.split("."))):
        raise ValueError("Enter a valid email address.")
    return value


def password_error(password, user):
    password = unicodedata.normalize("NFC", password)
    if not 15 <= len(password) <= 256:
        return "Use 15–256 characters. Spaces are allowed."
    if any(unicodedata.category(char).startswith("C") for char in password):
        return "Passwords must not contain control or invisible formatting characters."
    folded = password.casefold()
    compact = re.sub(r"[\W_]", "", folded)
    context = {user.username.casefold(), "daleel"}
    if user.email:
        context.add(user.email.split("@", 1)[0])
    context.update(part.casefold() for part in user.display_name.split() if len(part) >= 3)
    base = compact.strip("0123456789")
    sequences = ("abcdefghijklmnopqrstuvwxyz", "01234567890123456789",
                 "qwertyuiopasdfghjklzxcvbnm")
    if (folded in COMMON_PASSWORDS or compact in COMMON_PASSWORDS
            or len(set(folded)) < 5 or base in context
            or any(len(part) >= 3 and re.fullmatch(r"(?:" + re.escape(part) + r")+", base)
                   for part in context)
            or any(compact in sequence or compact in sequence[::-1] for sequence in sequences)
            or re.fullmatch(r"(password|qwerty|letmein|admin)+[0-9]*", compact)):
        return "Choose a less predictable password, unrelated to your name or account."
    return None


def generate_temporary_password():
    """256-bit random secret, held only in memory until one-time disclosure."""
    return secrets.token_urlsafe(32)
