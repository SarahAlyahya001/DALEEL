# Put DALEEL online

This guide publishes DALEEL on a server under your own domain, served over HTTPS, so that
search and rescue teams can use it from their browsers. It uses Docker: the DALEEL container
runs the application with the Waitress production server, and a [Caddy](https://caddyserver.com/)
container in front of it obtains and renews the HTTPS certificate automatically.

## What you need

- A Linux server, for example a virtual private server or a server in your organization,
  with [Docker Engine and the Compose plugin](https://docs.docker.com/engine/install/)
  installed.
- A domain or subdomain, such as `daleel.example.org`, whose DNS **A record** points to the
  server's public IP address.
- Ports **80** and **443** open to the internet, so that Caddy can obtain the certificate.

## Steps

<details>
<summary><b>1. Get DALEEL onto the server</b></summary>

Clone the repository on the server (or copy the folder to it) and open a terminal inside it:

```bash
git clone <repository URL> daleel
cd daleel
```

</details>

<details>
<summary><b>2. Set your domain</b></summary>

Create a file named `.env` that contains your domain:

```bash
echo "DALEEL_DOMAIN=daleel.example.org" > .env
```

</details>

<details>
<summary><b>3. Build DALEEL</b></summary>

```bash
docker compose build
```

The build installs the dependencies and downloads the two detection models, verifying their
SHA-256 hashes. It takes several minutes the first time.

</details>

<details>
<summary><b>4. Create the first Administrator</b></summary>

```bash
docker compose run --rm daleel python run.py init
```

The terminal shows a temporary Administrator username and password once. Copy them; they are
needed in step 6. If DALEEL was already initialized, the command changes nothing.

</details>

<details>
<summary><b>5. Start DALEEL</b></summary>

```bash
docker compose up -d
```

Both containers start now and again automatically after a server restart.

</details>

<details>
<summary><b>6. Open your domain</b></summary>

Open `https://daleel.example.org` (your domain). The first visit can take a few seconds while
Caddy obtains the certificate. Sign in with the temporary credentials from step 4, choose a
personal password, and create accounts for the analysts as described in the
[README](../README.md#try-daleel).

</details>

## Operating DALEEL

| Task | Command |
| --- | --- |
| See whether it is running | `docker compose ps` |
| Read the logs | `docker compose logs -f daleel` |
| Stop / start | `docker compose stop` / `docker compose start` |
| Update to a newer version | `git pull` then `docker compose up -d --build` |
| Reset a forgotten Administrator password | `docker compose exec daleel python run.py recover-admin` |
| Back up all data | `docker compose stop daleel`, `docker compose cp daleel:/app/instance ./daleel-backup`, `docker compose start daleel` |

All accounts, missions, uploaded files, evidence images and the session secret key are stored
in the `daleel-data` Docker volume, so they survive restarts and updates. Back it up regularly.

Good to know:

- DALEEL processes one mission at a time. A mission submitted while another is being
  processed is not created; DALEEL reports that processing is busy, and the analyst can
  submit it again when the first mission has finished.
- Maps are loaded by each user's browser from Esri World Imagery and OpenStreetMap, so users
  need internet access to see map tiles. Everything else is served by your server.
- DALEEL sends no email. Administrators create every account and hand over the temporary
  password personally.

## Running without Docker

DALEEL can also run directly on a server with Python 3.10. Install it as described in step 1
of the [README](../README.md#try-daleel), run `python run.py init` once in a terminal, and then
run `python run.py` as a service behind any reverse proxy that provides HTTPS (for example
Caddy or Nginx), with these environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DALEEL_HOST` | Address that the server listens on | `127.0.0.1` |
| `DALEEL_PORT` | Port that the server listens on | `5000` |
| `DALEEL_TRUSTED_PROXIES` | Number of reverse proxies in front of DALEEL; set to `1` behind a single proxy so that client addresses and HTTPS are recognized | `0` |
| `DALEEL_COOKIE_SECURE` | Set to `1` when the site is served over HTTPS, so sign-in cookies are sent only over HTTPS | not set |
| `DALEEL_SECRET_KEY` | Session signing key; if not set, a random key is created once and kept in `instance/v2/secret_key` | generated |
| `DALEEL_WEIGHTS_N`, `DALEEL_WEIGHTS_S` | Locations of the model files | `weights/yolov8n_sar_v1.pt`, `weights/yolov8s_sar_v1.pt` |

The reverse proxy must accept uploads of at least 505 MB (in Nginx:
`client_max_body_size 505m;`) and pass the `X-Forwarded-For`, `X-Forwarded-Proto` and
`X-Forwarded-Host` headers. Set `DALEEL_TRUSTED_PROXIES` only to the number of proxies that
actually exist; a higher value would let clients forge their address.
