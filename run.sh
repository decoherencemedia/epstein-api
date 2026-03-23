#!/bin/sh
set -eu
mkdir -p "$(dirname "$EPSTEIN_SQLITE_PATH")"
curl -fsSL -o "$EPSTEIN_SQLITE_PATH" "$DB_URL"
exec gunicorn --worker-tmp-dir /dev/shm -w 2 -b "0.0.0.0:${PORT:-8080}" api:app
