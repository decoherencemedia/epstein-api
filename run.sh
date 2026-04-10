#!/bin/sh
set -eu
mkdir -p "$(dirname "$EPSTEIN_SQLITE_PATH")"
curl -fsSL -o "$EPSTEIN_SQLITE_PATH" "$DB_URL"

# Verify the database downloaded correctly
python3 -c "import os,sqlite3 as q;p=os.environ['EPSTEIN_SQLITE_PATH'];assert os.path.getsize(p)>0;assert q.connect(p).execute('select 1 from faces limit 1').fetchone()"

exec gunicorn --worker-tmp-dir /dev/shm -w 2 -b "0.0.0.0:${PORT:-8080}" api:app
