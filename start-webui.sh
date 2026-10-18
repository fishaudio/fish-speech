#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"

if [ ! -f awesome_webui/dist/index.html ]; then
    echo "Build the WebUI first: (cd awesome_webui && npm ci && npm run build)" >&2
    exit 1
fi

echo "WebUI: http://<SPARK_LOCAL_IP>:8080/ui (or your --listen address)"
exec ./start-api.sh "$@"
