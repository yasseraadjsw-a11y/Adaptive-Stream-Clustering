#!/usr/bin/env sh
set -eu
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall -q src experiments scripts tests main.py
python main.py verify
