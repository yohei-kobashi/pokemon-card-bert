#!/usr/bin/env bash
for P in $(pgrep -f effrun); do kill "$P" 2>/dev/null; done
for P in $(pgrep -f eff_audit); do kill "$P" 2>/dev/null; done
echo eff-stopped
