#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Builds the footage the demo plays, into `assets/`.
#
#   ./make_assets.sh                 # synthetic camera view
#   ./make_assets.sh /path/to/real.mp4   # re-encode real footage instead
#
# Two forms come out of one source, because the capture code takes a different
# path for each and both have to be shown working:
#
#   hls/index.m3u8   segmented  -> the probe sees a manifest  -> Plan A, ffmpeg
#                                  pulls the media URL directly
#   store.mp4        one file   -> no manifest at all         -> Plan B, we
#                                  screen-record the player
#
# The burned-in clock counts from 00:00, deliberately **not** a fake wall
# clock. Whoever is demoing says "从 05:00 开始看 5 分钟", and 05:00 is what the
# scrubber says and what the overlay says. A 14:00-style overlay on a video
# whose timeline starts at zero means the two numbers on screen disagree, and
# the first person to notice will be the customer.
#
# The synthetic scene is not pretending to be real footage. It is a moving
# picture with a clock on it, enough to prove capture, windowing, the live
# preview and the report. **It will produce zero violations**, because there is
# nothing in it to violate anything. Pass a real clip as $1 to change that.

set -euo pipefail
cd "$(dirname "$0")"

SOURCE="${1:-}"
OUT=assets
DURATION="${DURATION:-1200}"      # 20 minutes: room for a time span well past
                                  # the start, without a huge image
WIDTH=854
HEIGHT=480
FPS=10
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf

rm -rf "$OUT"
mkdir -p "$OUT/hls"

if [[ -n "$SOURCE" ]]; then
  echo "Using real footage: $SOURCE"
  INPUT=(-i "$SOURCE")
  FILTER="scale=${WIDTH}:${HEIGHT},fps=${FPS}"
else
  echo "Synthesising a ${DURATION}s camera view (no source given)."
  # A fixed room, a member of staff moving behind the counter, and customers
  # arriving and leaving. `mod(t,...)` gives motion that repeats -- which is
  # honest about what this is, and means any window the demo picks has
  # something moving in it.
  INPUT=(-f lavfi -i "color=c=0x1e2126:s=${WIDTH}x${HEIGHT}:r=${FPS}:d=${DURATION}")
  FILTER="\
drawbox=x=0:y=300:w=${WIDTH}:h=180:color=0x33383f@1:t=fill,\
drawbox=x=60:y=250:w=734:h=60:color=0x6b4f2a@1:t=fill,\
drawbox=x=90:y=190:w=90:h=60:color=0x8a8f98@1:t=fill,\
drawbox=x=620:y=180:w=150:h=70:color=0x4a5560@1:t=fill,\
drawbox=x='120+180*abs(sin(t/9))':y=170:w=34:h=80:color=0x2e7d5b@1:t=fill,\
drawbox=x='560+120*abs(sin(t/13+1))':y=330:w=34:h=86:color=0xa04545@1:t=fill,\
drawbox=x='260+300*abs(sin(t/17+2))':y=340:w=34:h=86:color=0x3f5d9e@1:t=fill,\
drawbox=x='40+700*mod(t/47\,1)':y=360:w=30:h=80:color=0x8d7bb0@1:t=fill"
fi

# The overlay goes on last so it survives both branches.
OVERLAY="\
drawbox=x=0:y=0:w=${WIDTH}:h=34:color=0x000000@0.55:t=fill,\
drawtext=fontfile=${FONT}:text='CAM01 COUNTER':x=14:y=8:fontsize=18:fontcolor=0xe8e8e8,\
drawtext=fontfile=${FONT}:text='%{pts\\:hms}':x=${WIDTH}-150:y=8:fontsize=18:fontcolor=0x7ef0a0"

echo "== store.mp4 (progressive; drives Plan B) =="
ffmpeg -y -hide_banner -loglevel error "${INPUT[@]}" \
  -vf "${FILTER},${OVERLAY}" \
  -t "$DURATION" -c:v libx264 -preset veryfast -crf 30 -g $((FPS * 4)) \
  -pix_fmt yuv420p -movflags +faststart "$OUT/store.mp4"

echo "== hls/ (segmented; drives Plan A) =="
# Re-segment the file just produced rather than re-rendering: identical
# pictures on both paths, so a difference in the two audits is a difference in
# the capture path and not in what was filmed.
ffmpeg -y -hide_banner -loglevel error -i "$OUT/store.mp4" \
  -c copy -f hls -hls_time 6 -hls_playlist_type vod \
  -hls_segment_filename "$OUT/hls/seg%04d.ts" "$OUT/hls/index.m3u8"

echo
ls -la "$OUT" "$OUT/hls" | head -12
echo
echo "total: $(du -sh "$OUT" | cut -f1)"
ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$OUT/store.mp4"
