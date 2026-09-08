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
# Static footage over HTTP. No application, no state, nothing to authenticate
# against -- Cloud Run's own IAM is the only door, because the organisation
# policy will not allow `allUsers` on anything in this project.

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY static/ /srv/video/
COPY assets/ /srv/video/

# nginx:alpine runs as root and drops to `nginx` for workers. Cloud Run is
# happy with that, and the container writes nothing outside /tmp.
EXPOSE 8080
