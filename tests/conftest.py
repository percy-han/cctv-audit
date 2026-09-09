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

"""One line of setup, and the reason it has to exist.

Eleven tests construct a real `AuditPipeline`, whose `__init__` runs
`config.validate()` and refuses to build without a GCP project. On the machine
this suite was written on that project came from `computer_use_agent/.env` --
which is gitignored, because it is a developer's local settings. So the suite
passed here and failed everywhere else: a fresh clone of the pushed branch
scored 11 failed / 468 passed, all of them `GOOGLE_CLOUD_PROJECT is not set`.

Nobody noticed for six days. `./check.sh` runs the tests in *this* directory,
where the .env sits; only `--deep`, which re-clones from GitHub and runs the
tests inside the clone, is looking at what anyone else would get. That is the
check that caught it, on the day the branch was pushed to a second remote.

The value is deliberately fake and deliberately *not* read from anywhere. A
suite that behaves differently depending on what is in someone's .env is not
testing the code, it is testing the laptop -- so this pins the answer rather
than supplying a missing default. `GCP_PROJECT` and not `GOOGLE_CLOUD_PROJECT`
because the former wins in `config.py`, which makes this beat a .env that sets
the latter.

Assignment at import time, not a fixture: `config` is a module-level frozen
dataclass built from the environment when `computer_use_agent.config` is first
imported. A fixture runs after collection has already imported it.
"""

from __future__ import annotations

import os

# Not a real project. Nothing in this suite talks to a cloud; if a test ever
# starts to, the id being obviously fake is what makes that visible.
os.environ["GCP_PROJECT"] = "test-project-not-a-real-one"
