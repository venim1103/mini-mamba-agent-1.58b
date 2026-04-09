# Copyright 2026 venim1103
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

import os


DEFAULT_CONTEXT_LENGTH = 16_384
CONTEXT_LENGTH_ENV = "AGENT_CONTEXT_LENGTH"


def resolve_context_length(default=DEFAULT_CONTEXT_LENGTH):
    """Resolve model context length from env with a safe integer fallback."""
    raw = os.getenv(CONTEXT_LENGTH_ENV)
    if not raw:
        return int(default)

    try:
        value = int(raw)
    except ValueError:
        return int(default)

    return value if value > 0 else int(default)


CONTEXT_LENGTH = resolve_context_length()
