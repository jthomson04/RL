# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Helpers for code that needs to know whether it is running in Kubernetes."""

import os
from pathlib import Path

_POD_NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def is_in_kubernetes() -> bool:
    """Return whether the current process runs inside a Kubernetes pod."""
    return "KUBERNETES_SERVICE_HOST" in os.environ


def read_pod_namespace() -> str | None:
    """Return the pod namespace, or None when its projection is unavailable."""
    try:
        namespace = _POD_NAMESPACE_FILE.read_text().strip()
    except OSError:
        return None
    return namespace or None
