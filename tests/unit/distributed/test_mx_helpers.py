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

import os
import sys
import types

from nemo_rl.distributed.mx_helpers import MxConfig, pin_local_nic


def test_mx_config_accepts_stripe_nic_pin():
    mx_config = MxConfig.from_dict({"nic_pin": "stripe"})

    assert mx_config.nic_pin == "stripe"


def test_pin_local_nic_delegates_stripe_to_modelexpress(monkeypatch):
    calls = []

    def apply_nic_pin_for_device(*, device_id):
        calls.append(device_id)
        assert device_id == 3

    ucx_utils = types.ModuleType("modelexpress.ucx_utils")
    ucx_utils.apply_nic_pin_for_device = apply_nic_pin_for_device
    modelexpress = types.ModuleType("modelexpress")
    modelexpress.ucx_utils = ucx_utils
    monkeypatch.setitem(sys.modules, "modelexpress", modelexpress)
    monkeypatch.setitem(sys.modules, "modelexpress.ucx_utils", ucx_utils)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    monkeypatch.delenv("MX_RDMA_NIC_PIN", raising=False)

    pin_local_nic(device_id=3, mode="stripe")

    assert calls == [3]
    assert os.environ["MX_RDMA_NIC_PIN"] == "stripe"
    assert "UCX_NET_DEVICES" not in os.environ


def test_pin_local_nic_delegates_when_stripe_fallback_finds_no_nics(monkeypatch):
    calls = []

    def apply_nic_pin_for_device(*, device_id):
        calls.append(device_id)

    ucx_utils = types.ModuleType("modelexpress.ucx_utils")
    ucx_utils.apply_nic_pin_for_device = apply_nic_pin_for_device
    ucx_utils._list_compute_ib_nics = lambda min_rate_gbps=None: []
    modelexpress = types.ModuleType("modelexpress")
    modelexpress.ucx_utils = ucx_utils
    monkeypatch.setitem(sys.modules, "modelexpress", modelexpress)
    monkeypatch.setitem(sys.modules, "modelexpress.ucx_utils", ucx_utils)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    monkeypatch.delenv("MX_RDMA_NIC_PIN", raising=False)

    pin_local_nic(device_id=2, mode="stripe")

    assert calls == [2]
    assert os.environ["MX_RDMA_NIC_PIN"] == "stripe"
    assert "UCX_NET_DEVICES" not in os.environ


def test_pin_local_nic_stripe_fallback_sets_all_compute_nics(monkeypatch):
    calls = []
    ucx_utils = types.ModuleType("modelexpress.ucx_utils")
    ucx_utils.apply_nic_pin_for_device = lambda *, device_id: calls.append(device_id)
    ucx_utils._stripe_all_compute_nics = lambda: calls.append("stripe-helper")
    ucx_utils._list_compute_ib_nics = lambda min_rate_gbps=None: [
        ("mlx5_0", 0, 400.0, []),
        ("mlx5_1", 0, 400.0, []),
        ("mlx5_2", 0, 400.0, []),
        ("mlx5_3", 0, 400.0, []),
    ]
    modelexpress = types.ModuleType("modelexpress")
    modelexpress.ucx_utils = ucx_utils
    monkeypatch.setitem(sys.modules, "modelexpress", modelexpress)
    monkeypatch.setitem(sys.modules, "modelexpress.ucx_utils", ucx_utils)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    monkeypatch.delenv("UCX_MAX_RMA_RAILS", raising=False)
    monkeypatch.delenv("MX_RDMA_NIC_PIN", raising=False)

    pin_local_nic(device_id=0, mode="stripe")

    assert calls == []
    assert os.environ["MX_RDMA_NIC_PIN"] == "off"
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
    assert os.environ["UCX_MAX_RMA_RAILS"] == "4"


def test_pin_local_nic_preserves_legacy_explicit_pin(monkeypatch):
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    monkeypatch.delenv("MX_RDMA_NIC_PIN", raising=False)

    pin_local_nic(device_id=0, mode="mlx5_2:1")

    assert os.environ["UCX_NET_DEVICES"] == "mlx5_2:1"
    assert os.environ["MX_RDMA_NIC_PIN"] == "off"
