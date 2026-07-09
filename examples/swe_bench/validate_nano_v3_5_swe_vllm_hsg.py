#!/usr/bin/env python3
"""Read-only validation for the regular vLLM 0.23 SWE actor environment."""

import importlib
import importlib.metadata as metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from packaging.version import Version


def main() -> None:
    assert sys.version_info[:2] == (3, 13), sys.version

    import vllm
    import vllm.tool_parsers  # noqa: F401 - registers built-in parsers
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd())).resolve()
    assert metadata.version("vllm") == "0.23.0"
    assert (
        not Path(vllm.__file__)
        .resolve()
        .is_relative_to(Path("/opt/dynamo_venv").resolve())
    ), vllm.__file__
    assert Path("/opt/vllm_backports").read_text(encoding="utf-8").strip() == (
        "vllm#44814 45ffb397d1c7803a78c32846807c71d881e11189"
    )

    actor_reload_meta = (
        Path(vllm.__file__).resolve().parent
        / "model_executor/model_loader/reload/meta.py"
    )
    dynamo_reload_meta = (
        Path("/opt/dynamo_venv/lib/python3.12/site-packages/vllm")
        / "model_executor/model_loader/reload/meta.py"
    )
    assert actor_reload_meta.read_bytes() == dynamo_reload_meta.read_bytes(), (
        actor_reload_meta,
        dynamo_reload_meta,
    )

    comparison_packages = (
        "vllm",
        "torch",
        "torchaudio",
        "torchvision",
        "triton",
        "transformers",
        "tokenizers",
        "flashinfer-python",
        "flashinfer-cubin",
        "compressed-tensors",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-cu13",
    )
    version_probe = (
        "import importlib.metadata as m, json, sys; "
        "print(json.dumps({name: m.version(name) for name in sys.argv[1:]}))"
    )
    dynamo_versions = json.loads(
        subprocess.check_output(
            [
                "/opt/dynamo_venv/bin/python",
                "-c",
                version_probe,
                *comparison_packages,
            ],
            text=True,
        )
    )
    actor_versions = {name: metadata.version(name) for name in comparison_packages}
    optional_absent_packages = (
        "flashinfer-jit-cache",
        "nvidia-cutlass-dsl-libs-cu12",
    )
    optional_probe = (
        "import importlib.metadata as m, json, sys\n"
        "def get_version(name):\n"
        "    try:\n"
        "        return m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        return None\n"
        "print(json.dumps({name: get_version(name) for name in sys.argv[1:]}))"
    )
    dynamo_optional_versions = json.loads(
        subprocess.check_output(
            [
                "/opt/dynamo_venv/bin/python",
                "-c",
                optional_probe,
                *optional_absent_packages,
            ],
            text=True,
        )
    )
    actor_optional_versions = {}
    for name in optional_absent_packages:
        try:
            actor_optional_versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            actor_optional_versions[name] = None
    assert (
        actor_optional_versions
        == dynamo_optional_versions
        == {
            "flashinfer-jit-cache": None,
            "nvidia-cutlass-dsl-libs-cu12": None,
        }
    ), {
        "regular": actor_optional_versions,
        "dynamo": dynamo_optional_versions,
    }
    public_version_packages = {"torch", "torchaudio", "torchvision"}
    mismatches = {}
    for name in comparison_packages:
        actor_version = actor_versions[name]
        dynamo_version = dynamo_versions[name]
        matches = (
            Version(actor_version).base_version == Version(dynamo_version).base_version
            if name in public_version_packages
            else actor_version == dynamo_version
        )
        if not matches:
            mismatches[name] = {
                "regular": actor_version,
                "dynamo": dynamo_version,
            }
    assert not mismatches, mismatches

    importlib.import_module("cutlass.cute")

    subprocess.check_call(
        [
            "/opt/dynamo_venv/bin/python",
            "-c",
            "import cutlass.cute",
        ]
    )

    reasoning_plugin = (
        repo_root / "nemo_rl/models/generation/vllm/reasoning_parsers/"
        "nano_v3_reasoning_parser.py"
    )
    ReasoningParserManager.reasoning_parsers.pop("nano_v3", None)
    ReasoningParserManager.lazy_parsers.pop("nano_v3", None)
    ReasoningParserManager.import_reasoning_parser(str(reasoning_plugin))
    ReasoningParserManager.get_reasoning_parser("nano_v3")
    ToolParserManager.get_tool_parser("qwen3_coder")

    config_path = repo_root / "examples/swe_bench/grpo_nano_v3_5_swe_vllm_hsg.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        generation = yaml.safe_load(config_file)["policy"]["generation"]
    assert generation["backend"] == "vllm"
    assert "dynamo_cfg" not in generation
    assert generation["vllm_cfg"]["async_engine"] is True
    assert generation["vllm_cfg"]["expose_http_server"] is True
    engine_args = inspect.signature(AsyncEngineArgs).parameters
    unsupported_kwargs = sorted(set(generation["vllm_kwargs"]) - set(engine_args))
    assert not unsupported_kwargs, (
        f"unsupported vLLM 0.23 arguments: {unsupported_kwargs}"
    )

    print("regular vLLM", metadata.version("vllm"))
    print("vLLM source", Path(vllm.__file__).resolve())
    print("regular versions", json.dumps(actor_versions, sort_keys=True))
    print("Dynamo versions", json.dumps(dynamo_versions, sort_keys=True))
    print(
        "optional absent packages",
        json.dumps(actor_optional_versions, sort_keys=True),
    )
    print("vLLM NemotronH refit implementation: matched")
    print("tool parser qwen3_coder: registered")
    print("reasoning parser nano_v3: registered")
    print("regular-vLLM SWE configuration: validated")


if __name__ == "__main__":
    main()
