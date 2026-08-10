#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.17.0",
    "torch": "2.10.0",
    "transformers": "5.12.1",
    "huggingface-hub": "1.24.0",
    "ray": "2.56.0",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
    "swanlab": "0.9.1",
    "peft": "0.15.2",
}
PATCH_MARKER = "SHOPPING_GRPO_GRAPHGPO_PATCH_V7"
MAX_SAFE_RESPONSE_LENGTH = 20480
MAX_SAFE_SEQUENCE_LENGTH = 24576
CURRENT_RUNTIME_FILES = {
    "observation.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py",
    "pack_api.py": "environments/ShopSimulator/shop_env/shop_env/pack_api.py",
    "reward.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
    "slot_lease_pool.py": "environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py",
    "web_agent_text_env.py": "environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py",
}


def validate_reward_runtime_files(manifest, root):
    if manifest.get("lease_contract") != "explicit-client-release-v1":
        raise SystemExit(
            "Environment v2.1 manifest must select explicit-client-release-v1"
        )
    expected = manifest.get("runtime_files_sha256")
    if not isinstance(expected, dict) or set(expected) != set(CURRENT_RUNTIME_FILES):
        raise SystemExit(
            "Environment v2.1 manifest runtime_files_sha256 is missing or incomplete"
        )
    from shopping_grpo.environment.manifest import sha256_file

    mismatches = {}
    for name, relative_path in CURRENT_RUNTIME_FILES.items():
        actual = sha256_file(Path(root) / relative_path)
        if actual != expected[name]:
            mismatches[name] = {"expected": expected[name], "actual": actual}
    if mismatches:
        raise SystemExit(
            "Environment v2.1 runtime file hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_environment_contract():
    required_version = os.environ.get(
        "SHOPPING_ENVIRONMENT_VERSION",
        "shopsimulator-environment-v2.1",
    )
    if required_version != "shopsimulator-environment-v2.1":
        raise SystemExit(
            "this repository supports only shopsimulator-environment-v2.1"
        )
    manifest_path = os.environ.get("SHOPPING_ENV_MANIFEST")
    if not manifest_path or not Path(manifest_path).is_file():
        raise SystemExit(
            f"{required_version} requires SHOPPING_ENV_MANIFEST pointing to a frozen manifest"
        )
    try:
        from shopping_grpo.environment.manifest import validate_manifest

        manifest = validate_manifest(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {required_version} manifest: {exc}") from exc
    actual_environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.1",
    )
    if actual_environment_version != required_version:
        raise SystemExit(
            "environment manifest version mismatch: "
            f"expected {required_version}, got {actual_environment_version}"
        )
    tools_path = Path(
        os.environ.get(
            "SHOPPING_TOOL_CONFIG",
            Path(__file__).resolve().parents[1]
            / "configs/tools.json",
        )
    )
    tools = json.loads(tools_path.read_text(encoding="utf-8")).get("tools", [])
    tool_names = {
        item.get("tool_schema", {}).get("function", {}).get("name")
        for item in tools
    }
    if "finish_without_purchase" not in tool_names:
        raise SystemExit("Environment v2 tool config is missing finish_without_purchase")
    if int(manifest["max_steps"]) != 35:
        raise SystemExit("Environment v2 GRPO contract requires max_steps=35")
    validate_reward_runtime_files(
        manifest,
        Path(__file__).resolve().parents[1],
    )
    print(
        f"{required_version} manifest preflight passed: "
        + json.dumps(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "shopsimulator_commit": manifest["shopsimulator_commit"],
                "observation_version": manifest["observation_version"],
                "reward_version": manifest["reward"]["version"],
                "search_version": manifest["search"]["version"],
                "lease_contract": manifest.get("lease_contract"),
                "runtime_file_count": len(manifest.get("runtime_files_sha256") or {}),
            },
            sort_keys=True,
        )
    )


def compose_runtime_config(overrides):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise SystemExit(f"cannot parse GRPO config before preflight: {exc}") from exc

    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_name = os.environ.get("GRPO_CONFIG_NAME", "grpo")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def validate_dynamic_sampling(config, verl_source: Path, installed):
    dynamic_config = config.get("shopping_dynamic_sampling", {})
    if not bool(dynamic_config.get("enable", False)):
        return

    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            f"shopping dynamic sampling requires verl==0.8.0, got {installed.get('verl')}"
        )
    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not ray_trainer.is_file():
        raise SystemExit(f"cannot locate installed RayPPOTrainer source: {ray_trainer}")
    if PATCH_MARKER not in ray_trainer.read_text(encoding="utf-8"):
        raise SystemExit(
            "shopping dynamic sampling is enabled but the pinned veRL patch marker is missing; "
            "run scripts/apply_verl_dynamic_sampling_patch.py first"
        )

    try:
        from shopping_grpo.training.grpo.dynamic_sampling import (
            extract_shopping_group_signals,
            select_reward_varying_groups,
        )
    except ImportError as exc:
        raise SystemExit(f"shopping dynamic sampling helper is unavailable: {exc}") from exc
    utility, success, invalid, reasons = extract_shopping_group_signals(
        [
            {
                "infrastructure_invalid": False,
                "reward": {
                    "terminal_utility": reward,
                    "purchase_success": reward > 0,
                    "sampling_invalid": False,
                },
            }
            for reward in (0.0, 1.0, 0.0, 0.0)
        ]
    )
    indices, _ = select_reward_varying_groups(
        ["preflight"] * 4,
        [0.0, 1.0, 0.0, 0.0],
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    if indices != [0, 1, 2, 3]:
        raise SystemExit("shopping dynamic sampling helper failed its import-time sanity check")

    if dynamic_config.get("metric") != "seq_reward":
        raise SystemExit("shopping_dynamic_sampling.metric must be seq_reward")
    if int(dynamic_config.get("max_num_gen_batches", 0)) <= 0:
        raise SystemExit("shopping_dynamic_sampling.max_num_gen_batches must be positive")
    if int(dynamic_config.get("max_consecutive_skipped_updates", 0)) <= 0:
        raise SystemExit(
            "shopping_dynamic_sampling.max_consecutive_skipped_updates must be positive"
        )
    reward_tolerance = float(dynamic_config.get("reward_tolerance", -1))
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise SystemExit("shopping_dynamic_sampling.reward_tolerance must be finite and non-negative")
    if not bool(config.algorithm.rollout_correction.get("bypass_mode", False)):
        raise SystemExit("shopping dynamic sampling requires rollout_correction.bypass_mode=true")
    if not bool(config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
        raise SystemExit("shopping dynamic sampling requires rollout.calculate_log_probs=true")

    print(
        "shopping dynamic sampling preflight passed: "
        + json.dumps(
            {
                "enable": True,
                "metric": str(dynamic_config.metric),
                "max_num_gen_batches": int(dynamic_config.max_num_gen_batches),
                "max_consecutive_skipped_updates": int(
                    dynamic_config.max_consecutive_skipped_updates
                ),
                "reward_tolerance": reward_tolerance,
                "ray_trainer": str(ray_trainer),
                "marker": PATCH_MARKER,
            },
            sort_keys=True,
        )
    )


def validate_agent_progrpo(config):
    estimator = getattr(
        config.algorithm.adv_estimator,
        "value",
        config.algorithm.adv_estimator,
    )
    if str(estimator) != "agent_progrpo":
        return

    agent_config = config.algorithm.get("agent_progrpo", {})
    alpha = float(agent_config.get("alpha", -1.0))
    low_probability_fraction = float(
        agent_config.get("low_probability_fraction", -1.0)
    )
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise SystemExit("algorithm.agent_progrpo.alpha must be finite and in [0, 1]")
    if not math.isfinite(low_probability_fraction) or not 0.0 < low_probability_fraction <= 1.0:
        raise SystemExit(
            "algorithm.agent_progrpo.low_probability_fraction must be finite and in (0, 1]"
        )
    if bool(config.algorithm.get("norm_adv_by_std_in_grpo", True)):
        raise SystemExit(
            "Agent-ProGRPO requires algorithm.norm_adv_by_std_in_grpo=false"
        )
    if not bool(config.get("shopping_dynamic_sampling", {}).get("enable", False)):
        raise SystemExit("Agent-ProGRPO requires shopping_dynamic_sampling.enable=true")
    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise SystemExit("Agent-ProGRPO requires rollout.n>=2")

    try:
        from shopping_grpo.training.grpo.progrpo import (
            AGENT_PROGRPO_ESTIMATOR,
            compute_agent_progrpo_outcome_advantage,
        )
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    except ImportError as exc:
        raise SystemExit(f"Agent-ProGRPO estimator is unavailable: {exc}") from exc
    if get_adv_estimator_fn(AGENT_PROGRPO_ESTIMATOR) is not compute_agent_progrpo_outcome_advantage:
        raise SystemExit("Agent-ProGRPO estimator registry sanity check failed")

    print(
        "Agent-ProGRPO preflight passed: "
        + json.dumps(
            {
                "adv_estimator": AGENT_PROGRPO_ESTIMATOR,
                "alpha": alpha,
                "low_probability_fraction": low_probability_fraction,
                "norm_adv_by_std_in_grpo": False,
                "rollout_n": int(config.actor_rollout_ref.rollout.n),
            },
            sort_keys=True,
        )
    )


def validate_shopping_gigpo(config):
    estimator = getattr(
        config.algorithm.adv_estimator,
        "value",
        config.algorithm.adv_estimator,
    )
    if str(estimator) != "shopping_gigpo":
        return

    gigpo_config = config.algorithm.get("shopping_gigpo", {})
    process_verifier_enabled = bool(
        gigpo_config.get("process_verifier_enabled", False)
    )
    step_weight = float(gigpo_config.get("step_advantage_weight", -1.0))
    potential_gamma = float(gigpo_config.get("potential_gamma", -1.0))
    potential_reward_clip = float(
        gigpo_config.get("potential_reward_clip", -1.0)
    )
    minimum_group_size = int(gigpo_config.get("min_anchor_group_size", 0))
    micro_clip = float(gigpo_config.get("micro_advantage_clip", -1.0))
    online_judge_weight = float(gigpo_config.get("online_judge_weight", -1.0))
    if not process_verifier_enabled:
        raise SystemExit(
            "Shopping GiGPO requires "
            "algorithm.shopping_gigpo.process_verifier_enabled=true"
        )
    if not math.isfinite(step_weight) or not 0.0 < step_weight <= 1.0:
        raise SystemExit(
            "algorithm.shopping_gigpo.step_advantage_weight must be finite and in (0, 1]"
        )
    if not math.isfinite(potential_gamma) or not 0.0 < potential_gamma <= 1.0:
        raise SystemExit(
            "algorithm.shopping_gigpo.potential_gamma must be finite and in (0, 1]"
        )
    if not math.isfinite(potential_reward_clip) or potential_reward_clip < 0.0:
        raise SystemExit(
            "algorithm.shopping_gigpo.potential_reward_clip must be finite and non-negative"
        )
    if minimum_group_size < 2:
        raise SystemExit(
            "algorithm.shopping_gigpo.min_anchor_group_size must be at least 2"
        )
    if not math.isfinite(micro_clip) or micro_clip <= 0.0:
        raise SystemExit(
            "algorithm.shopping_gigpo.micro_advantage_clip must be finite and positive"
        )
    if not math.isfinite(online_judge_weight) or online_judge_weight != 0.0:
        raise SystemExit(
            "algorithm.shopping_gigpo.online_judge_weight must remain 0; "
            "trajectory Judge is evaluation-only"
        )
    if bool(config.algorithm.get("norm_adv_by_std_in_grpo", True)):
        raise SystemExit("Shopping GiGPO requires algorithm.norm_adv_by_std_in_grpo=false")
    if bool(config.get("shopping_dynamic_sampling", {}).get("enable", False)):
        raise SystemExit(
            "Shopping GiGPO requires shopping_dynamic_sampling.enable=false: "
            "the current sampler filters only on terminal reward and can discard "
            "constant-terminal groups that still have valid process advantages"
        )
    if int(config.actor_rollout_ref.rollout.n) < minimum_group_size:
        raise SystemExit(
            "Shopping GiGPO requires rollout.n>=min_anchor_group_size"
        )
    runtime_anchor_mode = os.environ.get(
        "SHOPPING_GIGPO_ANCHOR_MODE",
        "structured",
    )
    runtime_gamma = float(os.environ.get("SHOPPING_GIGPO_GAMMA", "1.0"))
    runtime_clip = float(os.environ.get("SHOPPING_GIGPO_POTENTIAL_CLIP", "0.25"))
    runtime_process_enabled = (
        os.environ.get("SHOPPING_GIGPO_PROCESS_ENABLE", "false").lower() == "true"
    )
    if not runtime_process_enabled:
        raise SystemExit(
            "Shopping GiGPO requires SHOPPING_GIGPO_PROCESS_ENABLE=true"
        )
    if runtime_anchor_mode != "structured":
        raise SystemExit("Shopping GiGPO formal runs require structured public anchors")
    if runtime_gamma != potential_gamma:
        raise SystemExit(
            "SHOPPING_GIGPO_GAMMA must equal algorithm.shopping_gigpo.potential_gamma"
        )
    if runtime_clip != potential_reward_clip:
        raise SystemExit(
            "SHOPPING_GIGPO_POTENTIAL_CLIP must equal "
            "algorithm.shopping_gigpo.potential_reward_clip"
        )

    try:
        from shopping_grpo.training.grpo.gigpo import (
            SHOPPING_GIGPO_ESTIMATOR,
            compute_shopping_gigpo_advantage,
        )
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    except ImportError as exc:
        raise SystemExit(f"Shopping GiGPO estimator is unavailable: {exc}") from exc
    if get_adv_estimator_fn(SHOPPING_GIGPO_ESTIMATOR) is not compute_shopping_gigpo_advantage:
        raise SystemExit("Shopping GiGPO estimator registry sanity check failed")

    print(
        "Shopping GiGPO preflight passed: "
        + json.dumps(
            {
                "adv_estimator": SHOPPING_GIGPO_ESTIMATOR,
                "process_verifier_enabled": True,
                "step_advantage_weight": step_weight,
                "potential_gamma": potential_gamma,
                "potential_reward_clip": potential_reward_clip,
                "anchor_mode": runtime_anchor_mode,
                "min_anchor_group_size": minimum_group_size,
                "micro_advantage_clip": micro_clip,
                "online_judge_weight": online_judge_weight,
                "rollout_n": int(config.actor_rollout_ref.rollout.n),
            },
            sort_keys=True,
        )
    )


def validate_shopping_graphgpo(config):
    """Validate the exact-public-state GraphGPO-lite training contract."""
    estimator = getattr(
        config.algorithm.adv_estimator,
        "value",
        config.algorithm.adv_estimator,
    )
    if str(estimator) != "shopping_graphgpo":
        return

    graph_config = config.algorithm.get("shopping_graphgpo", {})
    dynamic_config = config.get("shopping_dynamic_sampling", {})
    enabled = bool(graph_config.get("enabled", False))
    micro_weight = float(graph_config.get("micro_advantage_weight", -1.0))
    minimum_size = int(graph_config.get("min_contrast_size", 0))
    micro_clip = float(graph_config.get("micro_advantage_clip", -1.0))
    fallback = str(graph_config.get("fallback", ""))
    gamma = float(graph_config.get("potential_gamma", -1.0))
    potential_clip = float(graph_config.get("potential_reward_clip", -1.0))
    if not enabled:
        raise SystemExit(
            "Shopping GraphGPO requires algorithm.shopping_graphgpo.enabled=true"
        )
    if not math.isfinite(micro_weight) or not 0.0 < micro_weight <= 1.0:
        raise SystemExit(
            "algorithm.shopping_graphgpo.micro_advantage_weight must be in (0, 1]"
        )
    if minimum_size < 2:
        raise SystemExit(
            "algorithm.shopping_graphgpo.min_contrast_size must be at least 2"
        )
    if not math.isfinite(micro_clip) or micro_clip <= 0.0:
        raise SystemExit(
            "algorithm.shopping_graphgpo.micro_advantage_clip must be positive"
        )
    if fallback not in {"shopping_gigpo", "grpo"}:
        raise SystemExit(
            "algorithm.shopping_graphgpo.fallback must be shopping_gigpo or grpo"
        )
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise SystemExit(
            "algorithm.shopping_graphgpo.potential_gamma must be in (0, 1]"
        )
    if not math.isfinite(potential_clip) or potential_clip < 0.0:
        raise SystemExit(
            "algorithm.shopping_graphgpo.potential_reward_clip must be non-negative"
        )
    if bool(config.algorithm.get("norm_adv_by_std_in_grpo", True)):
        raise SystemExit(
            "Shopping GraphGPO requires algorithm.norm_adv_by_std_in_grpo=false"
        )
    if int(config.actor_rollout_ref.rollout.n) < minimum_size:
        raise SystemExit("Shopping GraphGPO requires rollout.n>=min_contrast_size")
    if not bool(dynamic_config.get("enable", False)):
        raise SystemExit(
            "Shopping GraphGPO requires shopping_dynamic_sampling.enable=true"
        )
    if not bool(dynamic_config.get("process_aware_zero_signal_gate", False)):
        raise SystemExit(
            "Shopping GraphGPO requires process_aware_zero_signal_gate=true"
        )
    if os.environ.get("SHOPPING_GIGPO_PROCESS_ENABLE", "false").lower() != "true":
        raise SystemExit(
            "Shopping GraphGPO requires SHOPPING_GIGPO_PROCESS_ENABLE=true"
        )

    try:
        from shopping_grpo.training.grpo.graphgpo import (
            SHOPPING_GRAPHGPO_ESTIMATOR,
            compute_shopping_graphgpo_advantage,
        )
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    except ImportError as exc:
        raise SystemExit(f"Shopping GraphGPO estimator is unavailable: {exc}") from exc
    if (
        get_adv_estimator_fn(SHOPPING_GRAPHGPO_ESTIMATOR)
        is not compute_shopping_graphgpo_advantage
    ):
        raise SystemExit("Shopping GraphGPO estimator registry sanity check failed")
    print(
        "Shopping GraphGPO preflight passed: "
        + json.dumps(
            {
                "adv_estimator": SHOPPING_GRAPHGPO_ESTIMATOR,
                "exact_public_observation_graph": True,
                "strict_goal": "reward_v3:gold_purchase",
                "micro_advantage_weight": micro_weight,
                "min_contrast_size": minimum_size,
                "micro_advantage_clip": micro_clip,
                "fallback": fallback,
                "process_aware_zero_signal_gate": True,
                "rollout_n": int(config.actor_rollout_ref.rollout.n),
            },
            sort_keys=True,
        )
    )


def validate_model_runtime(model_path):
    """Reject a nominally valid stack that cannot load the frozen Qwen3.5 model."""
    try:
        from transformers import AutoConfig
        from vllm.model_executor.models import ModelRegistry
    except ImportError as exc:
        raise SystemExit(f"cannot inspect GRPO model runtime: {exc}") from exc

    resolved = Path(model_path).resolve()
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise SystemExit(f"invalid GRPO_MODEL_PATH: {resolved}")
    try:
        config = AutoConfig.from_pretrained(resolved, trust_remote_code=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Transformers cannot load GRPO model config: {exc}") from exc

    architectures = list(getattr(config, "architectures", None) or [])
    supported_architectures = set(ModelRegistry.get_supported_archs())
    unsupported = [name for name in architectures if name not in supported_architectures]
    if str(getattr(config, "model_type", "")) != "qwen3_5":
        raise SystemExit(
            f"GRPO requires model_type=qwen3_5, got {getattr(config, 'model_type', None)}"
        )
    if not architectures or unsupported:
        raise SystemExit(
            "vLLM cannot load GRPO model architecture(s): "
            + ", ".join(unsupported or ["<missing>"])
        )
    print(
        "GRPO model runtime preflight passed: "
        + json.dumps(
            {
                "model": str(resolved),
                "model_type": str(config.model_type),
                "architectures": architectures,
            },
            sort_keys=True,
        )
    )


def validate_peft_lora_runtime(config):
    """Exercise the exact LoRA constructor shape used by veRL workers."""
    model = config.actor_rollout_ref.model
    if int(model.get("lora_rank", 0)) <= 0:
        return

    try:
        from peft import LoraConfig
        from shopping_grpo.training.grpo.peft_compat import PATCH_MODE_ATTR
    except ImportError as exc:
        raise SystemExit(f"cannot import PEFT LoRA runtime: {exc}") from exc

    target_parameters = model.get("target_parameters")
    mode = getattr(LoraConfig, PATCH_MODE_ATTR, None)
    if target_parameters is not None and mode != "native":
        raise SystemExit(
            "the pinned PEFT compatibility shim cannot train target_parameters LoRA"
        )
    try:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(model.lora_rank),
            lora_alpha=int(model.lora_alpha),
            target_modules=model.get("target_modules"),
            target_parameters=target_parameters,
            bias="none",
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"incompatible PEFT LoRA constructor: {exc}") from exc
    if int(lora_config.r) != int(model.lora_rank):
        raise SystemExit("PEFT LoRA constructor changed the configured rank")
    adapter_sync = not bool(model.get("lora", {}).get("merge", False))
    if adapter_sync:
        try:
            from vllm.model_executor.models.interfaces import supports_lora
            from vllm.model_executor.models.qwen3_5 import (
                Qwen3_5ForConditionalGeneration,
            )
        except ImportError as exc:
            raise SystemExit(f"cannot inspect vLLM Qwen3.5 LoRA support: {exc}") from exc
        if not supports_lora(Qwen3_5ForConditionalGeneration):
            raise SystemExit(
                "actor_rollout_ref.model.lora.merge=false requires vLLM "
                "Qwen3.5 LoRA support"
            )
    print(
        "PEFT LoRA runtime preflight passed: "
        + json.dumps(
            {
                "mode": mode,
                "rank": int(lora_config.r),
                "target_modules": str(lora_config.target_modules),
                "target_parameters": target_parameters,
                "adapter_sync": adapter_sync,
            },
            sort_keys=True,
        )
    )


def validate_swanlab_tracking(config):
    """Validate SwanLab only when the user explicitly enables it."""
    logger_backends = list(config.trainer.get("logger", []))
    if "swanlab" not in logger_backends:
        return
    forbidden = {"wandb", "tracking", "vemlp_wandb"} & set(logger_backends)
    if forbidden:
        raise SystemExit(
            "Reward v3 GRPO forbids W&B logger backends: "
            + ", ".join(sorted(forbidden))
        )
    if os.environ.get("SWANLAB_MODE") != "online":
        raise SystemExit("Reward v3 GRPO requires SWANLAB_MODE=online")
    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit(
            "Reward v3 GRPO requires SWANLAB_API_KEY in the launching environment"
        )
    log_dir = os.environ.get("SWANLAB_LOG_DIR")
    if not log_dir:
        raise SystemExit("Reward v3 GRPO requires SWANLAB_LOG_DIR")
    resolved_log_dir = Path(log_dir).resolve()
    if str(config.trainer.get("project_name")) != "shopping-grpo":
        raise SystemExit("Reward v3 GRPO SwanLab project must be shopping-grpo")
    print(
        "SwanLab online preflight passed: "
        + json.dumps(
            {
                "api_key": "present",
                "logger": logger_backends,
                "log_dir": str(resolved_log_dir),
                "mode": "online",
                "project": str(config.trainer.project_name),
                "run_name": str(config.trainer.experiment_name),
            },
            sort_keys=True,
        )
    )


def ppo_gradient_accumulation_steps(mini_batch_size: int, micro_batch_size: int) -> int:
    mini = int(mini_batch_size)
    micro = int(micro_batch_size)
    if mini <= 0 or micro <= 0:
        raise ValueError("PPO mini and micro batch sizes must be positive")
    if mini % micro:
        raise ValueError("PPO mini batch size must be divisible by micro batch size")
    return mini // micro


def validate_training_memory_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    total_length = prompt_length + response_length
    actor = config.actor_rollout_ref.actor
    model = config.actor_rollout_ref.model
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref
    smoke_mode = os.environ.get("GRPO_SMOKE_MODE") == "1"
    canary_mode = os.environ.get("GRPO_CANARY_MODE") == "1"
    performance_canary_mode = os.environ.get("GRPO_PERFORMANCE_CANARY_MODE") == "1"
    method_canary_mode = os.environ.get("GRPO_METHOD_CANARY_MODE") == "1"

    if sum((smoke_mode, canary_mode, performance_canary_mode, method_canary_mode)) > 1:
        raise SystemExit(
            "GRPO smoke, canary, performance-canary and method-canary modes "
            "are mutually exclusive"
        )

    if smoke_mode:
        if int(config.trainer.total_training_steps) != 1:
            raise SystemExit("GRPO smoke mode requires trainer.total_training_steps=1")
        if bool(config.trainer.val_before_train):
            raise SystemExit("GRPO smoke mode requires trainer.val_before_train=false")
        if total_length > 8192:
            raise SystemExit(
                f"GRPO smoke mode sequence length must be <=8192, got {total_length}"
            )

    if canary_mode:
        total_training_steps = int(config.trainer.total_training_steps)
        if not 2 <= total_training_steps <= 20:
            raise SystemExit(
                "GRPO canary mode requires trainer.total_training_steps between 2 and 20"
            )
        if bool(config.trainer.val_before_train):
            raise SystemExit("GRPO canary mode requires trainer.val_before_train=false")
        if total_length > 8192:
            raise SystemExit(
                f"GRPO canary mode sequence length must be <=8192, got {total_length}"
            )

    if performance_canary_mode:
        total_training_steps = int(config.trainer.total_training_steps)
        if not 3 <= total_training_steps <= 10:
            raise SystemExit(
                "GRPO performance-canary mode requires trainer.total_training_steps "
                "between 3 and 10"
            )
        if bool(config.trainer.val_before_train):
            raise SystemExit(
                "GRPO performance-canary mode requires trainer.val_before_train=false"
            )

    if method_canary_mode:
        total_training_steps = int(config.trainer.total_training_steps)
        if not 1 <= total_training_steps <= 5:
            raise SystemExit(
                "GRPO method-canary mode requires trainer.total_training_steps "
                "between 1 and 5"
            )
        if bool(config.trainer.val_before_train):
            raise SystemExit(
                "GRPO method-canary mode requires trainer.val_before_train=false"
            )
        if not bool(model.use_fused_kernels):
            raise SystemExit(
                "GRPO method-canary mode requires model.use_fused_kernels=true "
                "to avoid materializing full-vocabulary logits"
            )
        fused_backend = str(model.fused_kernel_options.get("impl_backend", ""))
        if fused_backend not in {"torch", "triton"}:
            raise SystemExit(
                "GRPO method-canary fused_kernel_options.impl_backend must be "
                "torch or triton"
            )
        if int(config.trainer.save_freq) != 1:
            raise SystemExit(
                "GRPO method-canary mode requires trainer.save_freq=1 so a later "
                "OOM does not discard completed optimizer steps"
            )
    else:
        fused_backend = None

    if response_length > MAX_SAFE_RESPONSE_LENGTH:
        raise SystemExit(
            "unsafe GRPO response budget: "
            f"max_response_length={response_length} exceeds {MAX_SAFE_RESPONSE_LENGTH}"
        )
    if total_length > MAX_SAFE_SEQUENCE_LENGTH:
        raise SystemExit(
            "unsafe GRPO sequence budget: "
            f"max_prompt_length + max_response_length = {total_length}, "
            f"limit is {MAX_SAFE_SEQUENCE_LENGTH}"
        )
    reduced_mode = smoke_mode or canary_mode
    required_token_budget = total_length if reduced_mode else MAX_SAFE_SEQUENCE_LENGTH
    for name, value in (
        ("rollout.max_model_len", int(rollout.max_model_len)),
        ("rollout.max_num_batched_tokens", int(rollout.max_num_batched_tokens)),
        (
            "rollout.log_prob_max_token_len_per_gpu",
            int(rollout.log_prob_max_token_len_per_gpu),
        ),
        ("actor.ppo_max_token_len_per_gpu", int(actor.ppo_max_token_len_per_gpu)),
        ("ref.log_prob_max_token_len_per_gpu", int(reference.log_prob_max_token_len_per_gpu)),
    ):
        if value != required_token_budget:
            raise SystemExit(
                f"unsafe or inconsistent GRPO memory budget: {name} must equal "
                f"{required_token_budget}, got {value}"
            )
    if bool(actor.use_dynamic_bsz):
        raise SystemExit(
            "actor.use_dynamic_bsz must be false so configured PPO micro batches are enforced"
        )
    actor_micro_batch_size = int(actor.ppo_micro_batch_size_per_gpu)
    actor_mini_batch_size = int(actor.ppo_mini_batch_size)
    try:
        gradient_accumulation_steps = ppo_gradient_accumulation_steps(
            actor_mini_batch_size,
            actor_micro_batch_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if bool(rollout.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "rollout.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(rollout.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("rollout.log_prob_micro_batch_size_per_gpu must equal 1")
    if bool(reference.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "ref.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(reference.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("ref.log_prob_micro_batch_size_per_gpu must equal 1")

    print(
        "GRPO training memory budget preflight passed: "
        + json.dumps(
            {
                "max_prompt_length": prompt_length,
                "max_response_length": response_length,
                "max_sequence_length": total_length,
                "smoke_mode": smoke_mode,
                "canary_mode": canary_mode,
                "performance_canary_mode": performance_canary_mode,
                "method_canary_mode": method_canary_mode,
                "actor_fused_kernels": bool(model.use_fused_kernels),
                "actor_fused_kernel_backend": fused_backend,
                "actor_mini_batch_size": actor_mini_batch_size,
                "actor_micro_batch_size_per_gpu": actor_micro_batch_size,
                "actor_gradient_accumulation_steps": gradient_accumulation_steps,
                "actor_dynamic_batch": False,
                "rollout_log_prob_micro_batch_size_per_gpu": 1,
                "rollout_log_prob_dynamic_batch": False,
                "reference_micro_batch_size_per_gpu": 1,
                "reference_dynamic_batch": False,
            },
            sort_keys=True,
        )
    )


def validate_text_only_rollout(config):
    """Require vLLM to omit the unused Qwen3.5 vision tower for shopping."""
    engine_kwargs = config.actor_rollout_ref.rollout.get("engine_kwargs", {}) or {}
    vllm_kwargs = engine_kwargs.get("vllm", {}) or {}
    if vllm_kwargs.get("language_model_only") is not True:
        raise SystemExit(
            "Shopping GRPO requires rollout.engine_kwargs.vllm."
            "language_model_only=true"
        )
    print(
        "text-only rollout preflight passed: "
        + json.dumps({"language_model_only": True}, sort_keys=True)
    )


def validate_rollout_seed_contract(config):
    """Require the seed mapping supported by veRL 0.8's vLLM engine."""
    rollout = config.actor_rollout_ref.rollout
    if "seed" in rollout:
        raise SystemExit(
            "veRL 0.8 RolloutConfig rejects actor_rollout_ref.rollout.seed; "
            "use actor_rollout_ref.rollout.engine_kwargs.vllm.seed"
        )
    engine_kwargs = rollout.get("engine_kwargs", {}) or {}
    vllm_kwargs = engine_kwargs.get("vllm", {}) or {}
    engine_seed = vllm_kwargs.get("seed")
    if isinstance(engine_seed, bool) or not isinstance(engine_seed, int):
        raise SystemExit(
            "Shopping GRPO requires integer "
            "actor_rollout_ref.rollout.engine_kwargs.vllm.seed"
        )
    if engine_seed < 0:
        raise SystemExit("vLLM rollout engine seed must be non-negative")
    engine_replica_count = int(rollout.get("data_parallel_size", 1))
    if engine_replica_count != 1:
        raise SystemExit(
            "explicit vLLM engine seed is admitted only with "
            "rollout.data_parallel_size=1; engine_kwargs would override "
            "veRL's replica-rank seed offset"
        )
    configured_data_seed = config.data.get("seed")
    data_seed = None if configured_data_seed is None else int(configured_data_seed)
    print(
        "rollout seed preflight passed: "
        + json.dumps(
            {
                "data_seed": data_seed,
                "engine_seed": engine_seed,
                "engine_replica_count": engine_replica_count,
                "engine_seed_path": "actor_rollout_ref.rollout.engine_kwargs.vllm.seed",
                "per_request_sampling_seed": None,
                "async_schedule_exact_replay": False,
            },
            sort_keys=True,
        )
    )


def main():
    config = compose_runtime_config(sys.argv[1:])
    validate_environment_contract()
    required_paths = {
        "GRPO_MODEL_PATH": os.environ.get("GRPO_MODEL_PATH"),
        "GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE"),
        "GRPO_VAL_FILE": os.environ.get("GRPO_VAL_FILE"),
    }
    missing = []
    for name, value in required_paths.items():
        if not value:
            missing.append(name)
        elif name == "GRPO_MODEL_PATH" and not Path(value).is_dir():
            missing.append(name)
        elif name != "GRPO_MODEL_PATH" and not Path(value).is_file():
            missing.append(name)
    if missing:
        raise SystemExit("missing GRPO input path(s): " + ", ".join(missing))
    validate_training_memory_budget(config)
    validate_text_only_rollout(config)
    validate_rollout_seed_contract(config)
    validate_agent_progrpo(config)
    validate_shopping_gigpo(config)
    validate_shopping_graphgpo(config)

    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit(
            "incompatible Python: expected >=3.10,<3.13, "
            f"got {sys.version.split()[0]}"
        )

    installed = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing GRPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible GRPO dependency: expected {package}=={expected}, got {installed[package]}"
            )
    try:
        import torch
        import verl
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
        from verl.tools.base_tool import BaseTool
        from verl.utils.tracking import Tracking
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL 0.8 install: required AgentLoop/Tool APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    verl_source = Path(verl.__file__).resolve()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the GRPO environment")
    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or not issubclass(ShopSimulatorTool, BaseTool)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("veRL 0.8 built-in qwen3_coder parser is unavailable")
    if "swanlab" not in Tracking.supported_backend:
        raise SystemExit("veRL 0.8 SwanLab tracking backend is unavailable")
    validate_model_runtime(required_paths["GRPO_MODEL_PATH"])
    validate_peft_lora_runtime(config)
    validate_dynamic_sampling(config, verl_source, installed)
    validate_swanlab_tracking(config)
    install_torch_padding_fallback()
    print(
        "GRPO runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in installed.items())
        + f", source={verl_source}"
    )


if __name__ == "__main__":
    main()
