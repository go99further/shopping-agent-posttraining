"""Narrow PEFT compatibility for veRL 0.8 LoRA construction.

veRL 0.8 always forwards ``target_parameters`` to ``peft.LoraConfig`` even
when the configured value is ``None``.  PEFT 0.15.2 predates that keyword,
but is otherwise the compatible PEFT release in the pinned Qwen3.5 runtime.
The shim below accepts only the semantically empty value; a real
``target_parameters`` request is rejected instead of being silently ignored.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any


PATCH_MARKER_ATTR = "_shopping_grpo_peft_target_parameters_compat_v1"
PATCH_MODE_ATTR = "_shopping_grpo_peft_target_parameters_mode"


def install_peft_lora_config_compat(lora_config_cls: type[Any] | None = None) -> str:
    """Return ``native`` or install the guarded ``none-only`` compatibility shim."""
    if lora_config_cls is None:
        from peft import LoraConfig

        lora_config_cls = LoraConfig

    if "target_parameters" in inspect.signature(lora_config_cls.__init__).parameters:
        setattr(lora_config_cls, PATCH_MODE_ATTR, "native")
        return "native"

    if getattr(lora_config_cls, PATCH_MARKER_ATTR, False):
        return str(getattr(lora_config_cls, PATCH_MODE_ATTR, "none-only-shim"))

    original_init = lora_config_cls.__init__

    @functools.wraps(original_init)
    def compatible_init(
        self: Any,
        *args: Any,
        target_parameters: Any = None,
        **kwargs: Any,
    ) -> None:
        if target_parameters is not None:
            raise TypeError(
                "PEFT compatibility shim only accepts target_parameters=None; "
                "upgrade the pinned runtime before training parameter-level LoRA"
            )
        original_init(self, *args, **kwargs)

    lora_config_cls.__init__ = compatible_init
    setattr(lora_config_cls, PATCH_MARKER_ATTR, True)
    setattr(lora_config_cls, PATCH_MODE_ATTR, "none-only-shim")
    return "none-only-shim"
