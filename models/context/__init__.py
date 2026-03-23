from __future__ import annotations

from .large_kernel_context import LargeKernelContext
from .transformer_context import TransformerContext


def build_context_module(cfg, channels: int):
    """Build the context module selected by cfg.model.context_module."""
    context_name = str(getattr(cfg.model, 'context_module', 'large_kernel')).lower()
    if context_name == 'large_kernel':
        return LargeKernelContext(channels)
    if context_name == 'transformer':
        return TransformerContext(cfg, channels)
    raise ValueError(
        f'Unsupported context module: {context_name}. '
        'Expected one of: large_kernel, transformer.'
    )


__all__ = ['LargeKernelContext', 'TransformerContext', 'build_context_module']
