"""Standalone companion to ComfyUI-H3-Motion-Context: resize the video
stream of its AV latent format so a saved context can be reused for
chain-continuity testing at a different resolution. Kept as a separate
pack (not a patch to that repo) so a `git pull` there never conflicts
with this file.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
