"""Resize the video stream of an H3 Motion Context AV latent.

H3 Motion Context Load Latent hands back {"samples": [video, audio]}, a
plain list rather than a real tensor or a comfy NestedTensor -- see that
pack's nodes.py, which does this on purpose so the payload cannot be
mistaken for a decodable latent by any other node. That also means the
stock Upscale Latent nodes cannot touch it: they call .shape on
samples["samples"] and a list has none.

This node exists for one situation: you saved a context latent (or are
about to) at one resolution and want to chain-test continuity at a
different one, without regenerating the previous clip. It unpacks the
list (or a live NestedTensor, before it is even saved), resizes only the
video stream spatially with comfy's own upscale, and repacks the same
list format so H3 Motion Context / H3 Motion Context Save Latent accept
it unchanged. Audio has no width/height, so it passes through untouched
and the frame/step count (dim 2) is never touched either -- only H and W
change, which is what MiniMaxH3MotionContext actually checks for a match.

Caveat that does not go away: the pinned frames are never re-denoised.
Any resize artifact (blur, ringing) lands straight in the final video,
unlike a normal Hi-Res Fix which cleans up the upscale with a second
sampling pass. Fine for testing whether the chain joins correctly; worth
re-checking visually before treating a resized context as final.
"""

import gc
import logging

import torch

import comfy.model_management as mm
import comfy.utils

_LOG = logging.getLogger("h3_motion_context_resize")

H3_VAE_DOWNSAMPLE = 16


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    if isinstance(samples, (list, tuple)):
        return list(samples)
    raise ValueError(
        "H3MotionContextResize: expected an H3 AV latent (nested video/audio "
        "pair, or the list saved by H3 Motion Context Load Latent), got %r"
        % type(samples))


class MiniMaxH3MotionContextResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "An H3 AV latent: the output of H3 Motion "
                               "Context Load Latent, or a live sampler "
                               "output before Save Latent."}),
                "upscale_method": (
                    ["bislerp", "bicubic", "bilinear", "area", "nearest-exact"],
                    {"default": "bislerp"}),
                "width": ("INT", {
                    "default": 512, "min": 16, "max": 8192, "step": 16,
                    "tooltip": "Target pixel width. Must equal the width of "
                               "the NEW clip you will chain from this "
                               "context, or MiniMaxH3MotionContext will "
                               "reject the mismatch."}),
                "height": ("INT", {
                    "default": 512, "min": 16, "max": 8192, "step": 16,
                    "tooltip": "Target pixel height. Must equal the height "
                               "of the NEW clip."}),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Where the interpolation runs. cuda uses "
                               "VRAM and stays fast even on long clips; cpu "
                               "runs in system RAM, which is what spikes "
                               "when the whole run is resized in one shot."}),
                "frame_chunk": ("INT", {
                    "default": 8, "min": 1, "max": 512,
                    "tooltip": "Latent steps resized per pass. common_upscale "
                               "folds every step of the run into one batch "
                               "before interpolating, so resizing all of "
                               "them at once briefly allocates several "
                               "copies sized for the WHOLE clip. Chunking "
                               "bounds that to frame_chunk steps at a time. "
                               "Lower it if memory is still tight, raise it "
                               "for speed on short clips."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "resize"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Resize only the video stream of an H3 Motion Context AV latent, "
        "for chaining continuity tests at a resolution other than the one "
        "the context was saved at. Audio and frame count are untouched. "
        "The result is never re-denoised, so resize artifacts are visible "
        "in the final render -- use this to validate the join, not as a "
        "substitute for regenerating the previous clip at the final "
        "resolution.")

    def resize(self, latent, upscale_method, width, height,
               device="cuda", frame_chunk=8):
        parts = _streams_from_latent(latent)
        if not parts:
            raise ValueError("H3MotionContextResize: latent has no streams")

        video = parts[0]
        was_4d = (video.ndim == 4)
        if was_4d:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(
                "H3MotionContextResize: expected video latent [B,C,T,H,W], "
                "got shape %s" % (tuple(video.shape),))

        w_latent = max(1, round(width / H3_VAE_DOWNSAMPLE))
        h_latent = max(1, round(height / H3_VAE_DOWNSAMPLE))
        aligned_w = w_latent * H3_VAE_DOWNSAMPLE
        aligned_h = h_latent * H3_VAE_DOWNSAMPLE
        if aligned_w != width or aligned_h != height:
            _LOG.warning(
                "h3_motion_context_resize: %dx%d is not a multiple of %d; "
                "snapping to %dx%d", width, height, H3_VAE_DOWNSAMPLE,
                aligned_w, aligned_h)

        if w_latent == video.shape[-1] and h_latent == video.shape[-2]:
            return (latent,)

        _LOG.info(
            "h3_motion_context_resize: video latent %dx%d -> %dx%d "
            "(pixels %dx%d -> %dx%d)",
            video.shape[-1], video.shape[-2], w_latent, h_latent,
            video.shape[-1] * H3_VAE_DOWNSAMPLE,
            video.shape[-2] * H3_VAE_DOWNSAMPLE, aligned_w, aligned_h)

        orig_device = video.device
        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available()
                            else "cpu")
        if device == "cuda" and dev.type == "cpu":
            _LOG.warning("h3_motion_context_resize: cuda requested but not "
                         "available, falling back to cpu")

        total_t = int(video.shape[2])
        chunks = []
        for start in range(0, total_t, frame_chunk):
            end = min(total_t, start + frame_chunk)
            piece = video[:, :, start:end].to(dev)
            piece = comfy.utils.common_upscale(
                piece, w_latent, h_latent, upscale_method, "disabled")
            chunks.append(piece.to(orig_device))
            del piece
            if dev.type == "cuda":
                mm.soft_empty_cache()

        del video
        video = torch.cat(chunks, dim=2) if len(chunks) > 1 else chunks[0]
        del chunks
        gc.collect()
        if dev.type == "cuda":
            mm.soft_empty_cache()

        if was_4d:
            video = video.squeeze(0)

        parts[0] = video
        out = latent.copy()
        # Deliberately a plain list, matching what H3 Motion Context Load
        # Latent produces: both MiniMaxH3MotionContext and
        # MiniMaxH3MotionContextSaveLatent accept it via the same
        # unbind-or-list check, and it is the format that fails loudly if
        # wired anywhere else by mistake.
        out["samples"] = parts
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContextResize": MiniMaxH3MotionContextResize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContextResize": "H3 Motion Context Resize",
}
