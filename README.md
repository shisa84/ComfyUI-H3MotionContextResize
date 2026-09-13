# ComfyUI-H3-MotionContext-Resize

[📺 Demo video](https://youtu.be/UkBW1Jdk5n0)

A single node, **H3 Motion Context Resize**, that resizes the video stream of an
[H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) AV
latent so a saved context can be reused for chain-continuity testing at a resolution
other than the one it was saved at.

## Why

`H3 Motion Context Load Latent` returns `{"samples": [video, audio]}` — a plain list,
not a tensor or a comfy `NestedTensor` — on purpose, so the payload can't be mistaken
for a decodable latent by any other node. That also means the stock `Upscale Latent`
nodes can't touch it: they call `.shape` on `samples["samples"]` and a list has none.

This node exists for one situation: you saved a context latent (or are about to) at
one resolution and want to chain-test continuity at a different one, without
regenerating the previous clip. It unpacks the list (or a live `NestedTensor`, before
it is even saved), resizes only the video stream spatially, and repacks the same list
format so `H3 Motion Context` / `H3 Motion Context Save Latent` accept it unchanged.
Audio and the frame/step count are never touched — only width and height change,
which is what `MiniMaxH3MotionContext` actually checks for a match.

**Caveat that does not go away:** the pinned frames are never re-denoised. Any resize
artifact (blur, ringing) lands straight in the final video, unlike a normal Hi-Res Fix
which cleans the upscale up with a second sampling pass. Use this to validate that a
chain joins correctly; re-check visually before treating a resized context as final.

## Usage

```
H3 Motion Context Load Latent -> H3 Motion Context Resize -> context_latent
                                                               (on H3 Motion Context)
```

Set `width`/`height` to the exact pixel resolution of the **new** clip you are
generating — `MiniMaxH3MotionContext` rejects a `context_latent` whose resolution
doesn't match.

## Inputs

| Input | Description |
|---|---|
| `latent` | An H3 AV latent: output of `H3 Motion Context Load Latent`, or a live sampler output before Save Latent. |
| `upscale_method` | `bislerp` (default) is the only method designed for latents rather than pixels — it slerps channel vectors instead of averaging them, which keeps the result in-distribution. `area`, `bicubic`, `bilinear`, `nearest-exact` are also available. |
| `width`, `height` | Target pixel size, must be a multiple of 16 (H3's VAE downsample factor); off-grid values are snapped and logged. |
| `device` | `cuda` (default) runs the interpolation in VRAM instead of system RAM. |
| `frame_chunk` | Latent steps resized per pass (default 8), bounding peak memory on long source clips instead of folding every step into one giant batch. |

## Requirements

- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
  installed, for the AV latent format this node reads and writes. Not imported
  directly — kept as a separate pack so a `git pull` there never conflicts with this
  repo.
- 8GB VRAM minimum (tested).

## Example Workflow

Load a workflow from `example_workflows/` (canvas format, drag onto the canvas)
to see the node wired up in a full chain-continuity pipeline.
