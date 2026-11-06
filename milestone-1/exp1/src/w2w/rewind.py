"""Paint, record, rewind and branch.

`Painter` runs its own sampling loop with the same arithmetic as Diffusers'
StableDiffusionPipeline (DDIM, eta 0, classifier-free guidance, no guidance rescale), so
a painting made here is pixel-identical to the pipeline's. Owning the loop gives three
things the pipeline doesn't:

  * the latent state before every step is kept, so any step can be returned to;
  * the model's guess of the finished picture at every step (DDIM's predicted x0) is
    decoded, so the painting can be watched as it forms;
  * a painting can branch: start from a saved step of another painting, with a changed
    prompt, and carry on from there.

DDIM with eta 0 is deterministic and keeps no state between steps, which is what makes
resuming from a saved step exact.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .attention import AttentionStore, RecordingCrossAttnProcessor


@dataclass
class Timeline:
    prompt: str
    seed: int
    start: int                                     # first step this painting computed itself
    latents: dict = field(default_factory=dict)    # step -> latent before that step (CPU)
    previews: dict = field(default_factory=dict)   # step -> PIL guess of the final picture after that step
    final: object = None                           # PIL image
    maps: np.ndarray | None = None                 # (steps computed, 77, 64, 64) float16
    masks: dict = field(default_factory=dict)      # step -> (64, 64) bool blend mask, when blending


class Painter:
    def __init__(self, pipe, device: str = "cuda", steps: int = 30, guidance: float = 7.5):
        from diffusers import DDIMScheduler
        if not isinstance(pipe.scheduler, DDIMScheduler):
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe, self.device, self.steps, self.guidance = pipe, device, steps, guidance
        self.store = AttentionStore()
        self.original = dict(pipe.unet.attn_processors)
        self.recording = {n: (RecordingCrossAttnProcessor(self.store, inner=p) if "attn2" in n else p)
                          for n, p in self.original.items()}
        self._mode = None

    def _record(self, on: bool):
        if on != self._mode:
            # set_attn_processor empties the dict it is given, so it always gets a copy
            self.pipe.unet.set_attn_processor(dict(self.recording if on else self.original))
            self._mode = on

    @torch.no_grad()
    def embed(self, prompt: str):
        pe, ne = self.pipe.encode_prompt(prompt, self.device, 1, True)
        return pe, ne

    def noise(self, seed: int):
        n = self.pipe.unet.config.sample_size
        g = torch.Generator(self.device).manual_seed(seed)
        return torch.randn((1, 4, n, n), generator=g, device=self.device, dtype=self.pipe.unet.dtype)

    @torch.no_grad()
    def decode(self, latents, safe: bool = False):
        """Latents to a PIL image. With `safe`, the pipeline's safety checker runs first and a
        flagged image comes back black, exactly as the pipeline does it."""
        vae = self.pipe.vae
        img = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        denorm = [True]
        if safe:
            img, flags = self.pipe.run_safety_checker(img, self.device, img.dtype)
            if flags is not None:
                denorm = [not f for f in flags]
        return self.pipe.image_processor.postprocess(img, output_type="pil", do_denormalize=denorm)[0]

    @torch.no_grad()
    def paint(self, prompt: str, seed: int, branch_from: Timeline | None = None, at: int = 0,
              record: bool = True, previews: bool = True, safe: bool = False, on_step=None,
              eta: float = 0.0, noise_seed: int | None = None, blend: dict | None = None) -> Timeline:
        """Paint `prompt` from noise, or branch from `branch_from` at step `at` with the new prompt.

        on_step(step, preview, maps) is called after every step, where `maps` is that step's
        (77, 64, 64) attention (None when not recording). `safe` runs the safety checker on
        every decoded picture.

        eta > 0 makes the steps stochastic (DDIM's eta; 1.0 behaves like DDPM), with fresh noise
        drawn from `noise_seed`, so repainting from the same step gives a different finish.

        blend = {"orig": Timeline, "slots_new": [...], "slots_orig": [...], "th": 0.3} keeps
        everything outside the edited words' region from the original painting at every step
        (Prompt-to-Prompt's local blending, Hertz et al. 2022). The region is where the edited
        words (in this painting) and the replaced or edited object (in the original) attend,
        averaged over the steps so far, scaled to 0..1, thresholded and dilated by one latent pixel.
        Needs `record=True` and the original's maps and latents."""
        assert blend is None or record, "local blending reads this painting's attention, so record=True"
        g = torch.Generator(self.device).manual_seed(noise_seed) if (eta > 0 and noise_seed is not None) else None
        acc_new = acc_old = None
        pe, ne = self.embed(prompt)
        emb = torch.cat([ne, pe])
        sch = self.pipe.scheduler
        sch.set_timesteps(self.steps, device=self.device)
        ts = sch.timesteps
        if branch_from is None or at == 0:
            lat = self.noise(seed) * sch.init_noise_sigma
            at = 0
        else:
            assert at in branch_from.latents, f"step {at} was not saved in the painting to branch from"
            lat = branch_from.latents[at].to(self.device)
        self._record(record)
        self.store.reset()
        tl = Timeline(prompt=prompt, seed=seed, start=at)
        for i in range(at, len(ts)):
            t = ts[i]
            tl.latents[i] = lat.detach().cpu()
            inp = sch.scale_model_input(torch.cat([lat] * 2), t)
            pred = self.pipe.unet(inp, t, encoder_hidden_states=emb, return_dict=False)[0]
            uncond, cond = pred.chunk(2)
            pred = uncond + self.guidance * (cond - uncond)
            out = sch.step(pred, t, lat, eta=eta, generator=g, return_dict=True)
            lat = out.prev_sample
            if record:
                self.store.end_step()
            if blend is not None:
                o = blend["orig"]
                new = self.store.maps[self.store.step - 1][blend["slots_new"]].float().sum(0)
                old = torch.from_numpy(o.maps[i - o.start][blend["slots_orig"]].astype(np.float32)).sum(0)
                acc_new = new if acc_new is None else acc_new + new
                acc_old = old if acc_old is None else acc_old + old
                norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
                m = torch.maximum(norm(acc_new), norm(acc_old))
                m = F.interpolate(m[None, None], size=lat.shape[-2:], mode="area")    # attention grid to latent grid
                m = (m > blend.get("th", 0.3)).float()
                m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
                tl.masks[i] = m[0, 0].bool()
                m = m.to(self.device, lat.dtype)
                lat = m * lat + (1 - m) * o.latents[i + 1].to(self.device, lat.dtype)
            if previews:
                tl.previews[i] = self.decode(out.pred_original_sample, safe=safe)
            if on_step is not None:
                on_step(i, tl.previews.get(i), self.store.maps.get(self.store.step - 1) if record else None)
        tl.latents[len(ts)] = lat.detach().cpu()
        tl.final = self.decode(lat, safe=safe)
        if record:
            tl.maps = self.store.stacked()
        return tl
