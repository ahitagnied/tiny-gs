import functools
import json
import logging
import math, struct, random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
from tqdm import trange
import torchvision.transforms.functional as TF
from torch import nn
from gsplat import rasterization

log = logging.getLogger("tiny_gs")


# Data primitives

@dataclass
class Camera:
    R: torch.Tensor      # (3, 3) world-to-cam rotation
    T: torch.Tensor      # (3,) world-to-cam translation
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    image: torch.Tensor  # (3, H, W) RGB in [0, 1]
    name: str

    @property
    def device(self) -> torch.device:
        return self.R.device


# Scene loaders

def qvec2mat(q):
    w, x, y, z = q
    return np.float32([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                       [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                       [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def _rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

def read_cameras_bin(path):
    out = {}
    with open(path, "rb") as f:
        for _ in range(_rd(f, "<Q")[0]):
            cid, mid = _rd(f, "<ii"); w, h = _rd(f, "<QQ")
            out[cid] = (mid, int(w), int(h), _rd(f, f"<{[3,4,4,5,8,8,8,12][mid]}d"))
    return out

def read_images_bin(path):
    out = {}
    with open(path, "rb") as f:
        for _ in range(_rd(f, "<Q")[0]):
            iid = _rd(f, "<I")[0]; q, t = _rd(f, "<4d"), _rd(f, "<3d")
            cid = _rd(f, "<I")[0]; name = b""
            while (c := f.read(1)) != b"\x00": name += c
            f.read(_rd(f, "<Q")[0] * 24)
            out[iid] = (q, t, cid, name.decode())
    return out

# COLMAP points3D.bin layout (colmap/src/colmap/scene/reconstruction.cc):
#   uint64 num_points
#   per point: uint64 id, 3*double xyz, 3*uint8 rgb, double error, uint64 track_len,
#              track_len * (uint32 image_id, uint32 point2D_idx)
_COUNT          = struct.Struct("<Q")
_POINT_HEADER   = struct.Struct("<Q3d3BdQ")
_TRACK_ENTRY_SZ = struct.calcsize("<II")

def read_points_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse COLMAP points3D.bin -> (xyz[N,3] float32, rgb[N,3] float32 in [0,1])."""
    buf = Path(path).read_bytes()
    if len(buf) < _COUNT.size:
        raise ValueError(f"{path}: {len(buf)} B is too small for points3D.bin header")

    (n,) = _COUNT.unpack_from(buf, 0)
    xyz = np.empty((n, 3), dtype=np.float32)
    rgb = np.empty((n, 3), dtype=np.float32)
    off = _COUNT.size

    for i in range(n):
        if off + _POINT_HEADER.size > len(buf):
            raise ValueError(f"{path}: truncated at point {i}/{n} (offset {off}/{len(buf)})")
        _id, x, y, z, r, g, b, _err, tlen = _POINT_HEADER.unpack_from(buf, off)
        xyz[i] = (x, y, z)
        rgb[i] = (r, g, b)
        off += _POINT_HEADER.size + tlen * _TRACK_ENTRY_SZ

    if off != len(buf):
        raise ValueError(f"{path}: parsed {off} B but file is {len(buf)} B (corrupt?)")

    rgb *= 1.0 / 255.0
    return xyz, rgb

def _make_camera(R, T, fx, fy, cx, cy, w, h, img, name, resolution, device):
    """Resize image + intrinsics to `resolution` and pack into a Camera."""
    if resolution != 1:
        nw, nh = max(1, round(w / resolution)), max(1, round(h / resolution))
        img = img.resize((nw, nh), Image.LANCZOS)
        fx, fy, cx, cy = fx / resolution, fy / resolution, cx / resolution, cy / resolution
        w, h = nw, nh
    return Camera(
        R=torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device),
        T=torch.as_tensor(np.asarray(T), dtype=torch.float32, device=device),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        W=int(w), H=int(h),
        image=TF.to_tensor(img).to(device), name=name,
    )

def load_scene(data_dir, resolution=1, device="cuda"):
    base = Path(data_dir); sp = base / "sparse" / "0"
    cams, imgs = read_cameras_bin(sp/"cameras.bin"), read_images_bin(sp/"images.bin")
    xyz, rgb = read_points_bin(sp/"points3D.bin")
    cameras = []
    for q, t, cid, name in sorted(imgs.values(), key=lambda x: x[3]):
        mid, w, h, p = cams[cid]
        if   mid == 0: fx = fy = p[0]; cx, cy = p[1], p[2]
        elif mid == 1: fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:          fx = fy = p[0]; cx, cy = p[2], p[3]
        img = Image.open(base / "images" / name).convert("RGB")
        cameras.append(_make_camera(qvec2mat(q), t, fx, fy, cx, cy, w, h,
                                    img, name, resolution, device))
    return cameras, xyz, rgb

def scene_extent(cameras):
    centers = np.array([(-c.R.cpu().numpy().T @ c.T.cpu().numpy()) for c in cameras])
    return float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)


# Gaussian model

C0 = 0.28209479177387814  # SH band-0 basis: rgb = 0.5 + C0 * f_dc
def rgb2sh(rgb): return (rgb - 0.5) / C0

class Gaussians:
    def __init__(self, sh_degree=3, device="cuda"):
        self.device = torch.device(device)
        self.max_sh, self.active_sh = sh_degree, 0
        self.xyz = self.f_dc = self.f_rest = self.scale = self.rot = self.opacity = None
        self.grad_accum = self.denom = self.max_r2d = self.opt = None

    @property
    def N(self): return self.xyz.shape[0]

    def from_pcd(self, xyz_np, rgb_np):
        N    = len(xyz_np)
        dev  = self.device
        sq_d = np.mean(KDTree(xyz_np).query(xyz_np, k=4)[0][:, 1:] ** 2, axis=1).clip(1e-7)
        self.xyz     = nn.Parameter(torch.tensor(xyz_np, dtype=torch.float32, device=dev))
        self.f_dc    = nn.Parameter(rgb2sh(torch.tensor(rgb_np, dtype=torch.float32, device=dev)).unsqueeze(1).contiguous())
        self.f_rest  = nn.Parameter(torch.zeros(N, (self.max_sh + 1) ** 2 - 1, 3, device=dev))
        self.scale   = nn.Parameter(torch.tensor(np.log(np.sqrt(sq_d)), dtype=torch.float32, device=dev)
                                    .unsqueeze(1).expand(-1, 3).contiguous())
        self.rot     = nn.Parameter(torch.cat([torch.ones(N, 1, device=dev),
                                               torch.zeros(N, 3, device=dev)], dim=1))
        self.opacity = nn.Parameter(torch.logit(torch.full((N, 1), 0.1, device=dev)))
        self.grad_accum = torch.zeros(N, 1, device=dev)
        self.denom      = torch.zeros(N, 1, device=dev)
        self.max_r2d    = torch.zeros(N,    device=dev)

    def setup_opt(self, cfg, spatial_scale):
        self.opt = torch.optim.Adam([
            {"params": [self.xyz],     "lr": cfg.pos_init * spatial_scale, "name": "xyz"},
            {"params": [self.f_dc],    "lr": cfg.feature_lr,               "name": "f_dc"},
            {"params": [self.f_rest],  "lr": cfg.feature_lr / 20,          "name": "f_rest"},
            {"params": [self.scale],   "lr": cfg.scaling_lr,               "name": "scaling"},
            {"params": [self.rot],     "lr": cfg.rotation_lr,              "name": "rotation"},
            {"params": [self.opacity], "lr": cfg.opacity_lr,               "name": "opacity"},
        ], eps=1e-15)

    def update_xyz_lr(self, step, cfg, spatial_scale):
        t  = min(step / cfg.pos_max_steps, 1.0)
        lr = math.exp(math.log(cfg.pos_init * spatial_scale) * (1 - t) +
                      math.log(cfg.pos_final * spatial_scale) * t)
        for g in self.opt.param_groups:
            if g["name"] == "xyz": g["lr"] = lr

    def _prune_opt(self, keep):
        out = {}
        for g in self.opt.param_groups:
            p = g["params"][0]; s = self.opt.state.pop(p, None)
            if s is not None:
                s["exp_avg"], s["exp_avg_sq"] = s["exp_avg"][keep], s["exp_avg_sq"][keep]
            g["params"][0] = pn = nn.Parameter(p[keep].requires_grad_(True))
            if s is not None: self.opt.state[pn] = s
            out[g["name"]] = pn
        return out

    def _cat_opt(self, extras):
        out = {}
        for g in self.opt.param_groups:
            ext = extras[g["name"]]
            p = g["params"][0]; s = self.opt.state.pop(p, None)
            if s is not None:
                s["exp_avg"]    = torch.cat([s["exp_avg"],    torch.zeros_like(ext)])
                s["exp_avg_sq"] = torch.cat([s["exp_avg_sq"], torch.zeros_like(ext)])
            g["params"][0] = pn = nn.Parameter(torch.cat([p, ext]).requires_grad_(True))
            if s is not None: self.opt.state[pn] = s
            out[g["name"]] = pn
        return out

    def _sync(self, t):
        self.xyz, self.f_dc, self.f_rest    = t["xyz"], t["f_dc"], t["f_rest"]
        self.scale, self.rot, self.opacity  = t["scaling"], t["rotation"], t["opacity"]

    def prune(self, keep):
        self._sync(self._prune_opt(keep))
        self.grad_accum = self.grad_accum[keep]
        self.denom      = self.denom[keep]
        self.max_r2d    = self.max_r2d[keep]

    def cat(self, extras):
        self._sync(self._cat_opt(extras))
        n = extras["xyz"].shape[0]
        dev = self.device
        self.grad_accum = torch.cat([self.grad_accum, torch.zeros(n, 1, device=dev)])
        self.denom      = torch.cat([self.denom,      torch.zeros(n, 1, device=dev)])
        self.max_r2d    = torch.cat([self.max_r2d,    torch.zeros(n,    device=dev)])

    def accumulate_stats(self, info, W, H):
        radii = info["radii"].squeeze(0).float()
        # gsplat: [N, 2] ellipse axes; older / other paths may use [N]
        if radii.dim() == 1:
            vis = radii > 0
            r2d = radii
        else:
            vis = (radii > 0).all(dim=-1)
            r2d = radii.max(dim=-1).values
        self.max_r2d[vis] = torch.maximum(self.max_r2d[vis], r2d[vis])
        # gsplat returns means2d in pixel coords, so .grad is in 1/pixel units.
        # INRIA's densify_grad_threshold (0.0002) was tuned on NDC-space grads,
        # which are W/2 (resp. H/2) larger. Rescale here so the threshold has
        # the same meaning as in INRIA / gsplat's DefaultStrategy.
        g2d = info["means2d"].grad.squeeze(0).clone()
        g2d[..., 0] *= 0.5 * W
        g2d[..., 1] *= 0.5 * H
        self.grad_accum[vis] += g2d[vis].norm(dim=-1, keepdim=True)
        self.denom[vis]      += 1

    @staticmethod
    def _build_rotation(r):
        # Batched torch mirror of `qvec2mat`; (w, x, y, z) -> R.
        q = F.normalize(r, dim=-1); w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
        R = torch.zeros(len(r), 3, 3, device=r.device)
        R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
        R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
        R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
        return R

    def _clone(self, grads, grad_thr, pct, ext):
        mask = (grads >= grad_thr) & (torch.exp(self.scale).max(dim=1).values <= pct * ext)
        self.cat({"xyz": self.xyz[mask], "f_dc": self.f_dc[mask], "f_rest": self.f_rest[mask],
                  "scaling": self.scale[mask], "rotation": self.rot[mask], "opacity": self.opacity[mask]})

    def _split(self, grads, grad_thr, pct, ext, K=2):
        dev    = self.device
        padded = torch.zeros(self.N, device=dev); padded[:len(grads)] = grads
        mask   = (padded >= grad_thr) & (torch.exp(self.scale).max(dim=1).values > pct * ext)

        stds    = torch.exp(self.scale[mask]).repeat(K, 1)
        samples = torch.normal(torch.zeros_like(stds), stds)
        Rmat    = self._build_rotation(self.rot[mask]).repeat(K, 1, 1)

        self.cat({
            "xyz":      torch.bmm(Rmat, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[mask].repeat(K, 1),
            "f_dc":     self.f_dc[mask].repeat(K, 1, 1),
            "f_rest":   self.f_rest[mask].repeat(K, 1, 1),
            "scaling":  torch.log(torch.exp(self.scale[mask]).repeat(K, 1) / (0.8 * K)),
            "rotation": self.rot[mask].repeat(K, 1),
            "opacity":  self.opacity[mask].repeat(K, 1),
        })
        self.prune(~torch.cat([mask, torch.zeros(K * mask.sum(), dtype=torch.bool, device=dev)]))

    def densify_and_prune(self, ext, cfg, step):
        grads = (self.grad_accum / self.denom).squeeze()
        grads[grads.isnan() | grads.isinf()] = 0.0
        self._clone(grads, cfg.grad_threshold, cfg.percent_dense, ext)
        self._split(grads, cfg.grad_threshold, cfg.percent_dense, ext)

        prune = (torch.sigmoid(self.opacity) < cfg.min_opacity).squeeze()
        if step > cfg.opacity_reset_interval:
            # INRIA 3DGS used ~20 px; gsplat ellipse radii are usually much larger — same cutoff deletes almost all splats.
            prune |= self.max_r2d > cfg.max_screen_size
            prune |= torch.exp(self.scale).max(dim=1).values > 0.1 * ext
        if (~prune).sum() == 0:
            prune = torch.zeros_like(prune)
        self.prune(~prune)
        torch.cuda.empty_cache()

    def reset_opacity(self):
        new_val = torch.logit(torch.min(torch.sigmoid(self.opacity),
                                        torch.full_like(self.opacity, 0.01)))
        for g in self.opt.param_groups:
            if g["name"] != "opacity": continue
            p = g["params"][0]; s = self.opt.state.pop(p, None)
            if s is not None:
                s["exp_avg"], s["exp_avg_sq"] = torch.zeros_like(new_val), torch.zeros_like(new_val)
            g["params"][0] = pn = nn.Parameter(new_val.requires_grad_(True))
            if s is not None: self.opt.state[pn] = s
            self.opacity = pn

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fdc   = self.f_dc.detach().transpose(1, 2).flatten(1).cpu().numpy()
        frest = self.f_rest.detach().transpose(1, 2).flatten(1).cpu().numpy()
        data  = np.concatenate([self.xyz.detach().cpu().numpy(), fdc, frest,
                                self.opacity.detach().cpu().numpy(),
                                self.scale.detach().cpu().numpy(),
                                self.rot.detach().cpu().numpy()], axis=1)
        cols  = (["x", "y", "z"] + [f"f_dc_{i}"   for i in range(fdc.shape[1])]
                 + [f"f_rest_{i}" for i in range(frest.shape[1])] + ["opacity"]
                 + [f"scale_{i}"  for i in range(3)] + [f"rot_{i}" for i in range(4)])
        arr = np.array([tuple(r) for r in data], dtype=[(c, "f4") for c in cols])
        PlyData([PlyElement.describe(arr, "vertex")]).write(path)


# Rendering & loss

def render(cam, gs, bg):
    device = cam.device
    vmat = torch.eye(4, device=device)
    vmat[:3, :3] = cam.R; vmat[:3, 3] = cam.T
    K = torch.tensor([[cam.fx, 0, cam.cx],
                      [0, cam.fy, cam.cy],
                      [0, 0, 1]], device=device)
    renders, _, info = rasterization(
        means=gs.xyz, quats=F.normalize(gs.rot, dim=-1),
        scales=torch.exp(gs.scale), opacities=torch.sigmoid(gs.opacity).squeeze(-1),
        colors=torch.cat([gs.f_dc, gs.f_rest], dim=1),
        viewmats=vmat.unsqueeze(0), Ks=K.unsqueeze(0).float(),
        width=cam.W, height=cam.H, sh_degree=gs.active_sh,
        near_plane=0.01, backgrounds=bg.unsqueeze(0), packed=False)
    return renders[0].permute(2, 0, 1), info

@functools.cache
def _ssim_window(device):
    x = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-x ** 2 / 4.5); g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).to(device)

def ssim(a, b):
    w  = _ssim_window(a.device); C1, C2 = 0.01 ** 2, 0.03 ** 2
    ma = F.conv2d(a,   w, padding=5, groups=3)
    mb = F.conv2d(b,   w, padding=5, groups=3)
    sa = F.conv2d(a*a, w, padding=5, groups=3) - ma ** 2
    sb = F.conv2d(b*b, w, padding=5, groups=3) - mb ** 2
    ab = F.conv2d(a*b, w, padding=5, groups=3) - ma * mb
    return ((2*ma*mb + C1) * (2*ab + C2) / ((ma**2 + mb**2 + C1) * (sa + sb + C2))).mean()


# Evaluation

@torch.no_grad()
def save_test_renders(test_cams, gs, bg, model_path, step):
    if not test_cams:
        return
    out_dir = Path(model_path) / f"test/iteration_{step}"
    (out_dir / "renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    psnrs, ssims = [], []
    for i, cam in enumerate(test_cams):
        img, _ = render(cam, gs, bg)
        img = img.clamp(0, 1)
        gt  = cam.image.clamp(0, 1)
        mse = F.mse_loss(img, gt).item()
        psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))
        ssims.append(ssim(img.unsqueeze(0), gt.unsqueeze(0)).item())
        stem = Path(cam.name).stem or f"{i:05d}"
        TF.to_pil_image(img.cpu()).save(out_dir / "renders" / f"{stem}.png")
        TF.to_pil_image(gt.cpu()).save(out_dir / "gt"      / f"{stem}.png")

    metrics = {"psnr":   round(sum(psnrs) / len(psnrs), 3),
               "ssim":   round(sum(ssims) / len(ssims), 4),
               "n_test": len(psnrs)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("  [iter %d] test PSNR = %.2f dB | SSIM = %.4f over %d views -> %s",
             step, metrics["psnr"], metrics["ssim"], metrics["n_test"], out_dir)


# Training driver

@dataclass
class OptConfig:
    pos_init:      float = 0.00016
    pos_final:     float = 0.0000016
    pos_max_steps: int   = 30_000
    feature_lr:    float = 0.0025
    scaling_lr:    float = 0.005
    rotation_lr:   float = 0.001
    opacity_lr:    float = 0.05


@dataclass
class DensifyConfig:
    from_iter:              int   = 500
    until_iter:             int   = 15_000
    interval:               int   = 100
    grad_threshold:         float = 0.0002
    percent_dense:          float = 0.01
    min_opacity:            float = 0.005
    opacity_reset_interval: int   = 3_000
    # gsplat ellipse radii are large; INRIA's 20 px cutoff would delete almost all splats.
    max_screen_size:        float = 400.0


@dataclass
class TrainConfig:
    source_path:      str
    model_path:       str  = "./output"
    resolution:       int  = 1
    llffhold:         int  = 8
    sh_degree:        int  = 3
    device:           str  = "cuda:2"
    log_level:        str  = "INFO"
    white_background: bool = False
    iterations:       int  = 30_000
    lambda_dssim:     float = 0.2
    save_iterations:  tuple[int, ...] = (7_000, 30_000)
    opt:     OptConfig     = field(default_factory=OptConfig)
    densify: DensifyConfig = field(default_factory=DensifyConfig)


def train(cfg: TrainConfig):
    # Grouped conv2d (SSIM) can hit CUDNN_STATUS_NOT_INITIALIZED on some CUDA/cuDNN stacks.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(cfg.device)
    save_iters = sorted(set(cfg.save_iterations) | {cfg.iterations})

    log.info("Loading scene ...")
    cameras, xyz, rgb = load_scene(cfg.source_path, cfg.resolution, device=device)
    ext = scene_extent(cameras)
    log.info("  %d cameras | %s points | extent = %.3f", len(cameras), f"{len(xyz):,}", ext)

    # INRIA convention: idx % llffhold == 0 -> test, rest -> train (LLFF hold-out).
    train_cams = [c for i, c in enumerate(cameras) if i % cfg.llffhold != 0]
    test_cams  = [c for i, c in enumerate(cameras) if i % cfg.llffhold == 0]
    log.info("  %d train / %d test cameras", len(train_cams), len(test_cams))

    gs = Gaussians(cfg.sh_degree, device=device)
    gs.from_pcd(xyz, rgb)
    gs.setup_opt(cfg.opt, spatial_scale=ext)

    bg = (torch.ones if cfg.white_background else torch.zeros)(3, device=device)
    Path(cfg.model_path).mkdir(parents=True, exist_ok=True)

    ema, vp_stack = 0.0, []
    pbar = trange(1, cfg.iterations + 1, desc="Training")

    for step in pbar:
        gs.update_xyz_lr(step, cfg.opt, ext)
        if step % 1000 == 0:
            gs.active_sh = min(gs.active_sh + 1, gs.max_sh)

        if not vp_stack: vp_stack = train_cams.copy()
        cam = vp_stack.pop(random.randint(0, len(vp_stack) - 1))

        img, info = render(cam, gs, bg)
        info["means2d"].retain_grad()

        gt   = cam.image
        loss = ((1 - cfg.lambda_dssim) * F.l1_loss(img, gt)
                + cfg.lambda_dssim * (1 - ssim(img.unsqueeze(0), gt.unsqueeze(0))))
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            pbar.set_postfix(loss=f"{ema:.4f}", N=f"{gs.N:,}")

            if step in save_iters:
                ply = Path(cfg.model_path) / f"point_cloud/iteration_{step}/point_cloud.ply"
                gs.save_ply(str(ply))
                log.info("  [iter %d] saved %s Gaussians -> %s", step, f"{gs.N:,}", ply)
                save_test_renders(test_cams, gs, bg, cfg.model_path, step)

            if step < cfg.densify.until_iter:
                gs.accumulate_stats(info, cam.W, cam.H)
                if step > cfg.densify.from_iter and step % cfg.densify.interval == 0:
                    gs.densify_and_prune(ext, cfg.densify, step)
                if step % cfg.densify.opacity_reset_interval == 0:
                    gs.reset_opacity()

        gs.opt.step()
        gs.opt.zero_grad(set_to_none=True)

    log.info("Done -- %s Gaussians (final at iter %d).", f"{gs.N:,}", cfg.iterations)


if __name__ == "__main__":
    cfg = TrainConfig(source_path="data/mipnerf360/bicycle")
    logging.basicConfig(level=getattr(logging, cfg.log_level), format="%(message)s")
    train(cfg)
