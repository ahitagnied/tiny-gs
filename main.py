# train.py — 3D Gaussian Splatting with gsplat backend
# 1:1 logic parity with graphdeco-inria/gaussian-splatting

import argparse
import json
import math, struct, random
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

# ─── SH ────────────────────────────────────────────────────────────────────────
C0 = 0.28209479177387814
def rgb2sh(rgb): return (rgb - 0.5) / C0

# ─── COLMAP binary readers ─────────────────────────────────────────────────────
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

def read_points_bin(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(_rd(f, "<Q")[0]):
            f.read(8); xyz.append(_rd(f, "<3d"))
            rgb.append([v / 255.0 for v in _rd(f, "<3B")])
            f.read(8 + _rd(f, "<Q")[0] * 8)
    return np.float32(xyz), np.float32(rgb)

def qvec2mat(q):
    w, x, y, z = q
    return np.float32([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                       [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                       [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def load_scene(data_dir, resolution=1):
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
        if resolution != 1:
            nw, nh = round(w / resolution), round(h / resolution)
            img = img.resize((nw, nh), Image.LANCZOS)
            fx /= resolution; fy /= resolution; cx /= resolution; cy /= resolution
            w, h = nw, nh
        cameras.append({"R": torch.tensor(qvec2mat(q)).cuda(),
                        "T": torch.tensor(t, dtype=torch.float32).cuda(),
                        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                        "W": int(w), "H": int(h),
                        "image": TF.to_tensor(img).cuda(), "name": name})
    return cameras, xyz, rgb

def _resolve_image_path(base: Path, file_path: str) -> Path:
    rel = file_path[2:] if file_path.startswith("./") else file_path
    img_path = base / rel
    if img_path.is_file():
        return img_path
    if not img_path.suffix:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            cand = img_path.with_suffix(ext)
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"Missing image for frame: {file_path} (tried {img_path})")

def load_scene_nerf(data_dir, transforms_file="transforms_train.json", resolution=1, num_points=100_000):
    """Blender / NeRF-style JSON: camera_angle_x, frames[].transform_matrix, file_path."""
    base = Path(data_dir)
    meta_path = base / transforms_file
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    fovx = float(meta["camera_angle_x"])
    frames = sorted(meta["frames"], key=lambda fr: str(fr["file_path"]))
    cameras, origins = [], []

    for fr in frames:
        img_path = _resolve_image_path(base, fr["file_path"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if resolution != 1:
            nw, nh = max(1, round(w / resolution)), max(1, round(h / resolution))
            img = img.resize((nw, nh), Image.LANCZOS)
            w, h = nw, nh
        fx = 0.5 * w / math.tan(0.5 * fovx)
        if "camera_angle_y" in meta:
            fovy = float(meta["camera_angle_y"])
            fy = 0.5 * h / math.tan(0.5 * fovy)
        else:
            fovy = 2.0 * math.atan(math.tan(0.5 * fovx) * h / w)
            fy = 0.5 * h / math.tan(0.5 * fovy)
        cx, cy = 0.5 * w, 0.5 * h

        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        # NeRF/Blender OpenGL vs COLMAP-style world used by gsplat viewmats
        c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
        w2c = np.linalg.inv(c2w)
        R, T = w2c[:3, :3].astype(np.float32), w2c[:3, 3].astype(np.float32)
        origins.append(c2w[:3, 3].astype(np.float64))

        cameras.append({"R": torch.tensor(R).cuda(),
                        "T": torch.tensor(T, dtype=torch.float32).cuda(),
                        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                        "W": int(w), "H": int(h),
                        "image": TF.to_tensor(img).cuda(), "name": img_path.name})

    origins = np.stack(origins, axis=0)
    lo, hi = origins.min(0), origins.max(0)
    diag = float(np.linalg.norm(hi - lo)) + 1e-6
    margin = 0.25 * diag
    lo, hi = lo - margin, hi + margin
    rng = np.random.default_rng(0)
    xyz = rng.uniform(lo, hi, size=(num_points, 3)).astype(np.float32)
    rgb = np.full((num_points, 3), 0.5, dtype=np.float32)
    return cameras, xyz, rgb

def scene_extent(cameras):
    centers = np.array([(-c["R"].cpu().numpy().T @ c["T"].cpu().numpy()) for c in cameras])
    return float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)

# ─── Gaussian Model ────────────────────────────────────────────────────────────
class Gaussians:
    def __init__(self, sh_degree=3):
        self.max_sh, self.active_sh = sh_degree, 0
        self.xyz = self.f_dc = self.f_rest = self.scale = self.rot = self.opacity = None
        self.grad_accum = self.denom = self.max_r2d = self.opt = None

    @property
    def N(self): return self.xyz.shape[0]

    def from_pcd(self, xyz_np, rgb_np):
        N    = len(xyz_np)
        sq_d = np.mean(KDTree(xyz_np).query(xyz_np, k=4)[0][:, 1:] ** 2, axis=1).clip(1e-7)
        self.xyz     = nn.Parameter(torch.tensor(xyz_np, dtype=torch.float32).cuda())
        self.f_dc    = nn.Parameter(rgb2sh(torch.tensor(rgb_np, dtype=torch.float32).cuda()).unsqueeze(1).contiguous())
        self.f_rest  = nn.Parameter(torch.zeros(N, (self.max_sh + 1) ** 2 - 1, 3, device="cuda"))
        self.scale   = nn.Parameter(torch.tensor(np.log(np.sqrt(sq_d)), dtype=torch.float32)
                                    .cuda().unsqueeze(1).expand(-1, 3).contiguous())
        self.rot     = nn.Parameter(torch.cat([torch.ones(N, 1, device="cuda"),
                                               torch.zeros(N, 3, device="cuda")], dim=1))
        self.opacity = nn.Parameter(torch.logit(torch.full((N, 1), 0.1, device="cuda")))
        self.grad_accum = torch.zeros(N, 1, device="cuda")
        self.denom      = torch.zeros(N, 1, device="cuda")
        self.max_r2d    = torch.zeros(N,    device="cuda")

    def setup_opt(self, args, spatial_scale):
        self.opt = torch.optim.Adam([
            {"params": [self.xyz],     "lr": args.pos_init * spatial_scale, "name": "xyz"},
            {"params": [self.f_dc],    "lr": args.feature_lr,               "name": "f_dc"},
            {"params": [self.f_rest],  "lr": args.feature_lr / 20,          "name": "f_rest"},
            {"params": [self.scale],   "lr": args.scaling_lr,               "name": "scaling"},
            {"params": [self.rot],     "lr": args.rotation_lr,              "name": "rotation"},
            {"params": [self.opacity], "lr": args.opacity_lr,               "name": "opacity"},
        ], eps=1e-15)

    def update_xyz_lr(self, step, args, spatial_scale):
        t  = min(step / args.pos_max_steps, 1.0)
        lr = math.exp(math.log(args.pos_init * spatial_scale) * (1 - t) +
                      math.log(args.pos_final * spatial_scale) * t)
        for g in self.opt.param_groups:
            if g["name"] == "xyz": g["lr"] = lr

    # ── Optimizer-aware tensor ops ────────────────────────────────────────────
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
        self.grad_accum = torch.cat([self.grad_accum, torch.zeros(n, 1, device="cuda")])
        self.denom      = torch.cat([self.denom,      torch.zeros(n, 1, device="cuda")])
        self.max_r2d    = torch.cat([self.max_r2d,    torch.zeros(n,    device="cuda")])

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

    # ── Densification ─────────────────────────────────────────────────────────
    def accumulate_stats(self, info):
        radii = info["radii"].squeeze(0).float()
        # gsplat: [N, 2] ellipse axes; older / other paths may use [N]
        if radii.dim() == 1:
            vis = radii > 0
            r2d = radii
        else:
            vis = (radii > 0).all(dim=-1)
            r2d = radii.max(dim=-1).values
        self.max_r2d[vis] = torch.maximum(self.max_r2d[vis], r2d[vis])
        g2d = info["means2d"].grad.squeeze(0)
        self.grad_accum[vis] += g2d[vis].norm(dim=-1, keepdim=True)
        self.denom[vis]      += 1

    @staticmethod
    def _build_rotation(r):
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
        padded = torch.zeros(self.N, device="cuda"); padded[:len(grads)] = grads
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
        self.prune(~torch.cat([mask, torch.zeros(K * mask.sum(), dtype=torch.bool, device="cuda")]))

    def densify_and_prune(self, ext, args, step):
        grads = (self.grad_accum / self.denom).squeeze()
        grads[grads.isnan() | grads.isinf()] = 0.0
        self._clone(grads, args.densify_grad_threshold, args.percent_dense, ext)
        self._split(grads, args.densify_grad_threshold, args.percent_dense, ext)

        prune = (torch.sigmoid(self.opacity) < args.min_opacity).squeeze()
        if step > args.opacity_reset_interval:
            # INRIA 3DGS used ~20 px; gsplat ellipse radii are usually much larger — same cutoff deletes almost all splats.
            prune |= self.max_r2d > args.max_screen_size
            prune |= torch.exp(self.scale).max(dim=1).values > 0.1 * ext
        if (~prune).sum() == 0:
            prune = torch.zeros_like(prune)
        self.prune(~prune)
        torch.cuda.empty_cache()

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

# ─── Render ────────────────────────────────────────────────────────────────────
def render(cam, gs, bg):
    vmat = torch.eye(4, device="cuda")
    vmat[:3, :3] = cam["R"]; vmat[:3, 3] = cam["T"]
    K = torch.tensor([[cam["fx"], 0, cam["cx"]],
                      [0, cam["fy"], cam["cy"]],
                      [0, 0, 1]], device="cuda")
    renders, _, info = rasterization(
        means=gs.xyz, quats=F.normalize(gs.rot, dim=-1),
        scales=torch.exp(gs.scale), opacities=torch.sigmoid(gs.opacity).squeeze(-1),
        colors=torch.cat([gs.f_dc, gs.f_rest], dim=1),
        viewmats=vmat.unsqueeze(0), Ks=K.unsqueeze(0).float(),
        width=cam["W"], height=cam["H"], sh_degree=gs.active_sh,
        near_plane=0.01, backgrounds=bg.unsqueeze(0), packed=False)
    return renders[0].permute(2, 0, 1), info

# ─── SSIM ──────────────────────────────────────────────────────────────────────
_WIN = None
def _ssim_window():
    global _WIN
    if _WIN is None:
        x = torch.arange(11, dtype=torch.float32) - 5
        g = torch.exp(-x ** 2 / 4.5); g /= g.sum()
        _WIN = g.outer(g).unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1).cuda()
    return _WIN

def ssim(a, b):
    w  = _ssim_window(); C1, C2 = 0.01 ** 2, 0.03 ** 2
    ma = F.conv2d(a,   w, padding=5, groups=3)
    mb = F.conv2d(b,   w, padding=5, groups=3)
    sa = F.conv2d(a*a, w, padding=5, groups=3) - ma ** 2
    sb = F.conv2d(b*b, w, padding=5, groups=3) - mb ** 2
    ab = F.conv2d(a*b, w, padding=5, groups=3) - ma * mb
    return ((2*ma*mb + C1) * (2*ab + C2) / ((ma**2 + mb**2 + C1) * (sa + sb + C2))).mean()

# ─── Test-set rendering ───────────────────────────────────────────────────────
@torch.no_grad()
def save_test_renders(test_cams, gs, bg, model_path, step):
    if not test_cams:
        return
    out_dir = Path(model_path) / f"test/iteration_{step}"
    (out_dir / "renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    psnrs = []
    for i, cam in enumerate(test_cams):
        img, _ = render(cam, gs, bg)
        img = img.clamp(0, 1)
        gt  = cam["image"].clamp(0, 1)
        mse = F.mse_loss(img, gt).item()
        psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))
        stem = Path(cam["name"]).stem or f"{i:05d}"
        TF.to_pil_image(img.cpu()).save(out_dir / "renders" / f"{stem}.png")
        TF.to_pil_image(gt.cpu()).save(out_dir / "gt"      / f"{stem}.png")
    psnr = sum(psnrs) / len(psnrs)
    print(f"  [iter {step}] test PSNR = {psnr:.2f} dB over {len(psnrs)} views → {out_dir}")

# ─── Training ──────────────────────────────────────────────────────────────────
def train(args):
    # Grouped conv2d (SSIM) can hit CUDNN_STATUS_NOT_INITIALIZED on some CUDA/cuDNN stacks.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print("Loading scene …")
    if args.scene == "nerf":
        cameras, xyz, rgb = load_scene_nerf(
            args.source_path,
            transforms_file=args.nerf_transforms,
            resolution=args.resolution,
            num_points=args.nerf_num_points,
        )
    elif args.scene == "colmap":
        cameras, xyz, rgb = load_scene(args.source_path, args.resolution)
    else:
        nerf_json = Path(args.source_path) / args.nerf_transforms
        if nerf_json.is_file():
            cameras, xyz, rgb = load_scene_nerf(
                args.source_path,
                transforms_file=args.nerf_transforms,
                resolution=args.resolution,
                num_points=args.nerf_num_points,
            )
        else:
            cameras, xyz, rgb = load_scene(args.source_path, args.resolution)
    ext = scene_extent(cameras)
    print(f"  {len(cameras)} cameras | {len(xyz):,} points | extent = {ext:.3f}")

    train_cams = [c for i, c in enumerate(cameras) if i % args.llffhold != 0]
    test_cams  = [c for i, c in enumerate(cameras) if i % args.llffhold == 0]
    print(f"  {len(train_cams)} train / {len(test_cams)} test cameras")

    gs = Gaussians(args.sh_degree)
    gs.from_pcd(xyz, rgb)
    gs.setup_opt(args, spatial_scale=ext)

    bg = (torch.ones if args.white_background else torch.zeros)(3, device="cuda")
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    ema, vp_stack = 0.0, []
    pbar = trange(1, args.iterations + 1, desc="Training")

    for step in pbar:
        gs.update_xyz_lr(step, args, ext)
        if step % 1000 == 0:
            gs.active_sh = min(gs.active_sh + 1, gs.max_sh)

        if not vp_stack: vp_stack = train_cams.copy()
        cam = vp_stack.pop(random.randint(0, len(vp_stack) - 1))

        img, info = render(cam, gs, bg)
        info["means2d"].retain_grad()

        gt   = cam["image"]
        loss = ((1 - args.lambda_dssim) * F.l1_loss(img, gt)
                + args.lambda_dssim * (1 - ssim(img.unsqueeze(0), gt.unsqueeze(0))))
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            pbar.set_postfix(loss=f"{ema:.4f}", N=f"{gs.N:,}")

            if step in args.save_iterations:
                ply = Path(args.model_path) / f"point_cloud/iteration_{step}/point_cloud.ply"
                gs.save_ply(str(ply))
                print(f"\n  [iter {step}] saved {gs.N:,} Gaussians → {ply}")
                save_test_renders(test_cams, gs, bg, args.model_path, step)

            if step < args.densify_until_iter:
                gs.accumulate_stats(info)
                if step > args.densify_from_iter and step % args.densification_interval == 0:
                    gs.densify_and_prune(ext, args, step)
                if (step % args.opacity_reset_interval == 0
                        or (args.white_background and step == args.densify_from_iter)):
                    gs.reset_opacity()

        gs.opt.step()
        gs.opt.zero_grad(set_to_none=True)

    ply = Path(args.model_path) / f"point_cloud/iteration_{args.iterations}/point_cloud.ply"
    gs.save_ply(str(ply))
    print(f"\nDone — {gs.N:,} Gaussians saved to {ply}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train 3D Gaussians (COLMAP or NeRF JSON + gsplat).")
    p.add_argument(
        "source_path",
        type=str,
        help="COLMAP root (sparse/0/*.bin + images/) or NeRF folder (transforms_train.json + images)",
    )
    p.add_argument(
        "--scene",
        choices=("auto", "colmap", "nerf"),
        default="auto",
        help="auto: use NeRF JSON if --nerf_transforms exists under source_path, else COLMAP",
    )
    p.add_argument("--nerf_transforms", type=str, default="transforms_train.json")
    p.add_argument("--nerf_num_points", type=int, default=100_000)
    p.add_argument("--model_path", type=str, default="./output")
    p.add_argument("--resolution", type=int, default=1)
    p.add_argument("--llffhold", type=int, default=8)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--white_background", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--lambda_dssim", type=float, default=0.2)
    p.add_argument("--pos_init", type=float, default=0.00016)
    p.add_argument("--pos_final", type=float, default=0.0000016)
    p.add_argument("--pos_max_steps", type=int, default=30_000)
    p.add_argument("--feature_lr", type=float, default=0.0025)
    p.add_argument("--scaling_lr", type=float, default=0.005)
    p.add_argument("--rotation_lr", type=float, default=0.001)
    p.add_argument("--opacity_lr", type=float, default=0.05)
    p.add_argument("--densify_from_iter", type=int, default=500)
    p.add_argument("--densify_until_iter", type=int, default=15_000)
    p.add_argument("--densification_interval", type=int, default=100)
    p.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    p.add_argument("--percent_dense", type=float, default=0.01)
    p.add_argument("--min_opacity", type=float, default=0.005)
    p.add_argument("--opacity_reset_interval", type=int, default=3000)
    p.add_argument(
        "--max_screen_size",
        type=float,
        default=400.0,
        help="Prune Gaussians whose max 2D radius (px) exceeds this after opacity_reset_interval (gsplat-scale; not INRIA 20px).",
    )
    p.add_argument(
        "--save_iterations",
        type=int,
        nargs="*",
        default=[7_000, 30_000],
    )
    train(p.parse_args())