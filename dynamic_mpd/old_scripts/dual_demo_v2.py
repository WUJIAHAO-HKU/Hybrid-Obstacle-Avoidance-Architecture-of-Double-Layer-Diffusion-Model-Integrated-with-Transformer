"""双Diffusion v2"""
import sys, os
MPD_ROOT = '/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public'
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import ComplexObstacleDataGenerator, ObstacleMotionConfig

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def to_torch(x, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device, dtype=torch.float32)


class UpperModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.config = None
        
    def load(self, path='/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth'):
        print(f"[Upper] Loading: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.config = ckpt['config']
        self.model = TrainableObstacleDiffusion(self.config, self.device)
        self.model.denoise_net.load_state_dict(ckpt['model_state_dict'])
        self.model.denoise_net.eval()
        for p in self.model.denoise_net.parameters():
            p.requires_grad = False
        print(f"  Epochs: {ckpt['num_epochs']}, Best: {ckpt['best_test_loss']:.4f}")
        return self
    
    def predict(self, h, n=15):
        with torch.no_grad():
            return self.model.predict(h, n)


class LowerModel:
    def __init__(self, device='cpu'):
        self.device = device
        
    def load(self):
        print("[Lower] Using sampling-based planner")
        return self
    
    def plan(self, start, goal, obstacles=None, n=25):
        start = to_torch(start, self.device)
        goal = to_torch(goal, self.device)
        
        H = 16
        trajs = []
        for _ in range(n):
            t = torch.linspace(0, 1, H, device=self.device)
            base = start + t.unsqueeze(1) * (goal - start)
            noise = torch.randn_like(base) * 0.15
            noise[0] = 0
            noise[-1] = 0
            for _ in range(3):
                noise[1:-1] = 0.25*noise[:-2] + 0.5*noise[1:-1] + 0.25*noise[2:]
            traj = base + noise
            traj[0] = start
            traj[-1] = goal
            trajs.append(traj)
        trajs = torch.stack(trajs)
        
        best = None
        best_c = float('inf')
        for traj in trajs:
            c = 0
            if obstacles:
                for obs in obstacles:
                    obs = to_torch(obs, self.device)
                    for pt in traj:
                        d = torch.norm(pt - obs).item()
                        if d < 0.1:
                            c += (0.1 - d) * 30
            for i in range(1, len(traj)):
                c += torch.norm(traj[i]-traj[i-1]).item() * 0.05
            if c < best_c:
                best_c = c
                best = traj
        return trajs, best


def gen_obs(n=3, steps=100):
    cfg = ObstacleMotionConfig(n, 8, 12, 0.1, 0.02, 0.6)
    gen = ComplexObstacleDataGenerator(cfg)
    types = ['arc', 'linear', 'zigzag']
    ts = [gen.generate_single_obstacle_trajectory(types[i%3], steps+30) for i in range(n)]
    return torch.stack(ts, dim=1)


def main():
    print("="*60)
    print("  Dual Diffusion Demo")
    print("="*60)
    
    device = 'cpu'
    upper = UpperModel(device).load()
    lower = LowerModel(device).load()
    
    n_obs = 3
    max_steps = 100
    obs_trajs = gen_obs(n_obs, max_steps)
    
    start = torch.tensor([-0.5, -0.5], device=device)
    goal = torch.tensor([0.5, 0.5], device=device)
    print(f"Start: {start.tolist()}, Goal: {goal.tolist()}")
    
    pos = start.clone()
    hist = [pos.clone()]
    obs_buf = []
    pred_h = []
    traj_h = []
    
    obs_len = upper.config.obs_history_len
    pred_len = upper.config.pred_horizon
    
    for step in range(max_steps):
        obs_buf.append(obs_trajs[step].clone())
        if len(obs_buf) > obs_len:
            obs_buf.pop(0)
        
        if len(obs_buf) < 4:
            hist.append(pos.clone())
            pred_h.append(None)
            traj_h.append(None)
            continue
        
        preds = []
        for i in range(n_obs):
            h = torch.stack([obs_buf[t][i] for t in range(len(obs_buf))])
            if len(h) < obs_len:
                pad = h[0:1].repeat(obs_len - len(h), 1)
                h = torch.cat([pad, h], dim=0)
            m, _, _ = upper.predict(h, 12)
            preds.append(m)
        preds = torch.stack(preds, dim=1)
        pred_h.append(preds.clone())
        
        future_obs = []
        for ti in [0, 4, 8, 11]:
            for i in range(n_obs):
                future_obs.append(preds[ti, i])
        
        _, best = lower.plan(pos, goal, future_obs, 20)
        traj_h.append(best.clone() if best is not None else None)
        
        if best is not None and len(best) >= 2:
            ctrl = best[1] - pos
            spd = torch.norm(ctrl)
            if spd > 0.025:
                ctrl = ctrl * 0.025 / spd
            pos = pos + ctrl
        
        hist.append(pos.clone())
        
        dist = torch.norm(pos - goal).item()
        if dist < 0.05:
            print(f"  [Step {step}] GOAL!")
            break
        if step % 20 == 0:
            print(f"  [Step {step}] dist={dist:.3f}")
    
    total = len(hist)
    final = torch.norm(hist[-1] - goal).item()
    print(f"Done: {total} steps, final={final:.3f}")
    
    # 可视化
    colors = ['#e74c3c', '#3498db', '#27ae60']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    
    ax = axes[0]
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.set_title('Trajectories', fontweight='bold')
    ax.grid(True, alpha=0.3)
    for i in range(n_obs):
        ot = obs_trajs[:total, i].numpy()
        ax.plot(ot[:,0], ot[:,1], '--', color=colors[i], alpha=0.5)
    rt = torch.stack(hist).numpy()
    ax.plot(rt[:,0], rt[:,1], 'purple', linewidth=2)
    ax.scatter(*start, c='green', s=120, marker='s', zorder=10)
    ax.scatter(*goal, c='gold', s=180, marker='*', zorder=10)
    
    ax2 = axes[1]
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_aspect('equal')
    ax2.set_title('Prediction (Upper)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    f = min(40, total-1)
    while f > 0 and pred_h[f] is None:
        f -= 1
    if pred_h[f] is not None:
        p = pred_h[f].numpy()
        for i in range(n_obs):
            end = min(f+12, obs_trajs.shape[0])
            real = obs_trajs[f:end, i].numpy()
            ax2.plot(real[:,0], real[:,1], '--', color=colors[i], alpha=0.5)
            ax2.plot(p[:,i,0], p[:,i,1], '-', color=colors[i], linewidth=2)
            ax2.scatter(obs_trajs[f,i,0], obs_trajs[f,i,1], c=colors[i], s=80)
    
    ax3 = axes[2]
    ax3.set_xlim(-0.7, 0.7)
    ax3.set_ylim(-0.7, 0.7)
    ax3.set_aspect('equal')
    ax3.set_title('Planning (Lower)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    if traj_h[f] is not None:
        plan = traj_h[f].numpy()
        ax3.plot(plan[:,0], plan[:,1], 'purple', linewidth=2.5)
        for i in range(n_obs):
            c = Circle((obs_trajs[f,i,0], obs_trajs[f,i,1]), 0.06, color=colors[i], alpha=0.4)
            ax3.add_patch(c)
    ax3.scatter(hist[f][0], hist[f][1], c='purple', s=120, marker='s', zorder=10)
    ax3.scatter(*goal, c='gold', s=180, marker='*', zorder=10)
    
    fig.suptitle('Dual Diffusion: Upper (Prediction) + Lower (Planning)', fontsize=13, fontweight='bold')
    out = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_v2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")
    
    # 动画
    fig, ax = plt.subplots(figsize=(8, 8))
    def update(f):
        ax.clear()
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Frame {f}', fontweight='bold')
        for i in range(n_obs):
            pos = obs_trajs[f, i]
            c = Circle((pos[0], pos[1]), 0.05, color=colors[i], alpha=0.6)
            ax.add_patch(c)
            if f < len(pred_h) and pred_h[f] is not None:
                p = pred_h[f][:, i].numpy()
                ax.plot(p[:,0], p[:,1], '-', color=colors[i], alpha=0.5, linewidth=2)
        if f > 0:
            rt = torch.stack(hist[:f+1]).numpy()
            ax.plot(rt[:,0], rt[:,1], 'purple', linewidth=2)
        ax.scatter(hist[f][0], hist[f][1], c='purple', s=150, marker='s', edgecolors='white', zorder=10)
        if f < len(traj_h) and traj_h[f] is not None:
            plan = traj_h[f].numpy()
            ax.plot(plan[:,0], plan[:,1], 'purple', linestyle='--', alpha=0.3)
        ax.scatter(*start, c='green', s=100, marker='s', zorder=5)
        ax.scatter(*goal, c='gold', s=150, marker='*', zorder=5)
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=total, interval=80)
    out2 = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_v2.gif'
    anim.save(out2, writer='pillow', fps=12)
    plt.close()
    print(f"Saved: {out2}")
    
    print("="*60)
    print(f"  RESULT: {'SUCCESS' if final < 0.1 else 'NEED MORE STEPS'}")
    print("="*60)


if __name__ == '__main__':
    main()
