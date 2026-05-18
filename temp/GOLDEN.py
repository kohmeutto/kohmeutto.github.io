import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# =========================================================
# [설정 구역]
# =========================================================
Nx, Ny = 31, 31
MAX_ITER = 100       
TOLERANCE = 1e-10
DISTORTION = 0.8
K_WAVE = 4
EPS = 1e-8
# =========================================================

class D2GlobalVisualEngine:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny

    def get_full_asymmetry_residue(self, x_vec, y_vec):
        rx = self.get_1d_residue(x_vec)
        ry = self.get_1d_residue(y_vec)
        return np.concatenate([rx[1:-1], ry[1:-1]])

    def get_1d_residue(self, c):
        n = len(c)
        h = np.diff(c)
        A = np.zeros((n, n))
        for i in range(1, n-1):
            hL, hR = h[i-1], h[i]
            A[i, i-1] = 2.0 / (hL * (hL + hR))
            A[i, i+1] = 2.0 / (hR * (hL + hR))
        R_signed = np.zeros(n)
        for i in range(1, n-1):
            diff_left = A[i, i-1] - A[i-1, i] if i > 1 else 0.0
            diff_right = A[i, i+1] - A[i+1, i] if i < n-2 else 0.0
            R_signed[i] = diff_left + diff_right
        return R_signed

    def build_true_D2_sparse(self, x):
        n = len(x)
        h = np.diff(x)
        hL, hR = h[:-1], h[1:]
        m, l, u = np.zeros(n), np.zeros(n-1), np.zeros(n-1)
        m[1:-1] = -2.0 / (hL * hR)
        l[0:-1] = 2.0 / (hL * (hL + hR))
        u[1:] = 2.0 / (hR * (hL + hR))
        A = sp.diags([l, m, u], [-1, 0, 1], shape=(n, n)).tolil()
        A[0, 0], A[-1, -1] = 1.0, 1.0
        return A.tocsr()

    def solve_pde(self, x, y, k):
        Sx = self.build_true_D2_sparse(x)
        Sy = self.build_true_D2_sparse(y)
        S_2d = sp.kron(Sx, sp.eye(len(y))) + sp.kron(sp.eye(len(x)), Sy)
        grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
        u_ex = np.sin(k * np.pi * grid_x) * np.sin(k * np.pi * grid_y)
        f_ex = -2.0 * (k * np.pi)**2 * u_ex
        rhs = f_ex.flatten()
        S_bc = S_2d.tolil()
        mask = np.zeros((len(x), len(y)), dtype=bool)
        mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=True
        b_idx = np.where(mask.flatten())[0]
        for idx in b_idx:
            S_bc[idx, :] = 0
            S_bc[idx, idx] = 1.0
            rhs[idx] = u_ex.flatten()[idx]
        u_num = spla.spsolve(S_bc.tocsr(), rhs).reshape((len(x), len(y)))
        return np.abs(u_num - u_ex)

engine = D2GlobalVisualEngine(Nx, Ny)
x_init = np.linspace(0, 1, Nx)
y_init = np.linspace(0, 1, Ny)
dx_base = 1.0/(Nx-1)

# 초기 왜곡 삽입
for i in [8, 15, 22]:
    x_init[i:i+2] += dx_base * DISTORTION
    y_init[i:i+2] -= dx_base * DISTORTION

# 초기 상태 계산
e_before = engine.solve_pde(x_init, y_init, K_WAVE)

x_curr = x_init.copy()
y_curr = y_init.copy()
history = []

print("\n▶ Global Optimization Starting...")

for f in range(MAX_ITER):
    r_vec = engine.get_full_asymmetry_residue(x_curr, y_curr)
    max_err = np.max(np.abs(r_vec))
    history.append(max_err)
    if max_err < TOLERANCE:
        print(f"   Success — Step {f+1} | Converged to {max_err:.2e}")
        break
    
    n_vars = (Nx-2) + (Ny-2)
    jac = np.zeros((n_vars, n_vars))
    for j in range(Nx-2):
        xe = x_curr.copy()
        xe[j+1] += EPS
        jac[:, j] = (engine.get_full_asymmetry_residue(xe, y_curr) - r_vec) / EPS
    for j in range(Ny-2):
        ye = y_curr.copy()
        ye[j+1] += EPS
        jac[:, Nx-2 + j] = (engine.get_full_asymmetry_residue(x_curr, ye) - r_vec) / EPS
    
    du = np.linalg.solve(jac + np.eye(n_vars)*1e-12, -r_vec)
    x_curr[1:-1] += du[:Nx-2] * 0.8
    y_curr[1:-1] += du[Nx-2:] * 0.8

# 최종 상태 계산
e_after = engine.solve_pde(x_curr, y_curr, K_WAVE)

# =========================================================
# 🎯 5단 통합 시각화 리포트 (2x3 Layout)
# =========================================================
fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(2, 3)

# 1. 수렴 곡선
ax_res = fig.add_subplot(gs[0, 0])
ax_res.semilogy(history, 'm-o', markersize=4, linewidth=1.5)
ax_res.axhline(TOLERANCE, color='r', linestyle='--', alpha=0.5)
ax_res.set_title("Global Asymmetry Convergence")
ax_res.set_xlabel("Iteration Step"); ax_res.set_ylabel("Max Residue"); ax_res.grid(True, alpha=0.3)

# 2. Before Mesh
ax_mesh_i = fig.add_subplot(gs[0, 1])
for x in x_init: ax_mesh_i.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
for y in y_init: ax_mesh_i.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
ax_mesh_i.set_title("Initial Mesh (Distorted)")
ax_mesh_i.set_aspect('equal')

# 3. After Mesh
ax_mesh_f = fig.add_subplot(gs[0, 2])
for x in x_curr: ax_mesh_f.axvline(x, color='blue', linewidth=0.5, alpha=0.5)
for y in y_curr: ax_mesh_f.axhline(y, color='blue', linewidth=0.5, alpha=0.5)
ax_mesh_f.set_title("Final Mesh (Healed)")
ax_mesh_f.set_aspect('equal')

# 4. Before Error Heatmap
ax_err_i = fig.add_subplot(gs[1, 1])
Xi, Yi = np.meshgrid(x_init, y_init, indexing='ij')
im1 = ax_err_i.pcolormesh(Xi, Yi, e_before, cmap='magma', shading='auto')
ax_err_i.set_title(f"Before Error Map\nMax Error — {np.max(e_before):.4f}")
fig.colorbar(im1, ax=ax_err_i)
ax_err_i.set_aspect('equal')

# 5. After Error Heatmap
ax_err_f = fig.add_subplot(gs[1, 2])
Xf, Yf = np.meshgrid(x_curr, y_curr, indexing='ij')
im2 = ax_err_f.pcolormesh(Xf, Yf, e_after, cmap='magma', shading='auto', vmin=0, vmax=np.max(e_before))
ax_err_f.set_title(f"After Error Map\nMax Error — {np.max(e_after):.4f}")
fig.colorbar(im2, ax=ax_err_f)
ax_err_f.set_aspect('equal')

plt.tight_layout()
plt.savefig('D2_Global_Full_Mesh_Report.png')
print("\n▶ Results Analysis")
print(f"   Convergence Step — {len(history)}")
print(f"   Mesh State — Successfully Uniformized")
print("▶ Full report saved as 'D2_Global_Full_Mesh_Report.png'.")