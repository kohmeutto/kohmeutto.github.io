import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =========================================================
# [설정 구역]
# =========================================================
Nx, Ny = 31, 31
MAX_ITER = 60
DISTORTION = 0.8
K_WAVE = 4
EPS = 1e-8
# =========================================================

class D2LeibnizVisualMaster:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        x_uni = np.linspace(0, 1, nx)
        self.ref_residue = self.calc_leibniz_raw(x_uni)

    def calc_leibniz_raw(self, x):
        n = len(x); h = np.diff(x)
        D = np.zeros((n, n))
        for i in range(n-1): D[i, i] = -1.0/h[i]; D[i, i+1] = 1.0/h[i]
        RD = D - D.T
        M = D @ RD + RD @ D
        return M[1:-1, 1:-1].diagonal()

    def get_healing_points(self, x):
        return self.calc_leibniz_raw(x) - self.ref_residue

    def get_global_residue(self, x_vec, y_vec):
        rx = self.get_healing_points(x_vec)
        ry = self.get_healing_points(y_vec)
        return np.concatenate([rx, ry])

    def solve_pde(self, x, y, k):
        def build_A(c):
            n = len(c); h = np.diff(c); hL, hR = h[:-1], h[1:]
            l, m, u = np.zeros(n-1), np.zeros(n), np.zeros(n-1)
            m[1:-1] = -2.0/(hL*hR); l[0:-1] = 2.0/(hL*(hL+hR)); u[1:] = 2.0/(hR*(hL+hR))
            mat = sp.diags([l, m, u], [-1, 0, 1], shape=(n, n)).tolil()
            mat[0, 0] = mat[-1, -1] = 1.0; return mat.tocsr()
        Sx, Sy = build_A(x), build_A(y)
        S_2d = sp.kron(Sx, sp.eye(len(y))) + sp.kron(sp.eye(len(x)), Sy)
        grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
        u_ex = np.sin(k*np.pi*grid_x) * np.sin(k*np.pi*grid_y)
        rhs = (-2.0 * (k*np.pi)**2 * u_ex).flatten()
        S_bc = S_2d.tolil(); mask = np.zeros((len(x), len(y)), dtype=bool)
        mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=True
        for idx in np.where(mask.flatten())[0]: S_bc[idx,:]=0; S_bc[idx,idx]=1.0; rhs[idx]=u_ex.flatten()[idx]
        return spla.spsolve(S_bc.tocsr(), rhs).reshape((len(x), len(y)))

engine = D2LeibnizVisualMaster(Nx, Ny)
x_curr = np.linspace(0, 1, Nx); y_curr = np.linspace(0, 1, Ny)
dx = 1.0/(Nx-1)
for i in [10, 20]: x_curr[i] += dx * DISTORTION; y_curr[i] -= dx * DISTORTION
x_init, y_init = x_curr.copy(), y_curr.copy()

history = []
print("\n▶ [Leibniz Master Report] Optimization Starting...")

for f in range(MAX_ITER):
    r_vec = engine.get_global_residue(x_curr, y_curr)
    max_err = np.max(np.abs(r_vec))
    history.append(max_err)
    if max_err < 1e-10: break
    
    n_vars = (Nx-2) + (Ny-2)
    jac = np.zeros((len(r_vec), n_vars))
    for j in range(Nx-2):
        xe = x_curr.copy(); xe[j+1] += EPS
        jac[:, j] = (engine.get_global_residue(xe, y_curr) - r_vec) / EPS
    for j in range(Ny-2):
        ye = y_curr.copy(); ye[j+1] += EPS
        jac[:, Nx-2 + j] = (engine.get_global_residue(x_curr, ye) - r_vec) / EPS
    
    du, _, _, _ = np.linalg.lstsq(jac, -r_vec, rcond=None)
    x_curr[1:-1] += du[:Nx-2] * 0.8; y_curr[1:-1] += du[Nx-2:] * 0.8

# 결과 데이터 생성
u_ex_grid = np.sin(K_WAVE * np.pi * np.meshgrid(x_init, y_init, indexing='ij')[0]) * \
            np.sin(K_WAVE * np.pi * np.meshgrid(x_init, y_init, indexing='ij')[1])
e_before = np.abs(engine.solve_pde(x_init, y_init, K_WAVE) - u_ex_grid)
u_ex_final = np.sin(K_WAVE * np.pi * np.meshgrid(x_curr, y_curr, indexing='ij')[0]) * \
             np.sin(K_WAVE * np.pi * np.meshgrid(x_curr, y_curr, indexing='ij')[1])
e_after = np.abs(engine.solve_pde(x_curr, y_curr, K_WAVE) - u_ex_final)

# =========================================================
# 🎯 [Final Integrated Report] 2x3 Layout
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# 1. 수렴 곡선
axes[0, 0].semilogy(history, 'r-o', markersize=3)
axes[0, 0].set_title("1. Leibniz Point Convergence"); axes[0, 0].grid(True, alpha=0.3)

# 2. 초기 메쉬 형상
for x in x_init: axes[0, 1].axvline(x, color='red', alpha=0.2, lw=0.5)
for y in y_init: axes[0, 1].axhline(y, color='red', alpha=0.2, lw=0.5)
axes[0, 1].set_title("2. Initial Distorted Mesh"); axes[0, 1].set_aspect('equal')

# 3. 최종 메쉬 형상
for x in x_curr: axes[0, 2].axvline(x, color='green', alpha=0.3, lw=0.5)
for y in y_curr: axes[0, 2].axhline(y, color='green', alpha=0.3, lw=0.5)
axes[0, 2].set_title("3. Healed Final Mesh"); axes[0, 2].set_aspect('equal')

# 4. Leibniz 포인트 분포 (Before/After)
init_points = engine.get_global_residue(x_init, y_init)
final_points = engine.get_global_residue(x_curr, y_curr)
axes[1, 0].plot(init_points, 'r.', label='Initial Points')
axes[1, 0].plot(final_points, 'g-', label='Final Points')
axes[1, 0].set_title("4. Leibniz Point Analysis"); axes[1, 0].legend()

# 5. 초기 에러 맵
Xi, Yi = np.meshgrid(x_init, y_init, indexing='ij')
im1 = axes[1, 1].pcolormesh(Xi, Yi, e_before, cmap='magma', shading='auto')
axes[1, 1].set_title(f"5. Error Before (Max: {np.max(e_before):.4f})"); fig.colorbar(im1, ax=axes[1, 1])

# 6. 최종 에러 맵
Xf, Yf = np.meshgrid(x_curr, y_curr, indexing='ij')
im2 = axes[1, 2].pcolormesh(Xf, Yf, e_after, cmap='magma', shading='auto', vmin=0, vmax=np.max(e_before))
axes[1, 2].set_title(f"6. Error After (Max: {np.max(e_after):.4f})"); fig.colorbar(im2, ax=axes[1, 2])

plt.tight_layout(); plt.savefig('Leibniz_Master_Full_Report.png')
print("\n▶ Integrated Report saved as 'Leibniz_Master_Full_Report.png'.")