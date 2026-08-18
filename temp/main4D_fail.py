import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys

# =========================================================
# [설정 구역]
# =========================================================
Nx, Ny = 31, 31
TOTAL_FRAMES = 50
distortion_strength = 0.4 
k_wave = 2
# =========================================================

class D4RichDiagnosticEngine:
    def __init__(self, nx, ny, k):
        self.nx, self.ny = nx, ny
        self.k = k
        self.Ix, self.Iy = sp.eye(nx), sp.eye(ny)
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
        
        # D4 시스템을 위한 Exact Solution
        self.u_ex = np.sin(k * np.pi * grid_x)**2 * np.sin(k * np.pi * grid_y)**2
        
        # 경계 마스크 (D4는 2개 층의 경계가 필요)
        mask = np.zeros((nx, ny), dtype=bool)
        mask[0:2, :]=mask[-2:, :]=mask[:, 0:2]=mask[:, -2:]=True
        self.b_idx = np.where(mask.flatten())[0]

    def build_d4_ops(self, coords):
        n = len(coords); h = np.diff(coords)
        D = np.zeros((n, n))
        for i in range(n-1):
            inv_h = 1.0 / h[i]
            D[i, i] = -inv_h; D[i, i+1] = inv_h
        D[-1, -1] = 1.0/h[-1]; D[-1, -2] = -1.0/h[-1]
        
        S2 = np.zeros((n, n))
        for i in range(1, n-1):
            S2[i, i-1] = 1.0/h[i-1]
            S2[i, i] = -(1.0/h[i-1] + 1.0/h[i]); S2[i, i+1] = 1.0/h[i]
        
        S2_sp = sp.csr_matrix(S2)
        S4_sp = S2_sp @ S2_sp # S4 = S2 * S2
        return S2_sp, S4_sp

    def get_d4_diagnostic_residue(self, coords, use_filter=True):
        S2, S4 = self.build_d4_ops(coords)
        R2 = (S2 - S2.T).toarray()
        S2_mat = S2.toarray()
        
        # 교수님 식 확장: R(D4) = S2*R(S2) + R(S2)*S2.T
        T1 = np.abs(S2_mat @ R2).max(axis=1) # 전파되는 왜곡 (치료 대상)
        T2 = np.abs(R2 @ S2_mat.T).max(axis=1) # 주입되는 흉터 (보존 대상)
        
        weight = T1 / (T1 + T2 + 1e-12) if use_filter else 1.0
        
        S4_bc = S4.tolil()
        for i in [0, 1, -2, -1]: S4_bc[i, :]=0; S4_bc[i, i]=1.0
        R4_bc = S4_bc.tocsr() - S4_bc.tocsr().T
        res_pure = np.sqrt(np.array(R4_bc.power(2).sum(axis=1)).flatten())
        return res_pure * weight, res_pure

    def solve(self, x, y):
        _, S4x = self.build_d4_ops(x); _, S4y = self.build_d4_ops(y)
        S_4d = sp.kron(S4x, self.Iy) + sp.kron(self.Ix, S4y)
        u_vec = self.u_ex.flatten()
        rhs = S_4d @ u_vec # f = L*u
        S_bc = S_4d.tolil()
        for idx in self.b_idx: S_bc[idx, :] = 0; S_bc[idx, idx] = 1.0
        rhs[self.b_idx] = u_vec[self.b_idx]
        u_num = spla.spsolve(S_bc.tocsr(), rhs).reshape((self.nx, self.ny))
        R_f = S_bc.tocsr() - S_bc.tocsr().T
        res_map = np.sqrt(np.array(R_f.power(2).sum(axis=1)).flatten()).reshape((self.nx, self.ny))
        return u_num, np.abs(u_num - self.u_ex), res_map

# 초기화 및 실행
engine = D4RichDiagnosticEngine(Nx, Ny, k_wave)
x_init = np.linspace(0, 1, Nx); y_init = np.linspace(0, 1, Ny)
# 국소 왜곡 주입
for i in [15]: x_init[i:i+2] += (1.0/(Nx-1)) * distortion_strength

xA, yA = x_init.copy(), y_init.copy()
xB, yB = x_init.copy(), y_init.copy()
history = {'xA':[], 'yA':[], 'eA':[], 'rA':[], 'xB':[], 'yB':[], 'eB':[], 'rB':[]}
eps = 1e-8

print("\n▶ [D4 Rich Analysis] 엔진 가동 중...")
for f in range(TOTAL_FRAMES):
    # Method A: Plain
    uA, eA, rA = engine.solve(xA, yA)
    for c in [xA, yA]:
        _, rv = engine.get_d4_diagnostic_residue(c, False)
        jac = np.zeros((Nx, Nx))
        for j in range(2, Nx-2):
            tmp = c.copy(); tmp[j]+=eps; _, rv_e = engine.get_d4_diagnostic_residue(tmp, False)
            jac[:, j] = (rv_e - rv)/eps
        du = np.linalg.solve(jac[2:-2, 2:-2]+np.eye(Nx-4)*1e-8, -rv[2:-2])
        c[2:-2] += np.clip(du, -0.2/(Nx-1), 0.2/(Nx-1)) * 0.4

    # Method B: Diagnostic
    uB, eB, rB = engine.solve(xB, yB)
    for c in [xB, yB]:
        rv_w, _ = engine.get_d4_diagnostic_residue(c, True)
        jac = np.zeros((Nx, Nx))
        for j in range(2, Nx-2):
            tmp = c.copy(); tmp[j]+=eps; rv_w_e, _ = engine.get_d4_diagnostic_residue(tmp, True)
            jac[:, j] = (rv_w_e - rv_w)/eps
        du = np.linalg.solve(jac[2:-2, 2:-2]+np.eye(Nx-4)*1e-8, -rv_w[2:-2])
        c[2:-2] += np.clip(du, -0.2/(Nx-1), 0.2/(Nx-1)) * 0.4

    for k, v in zip(['xA','yA','eA','rA','xB','yB','eB','rB'], [xA, yA, eA, rA, xB, yB, eB, rB]):
        history[k].append(v.copy())
    sys.stdout.write(f'\r▶ 진행: {f+1}/{TOTAL_FRAMES} | Error B: {np.mean(eB):.4e}')
    sys.stdout.flush()

# --- 최종 리포트 출력 (교수님께서 요청하신 풍부한 표) ---
print("\n\n" + "="*95)
last = -1
eA_f, rA_f, eB_f, rB_f = history['eA'][last], history['rA'][last], history['eB'][last], history['rB'][last]

def get_stats(res_map):
    # bulk 1~N-1
    b1 = res_map[1:-1, 1:-1]
    # bulk 5~N-5
    b5 = res_map[5:-5, 5:-5]
    return np.max(b1), np.sum(b1), np.max(b5), np.sum(b5)

m1A, s1A, m5A, s5A = get_stats(rA_f)
m1B, s1B, m5B, s5B = get_stats(rB_f)

print(f"{'Final Iteration Analysis (Ultimate D4)':<45} | {'Method A (Plain)':<20} | {'Method B (Diag)'}")
print("-" * 95)
rows = [
    ("1. bulk abs max residue (1~N-1)", m1A, m1B),
    ("2. bulk sum residue (1~N-1)", s1A, s1B),
    ("3. bulk abs max residue (5~N-5)", m5A, m5B),
    ("4. bulk sum residue (5~N-5)", s5A, s5B),
    ("5. abs max error", np.max(eA_f), np.max(eB_f)),
    ("6. abs mean error", np.mean(eA_f), np.mean(eB_f))
]
for name, vA, vB in rows: print(f"{name:<45} | {vA:.10e} | {vB:.10e}")
print("="*95 + "\n")

# GIF 저장
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ve, vr = np.max(history['eA'][0]), np.max(history['rA'][0])
def update(frame):
    for ax in axes.flatten(): ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    XA, YA = np.meshgrid(history['xA'][frame], history['yA'][frame], indexing='ij')
    XB, YB = np.meshgrid(history['xB'][frame], history['yB'][frame], indexing='ij')
    axes[0,0].pcolormesh(XA, YA, history['eA'][frame], vmin=0, vmax=ve, cmap='magma', shading='auto'); axes[0,0].set_title("A. Error")
    axes[0,1].pcolormesh(XB, YB, history['eB'][frame], vmin=0, vmax=ve, cmap='magma', shading='auto'); axes[0,1].set_title("B. Error (Diag)")
    axes[1,0].pcolormesh(XA, YA, history['rA'][frame], vmin=0, vmax=vr, cmap='viridis', shading='auto'); axes[1,0].set_title("A. Residue")
    axes[1,1].pcolormesh(XB, YB, history['rB'][frame], vmin=0, vmax=vr, cmap='viridis', shading='auto'); axes[1,1].set_title("B. Residue (Diag)")
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=100)
ani.save('Ultimate_D4_Rich_Diagnostic.gif', writer=PillowWriter(fps=10))
plt.close()
print("▶ GIF 저장 완료: Ultimate_D4_Rich_Diagnostic.gif")