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
distortion_strength = 0.75
k_wave = 4
# =========================================================

class UltimateHealingEngine:
    def __init__(self, nx, ny, k):
        self.nx, self.ny = nx, ny
        self.k = k
        self.Ix, self.Iy = sp.eye(nx), sp.eye(ny)
        
        # 정답 및 소스항 미리 계산
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
        self.u_ex = np.sin(k * np.pi * grid_x) * np.sin(k * np.pi * grid_y)
        self.f_ex = -2.0 * (k * np.pi)**2 * self.u_ex
        
        # 경계 마스크 설정
        mask = np.zeros((nx, ny), dtype=bool)
        mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1] = True, True, True, True
        self.b_idx = np.where(mask.flatten())[0]

    def build_square_ops(self, coords):
        """교수님의 연산자 분해를 위해 NxN 정사각 행렬 D와 S를 조립합니다"""
        n = len(coords)
        h = np.diff(coords)
        
        # 1. 정사각 1차 미분 연산자 D (N x N)
        D = np.zeros((n, n))
        for i in range(n-1):
            inv_h = 1.0 / h[i]
            D[i, i] = -inv_h; D[i, i+1] = inv_h
        # 마지막 행 후진 차분
        D[-1, -1] = 1.0/h[-1]; D[-1, -2] = -1.0/h[-1]
        
        # 2. Stiffness 행렬 S (N x N)
        S = np.zeros((n, n))
        for i in range(1, n-1):
            S[i, i-1] = 1.0/h[i-1]
            S[i, i] = -(1.0/h[i-1] + 1.0/h[i])
            S[i, i+1] = 1.0/h[i]
        return sp.csr_matrix(D), sp.csr_matrix(S)

    def get_diagnostic_residue(self, coords):
        """교수님의 식 R(D^2) = D(RD) + (RD)D를 이용한 가중치 레지듀 산출"""
        D, S = self.build_square_ops(coords)
        R_S = (S - S.T).toarray()
        D_mat = D.toarray()
        
        # 진단항 계산
        T1 = np.abs(D_mat @ R_S) # D(RD): 격자 왜곡 기원 (치유 가능)
        T2 = np.abs(R_S @ D_mat) # (RD)D: 경계 주입 기원 (필연적 흉터)
        
        # 🎯 진단 가중치: 치유 가능한 T1의 비중이 높을수록 자코비안이 더 신뢰함
        t1_max = T1.max(axis=1)
        t2_max = T2.max(axis=1)
        weight = t1_max / (t1_max + t2_max + 1e-12)
        
        # 실제 연산 행렬(BC 포함)의 레지듀
        S_bc = S.tolil()
        S_bc[0,:], S_bc[-1,:], S_bc[0,0], S_bc[-1,-1] = 0, 0, 1.0, 1.0
        R_bc = S_bc.tocsr() - S_bc.tocsr().T
        res_vec = np.sqrt(np.array(R_bc.power(2).sum(axis=1)).flatten())
        
        # 가중치 적용: 흉터 성분(T2)이 큰 지점은 자코비안이 무시하도록 필터링
        return res_vec * weight, res_vec

    def solve(self, x, y):
        _, Sx = self.build_square_ops(x); _, Sy = self.build_square_ops(y)
        S_2d = sp.kron(Sx, self.Iy) + sp.kron(self.Ix, Sy)
        
        hx, hy = np.diff(x), np.diff(y)
        wx = np.zeros(self.nx); wx[1:-1]=(hx[:-1]+hx[1:])/2; wx[0]=hx[0]/2; wx[-1]=hx[-1]/2
        wy = np.zeros(self.ny); wy[1:-1]=(hy[:-1]+hy[1:])/2; wy[0]=hy[0]/2; wy[-1]=hy[-1]/2
        rhs = (np.outer(wx, wy).flatten()) * self.f_ex.flatten()
        
        S_bc = S_2d.tolil()
        for idx in self.b_idx: S_bc[idx, :] = 0; S_bc[idx, idx] = 1.0
        rhs[self.b_idx] = self.u_ex.flatten()[self.b_idx]
        
        u_num = spla.spsolve(S_bc.tocsr(), rhs).reshape((self.nx, self.ny))
        R_final = S_bc.tocsr() - S_bc.tocsr().T
        res_map = np.sqrt(np.array(R_final.power(2).sum(axis=1)).flatten()).reshape((self.nx, self.ny))
        return u_num, np.abs(u_num - self.u_ex), res_map

# --- 메인 프로세스 ---
engine = UltimateHealingEngine(Nx, Ny, k_wave)
x_init = np.linspace(0, 1, Nx); y_init = np.linspace(0, 1, Ny)
# 초기 왜곡 주입
dist = (1.0/(Nx-1)) * distortion_strength
for i in [8, 18, 25]:
    x_init[i:i+2] += dist; y_init[i:i+2] -= dist

# Method A (기본 전역 힐링) vs Method B (진단 기반 전역 힐링)
xA, yA = x_init.copy(), y_init.copy()
xB, yB = x_init.copy(), y_init.copy()
history = {'xA':[], 'yA':[], 'eA':[], 'rA':[], 'xB':[], 'yB':[], 'eB':[], 'rB':[]}
eps = 1e-8

print("\n▶ [Step 1] 진단 기반 전역 힐링 엔진 가동")
for f in range(TOTAL_FRAMES):
    # Method A: 일반 전역 힐링
    uA, eA, rA = engine.solve(xA, yA)
    for coord in [xA, yA]:
        _, rv = engine.get_diagnostic_residue(coord)
        jac = np.zeros((Nx, Nx))
        for j in range(1, Nx-1):
            tmp = coord.copy(); tmp[j] += eps
            _, rv_e = engine.get_diagnostic_residue(tmp)
            jac[:, j] = (rv_e - rv) / eps
        du = np.linalg.solve(jac[1:-1, 1:-1] + np.eye(Nx-2)*1e-10, -rv[1:-1])
        coord[1:-1] += np.clip(du, -0.3/(Nx-1), 0.3/(Nx-1)) * 0.6

    # Method B: [교수님 알고리즘] 진단 가중치 전역 힐링
    uB, eB, rB = engine.solve(xB, yB)
    for coord in [xB, yB]:
        rv_w, _ = engine.get_diagnostic_residue(coord)
        jac = np.zeros((Nx, Nx))
        for j in range(1, Nx-1):
            tmp = coord.copy(); tmp[j] += eps
            rv_w_e, _ = engine.get_diagnostic_residue(tmp)
            jac[:, j] = (rv_w_e - rv_w) / eps
        du = np.linalg.solve(jac[1:-1, 1:-1] + np.eye(Nx-2)*1e-10, -rv_w[1:-1])
        coord[1:-1] += np.clip(du, -0.3/(Nx-1), 0.3/(Nx-1)) * 0.6

    for k, v in zip(['xA','yA','eA','rA','xB','yB','eB','rB'], [xA, yA, eA, rA, xB, yB, eB, rB]):
        history[k].append(v.copy())
    
    sys.stdout.write(f'\r▶ 진행: {f+1}/{TOTAL_FRAMES} | Error B: {np.mean(eB):.4e}')
    sys.stdout.flush()

# --- 최종 리포트 출력 ---
print("\n\n" + "="*95)
last = -1
eA_f, rA_f, eB_f, rB_f = history['eA'][last], history['rA'][last], history['eB'][last], history['rB'][last]
rows = [
    ("1. bulk abs max residue (1~N-1)", np.max(np.abs(rA_f[1:-1, 1:-1])), np.max(np.abs(rB_f[1:-1, 1:-1]))),
    ("2. bulk sum residue (1~N-1)", np.sum(rA_f[1:-1, 1:-1]), np.sum(rB_f[1:-1, 1:-1])),
    ("3. bulk abs max residue (5~N-5)", np.max(np.abs(rA_f[5:-5, 5:-5])), np.max(np.abs(rB_f[5:-5, 5:-5]))),
    ("4. bulk sum residue (5~N-5)", np.sum(rA_f[5:-5, 5:-5]), np.sum(rB_f[5:-5, 5:-5])),
    ("5. abs max error", np.max(eA_f), np.max(eB_f)),
    ("6. abs mean error", np.mean(eA_f), np.mean(eB_f))
]
print(f"{'Final Iteration Analysis (Ultimate)':<45} | {'Method A (Plain)':<20} | {'Method B (Diag)'}")
print("-" * 95)
for name, vA, vB in rows: print(f"{name:<45} | {vA:.10e} | {vB:.10e}")
print("="*95 + "\n")

# --- 시각화 및 GIF 저장 ---
print("[Step 2] GIF 애니메이션 저장 중...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ve, vr = np.max(history['eA'][0]), np.max(history['rA'][0])

def update(frame):
    for ax in axes.flatten(): ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    XA, YA = np.meshgrid(history['xA'][frame], history['yA'][frame], indexing='ij')
    XB, YB = np.meshgrid(history['xB'][frame], history['yB'][frame], indexing='ij')
    axes[0,0].pcolormesh(XA, YA, history['eA'][frame], vmin=0, vmax=ve, cmap='magma', shading='auto')
    axes[0,0].set_title(f"A. Plain Global Step {frame}")
    axes[0,1].pcolormesh(XB, YB, history['eB'][frame], vmin=0, vmax=ve, cmap='magma', shading='auto')
    axes[0,1].set_title(f"B. Diagnostic Global Step {frame}")
    axes[1,0].pcolormesh(XA, YA, history['rA'][frame], vmin=0, vmax=vr, cmap='viridis', shading='auto')
    axes[1,1].pcolormesh(XB, YB, history['rB'][frame], vmin=0, vmax=vr, cmap='viridis', shading='auto')
    sys.stdout.write(f'\r▶ 렌더링: {frame+1}/{TOTAL_FRAMES}')
    sys.stdout.flush()

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=100)
ani.save('Ultimate_D2_Diagnostic.gif', writer=PillowWriter(fps=10))
plt.close()
print("\n▶ 완료: Ultimate_D2_Diagnostic.gif")