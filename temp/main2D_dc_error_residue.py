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

dx_base = 1.0 / (Nx - 1)
dy_base = 1.0 / (Ny - 1)

def exact_sol(x, y):
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(k_wave * np.pi * X) * np.sin(k_wave * np.pi * Y)
    f = -2.0 * (k_wave * np.pi)**2 * u
    return u, f

def build_S(u_vec):
    n = len(u_vec)
    h = np.diff(u_vec)
    Df = sp.diags([-1.0/h, 1.0/h], [0, 1], shape=(n-1, n))
    Db = sp.diags([-1.0, 1.0], [-1, 0], shape=(n, n-1))
    return (Db @ Df).tocsr()

def solve_with_residue(x, y):
    nx, ny = len(x), len(y)
    Sx, Sy = build_S(x), build_S(y)
    S_2d = sp.kron(Sx, sp.eye(ny)) + sp.kron(sp.eye(nx), Sy)
    u_ex, f_ex = exact_sol(x, y)
    hx, hy = np.diff(x), np.diff(y)
    wx = np.zeros(nx); wx[1:-1]=(hx[:-1]+hx[1:])/2; wx[0]=hx[0]/2; wx[-1]=hx[-1]/2
    wy = np.zeros(ny); wy[1:-1]=(hy[:-1]+hy[1:])/2; wy[0]=hy[0]/2; wy[-1]=hy[-1]/2
    W = np.outer(wx, wy).flatten()
    rhs = W * f_ex.flatten()
    S_bc = S_2d.tolil()
    mask = np.zeros((nx, ny), dtype=bool); mask[0,:]=mask[-1,:]=mask[:,0]=mask[:,-1]=True
    b_idx = np.where(mask.flatten())[0]
    for idx in b_idx: S_bc[idx, :] = 0; S_bc[idx, idx] = 1.0
    rhs[b_idx] = u_ex.flatten()[b_idx]
    u_num = spla.spsolve(S_bc.tocsr(), rhs).reshape((nx, ny))
    
    # 🎯 BC가 적용된 최종 행렬의 비대칭성 추출 (Interface Scar 포함)
    S_final = S_bc.toarray()
    R_mat = S_final - S_final.T
    res_node_wise = np.sqrt(np.sum(R_mat**2, axis=1)).reshape((nx, ny))
    return u_num, np.abs(u_num - u_ex), res_node_wise

def get_residue_logic(u_vec):
    S = build_S(u_vec).toarray()
    S_eff = S.copy()
    S_eff[0, :] = 0; S_eff[0, 0] = 1.0; S_eff[-1, :] = 0; S_eff[-1, -1] = 1.0
    R_mat = S_eff - S_eff.T
    return np.sqrt(np.sum(R_mat**2, axis=1))

# --- 초기화 ---
print(f"▶ 시스템 초기화: 격자 크기 {Nx}x{Ny}")
x_init = np.linspace(0, 1, Nx); y_init = np.linspace(0, 1, Ny)
for i in [8, 15, 22]:
    x_init[i:i+2] += dx_base * distortion_strength
    y_init[i:i+2] -= dy_base * distortion_strength

history = {'xA': [], 'yA': [], 'eA': [], 'rA': [], 'xB': [], 'yB': [], 'eB': [], 'rB': []}
xA, yA = x_init.copy(), y_init.copy()
xB, yB = x_init.copy(), y_init.copy()
uA_prev, _, _ = solve_with_residue(xA, yA)
eps = 1e-8

# --- 힐링 루프 (실시간 상태 보고) ---
print("\n[1단계] 자코비안 힐링 루프 가동 중...")
for f in range(TOTAL_FRAMES):
    # Method A
    uA_curr, eA, rA = solve_with_residue(xA, yA)
    u_diff = np.abs(uA_curr - uA_prev)
    ix, iy = np.unravel_index(np.argmax(u_diff[1:-1, 1:-1]), (Nx-2, Ny-2))
    ix, iy = ix+1, iy+1
    rx = get_residue_logic(xA); rval = rx[ix]
    xA_eps = xA.copy(); xA_eps[ix] += eps
    jx = (get_residue_logic(xA_eps)[ix] - rval) / eps
    xA[ix] -= (rval / (jx + 1e-10)) * 0.4
    uA_prev = uA_curr.copy()

    # Method B
    uB, eB, rB = solve_with_residue(xB, yB)
    rb = get_residue_logic(xB); jac = np.zeros((Nx, Nx))
    for j in range(1, Nx-1):
        xb_e = xB.copy(); xb_e[j] += eps
        jac[:, j] = (get_residue_logic(xb_e) - rb) / eps
    du = np.linalg.solve(jac[1:-1, 1:-1] + np.eye(Nx-2)*1e-10, -rb[1:-1])
    xB[1:-1] += np.clip(du, -dx_base*0.3, dx_base*0.3) * 0.6
    
    ryb = get_residue_logic(yB); jacy = np.zeros((Ny, Ny))
    for j in range(1, Ny-1):
        yb_e = yB.copy(); yb_e[j] += eps
        jacy[:, j] = (get_residue_logic(yb_e) - ryb) / eps
    duy = np.linalg.solve(jacy[1:-1, 1:-1] + np.eye(Ny-2)*1e-10, -ryb[1:-1])
    yB[1:-1] += np.clip(duy, -dy_base*0.3, dy_base*0.3) * 0.6

    history['xA'].append(xA.copy()); history['yA'].append(yA.copy()); history['eA'].append(eA.copy()); history['rA'].append(rA.copy())
    history['xB'].append(xB.copy()); history['yB'].append(yB.copy()); history['eB'].append(eB.copy()); history['rB'].append(rB.copy())

    # 실시간 프로그레스 바
    prog = (f + 1) / TOTAL_FRAMES * 100
    sys.stdout.write(f'\r▶ 진행률: {prog:5.1f}% | 현재 단계: {f+1}/{TOTAL_FRAMES} | Error B: {np.mean(eB):.4e}')
    sys.stdout.flush()

# --- 최종 리포트 출력 ---
print("\n\n" + "="*95)
last = -1
eA_fin, rA_fin = history['eA'][last], history['rA'][last]
eB_fin, rB_fin = history['eB'][last], history['rB'][last]
rows = [
    ("1. bulk abs max residue (1~N-1)", np.max(np.abs(rA_fin[1:-1, 1:-1])), np.max(np.abs(rB_fin[1:-1, 1:-1]))),
    ("2. bulk sum residue (1~N-1)", np.sum(rA_fin[1:-1, 1:-1]), np.sum(rB_fin[1:-1, 1:-1])),
    ("3. bulk abs max residue (5~N-5)", np.max(np.abs(rA_fin[5:-5, 5:-5])), np.max(np.abs(rB_fin[5:-5, 5:-5]))),
    ("4. bulk sum residue (5~N-5)", np.sum(rA_fin[5:-5, 5:-5]), np.sum(rB_fin[5:-5, 5:-5])),
    ("5. abs max error", np.max(eA_fin), np.max(eB_fin)),
    ("6. abs mean error", np.mean(eA_fin), np.mean(eB_fin))
]
print(f"{'Final Iteration Analysis':<45} | {'Method A (Local)':<20} | {'Method B (Global)'}")
print("-" * 95)
for name, vA, vB in rows: print(f"{name:<45} | {vA:.10e} | {vB:.10e}")
print("="*95)

# --- [중요] 애니메이션 생성 및 저장 ---
print("\n[2단계] GIF 애니메이션 렌더링 시작...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
max_e = np.max(history['eA'][0])
max_r = np.max(history['rA'][0])

def update(frame):
    for ax in axes.flatten():
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    XA, YA = np.meshgrid(history['xA'][frame], history['yA'][frame], indexing='ij')
    XB, YB = np.meshgrid(history['xB'][frame], history['yB'][frame], indexing='ij')
    
    axes[0,0].pcolormesh(XA, YA, history['eA'][frame], vmin=0, vmax=max_e, cmap='magma', shading='auto')
    axes[0,0].set_title(f"A. Local Error Step {frame}")
    axes[0,1].pcolormesh(XB, YB, history['eB'][frame], vmin=0, vmax=max_e, cmap='magma', shading='auto')
    axes[0,1].set_title(f"B. Global Residue Step {frame}")
    axes[1,0].pcolormesh(XA, YA, history['rA'][frame], vmin=0, vmax=max_r, cmap='viridis', shading='auto')
    axes[1,0].set_title("A. Residue Map")
    axes[1,1].pcolormesh(XB, YB, history['rB'][frame], vmin=0, vmax=max_r, cmap='viridis', shading='auto')
    axes[1,1].set_title("B. Residue Map")
    sys.stdout.write(f'\r▶ 렌더링 중: {frame+1}/{TOTAL_FRAMES}')
    sys.stdout.flush()

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=100)
ani.save('D2_Final_Verification.gif', writer=PillowWriter(fps=10))
plt.close()
print("\n\n▶ 저장이 완료되었습니다: D2_Final_Verification.gif")