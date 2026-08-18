import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =========================================================
# [설정] 가우시안 가변 계수 시스템 (D2 Only)
# =========================================================
N = 31
MAX_ITER = 80
TOLERANCE = 1e-12
DIST_FAC = 0.6
EPS = 1e-8

# 가우시안 분포 계수 (도메인 중앙에 위치)
def get_sigma_2d(x_grid, y_grid):
    xc, yc, w = 0.5, 0.5, 0.2
    return 1.0 + 2.0 * np.exp(-((x_grid-xc)**2 + (y_grid-yc)**2) / (2*w**2))

class GaussianOperatorArena:
    def __init__(self, n):
        self.n = n
        xu = np.linspace(0, 1, n); hu = np.diff(xu)
        Du = np.zeros((n, n))
        for i in range(n-1): Du[i, i] = -1.0/hu[i]; Du[i, i+1] = 1.0/hu[i]
        # Leibniz 기준점: 순수 미분 구조의 잔차
        self.ref_leibniz = (Du @ (Du - Du.T) + (Du - Du.T) @ Du)[1:-1, 1:-1].diagonal()

    def get_res_direct(self, c_x, c_y):
        # 1D 성분별로 계산 (2D 행렬 전체 비대칭은 연산량이 과하므로 대변인 지표 사용)
        n = len(c_x); hx = np.diff(c_x); hy = np.diff(c_y)
        # 가우시안 계수가 포함된 연산자 조립 (Direct는 이 전체를 목표로 함)
        def get_1d_R(c):
            sig = 1.0 + 2.0 * np.exp(-(c-0.5)**2 / (2*0.2**2))
            sm = 0.5 * (sig[:-1] + sig[1:])
            A = np.zeros((n, n))
            for i in range(1, n-1):
                vL = sm[i-1] * 2.0 / (hx[i-1] * (hx[i-1] + hx[i]))
                vR = sm[i] * 2.0 / (hx[i] * (hx[i-1] + hx[i]))
                A[i, i-1], A[i, i+1], A[i, i] = vL, vR, -(vL + vR)
            return np.sum(np.abs(A - A.T), axis=1)[1:-1]
        return np.concatenate([get_1d_R(c_x), get_1d_R(c_y)])

    def get_res_leibniz(self, c):
        n = len(c); h = np.diff(c)
        D = np.zeros((n, n))
        for i in range(n-1): D[i, i] = -1.0/h[i]; D[i, i+1] = 1.0/h[i]
        RD = D - D.T
        M = D @ RD + RD @ D
        return M[1:-1, 1:-1].diagonal() - self.ref_leibniz

def run_bench(mode):
    arena = GaussianOperatorArena(N)
    xc, yc = np.linspace(0, 1, N), np.linspace(0, 1, N)
    dx = 1.0/(N-1)
    # 초기 왜곡
    for i in [10, 20]: xc[i] += dx*DIST_FAC; yc[i] -= dx*DIST_FAC
    
    hist = []
    for f in range(MAX_ITER):
        if mode == 'direct':
            r = arena.get_res_direct(xc, yc)
        else:
            r = np.concatenate([arena.get_res_leibniz(xc), arena.get_res_leibniz(yc)])
            
        err = np.max(np.abs(r)); hist.append(err)
        if err < TOLERANCE: break
        
        jac = np.zeros((len(r), (N-2)*2))
        for j in range(N-2):
            xe, ye = xc.copy(), yc.copy(); xe[j+1] += EPS
            if mode == 'direct': re = arena.get_res_direct(xe, yc)
            else: re = np.concatenate([arena.get_res_leibniz(xe), arena.get_res_leibniz(yc)])
            jac[:, j] = (re - r) / EPS
            
            xe, ye = xc.copy(), yc.copy(); ye[j+1] += EPS
            if mode == 'direct': re = arena.get_res_direct(xc, ye)
            else: re = np.concatenate([arena.get_res_leibniz(xc), arena.get_res_leibniz(ye)])
            jac[:, N-2+j] = (re - r) / EPS
            
        du, _, _, _ = np.linalg.lstsq(jac, -r, rcond=None)
        du = np.clip(du, -dx*0.3, dx*0.3)
        xc[1:-1] += du[:N-2] * 0.6; yc[1:-1] += du[N-2:] * 0.6
        xc, yc = np.sort(xc), np.sort(yc)
    return xc, yc, hist

# 시뮬레이션
x1, y1, h1 = run_bench('direct')
x2, y2, h2 = run_bench('leibniz')

# --- [시각화: 가우시안 배경 위의 격자] ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

def draw_full_mesh(ax, x, y, title, color):
    # 가우시안 배경 표시
    gx, gy = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    sz = sigma_func_bg(gx, gy) # 가우시안 등고선
    ax.contourf(gx, gy, sz, levels=20, cmap='Greys', alpha=0.2)
    # 격자 그리기
    for xi in x: ax.plot([xi, xi], [y[0], y[-1]], color=color, lw=0.7, alpha=0.6)
    for yi in y: ax.plot([x[0], x[-1]], [yi, yi], color=color, lw=0.7, alpha=0.6)
    ax.set_title(title); ax.set_aspect('equal')

def sigma_func_bg(x, y): return np.exp(-((x-0.5)**2 + (y-0.5)**2) / (2*0.2**2))

draw_full_mesh(axes[0], x1, y1, "1. Healed Mesh (Direct)", 'blue')
draw_full_mesh(axes[1], x2, y2, "2. Healed Mesh (Leibniz)", 'red')

axes[2].semilogy(h1, 'b-o', label='Direct', markersize=3)
axes[2].semilogy(h2, 'r--x', label='Leibniz', markersize=3)
axes[2].set_title("3. Convergence on Gaussian L"); axes[2].legend(); axes[2].grid(True)

plt.tight_layout(); plt.savefig('Gaussian_D2_Battle.png')
print("▶ 가우시안 벤치마크 완료 (Gaussian_D2_Battle.png)")