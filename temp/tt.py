import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# =========================================================
# [설정] 연산자 자율 적응 시스템 (No Target Data)
# =========================================================
N = 31
MAX_ITER = 120
TOLERANCE = 1e-10
EPS = 1e-7

# 가변 계수: 중앙에서 급격히 변하는 정션 (연산자에게 스트레스 부여)
def get_sigma(x):
    return 1.0 + 5.0 * np.exp(-(x-0.5)**2 / (0.1**2))

class SelfAdaptiveArena:
    def __init__(self, n):
        self.n = n

    def calc_leibniz_energy(self, c):
        """연산자가 스스로 느끼는 라이프니츠 위배량 M = D(RD) + (RD)D"""
        n = len(c); h = np.diff(c)
        D = np.zeros((n, n))
        for i in range(n-1): D[i, i] = -1.0/h[i]; D[i, i+1] = 1.0/h[i]
        RD = D - D.T
        M = D @ RD + RD @ D
        # 🎯 핵심: 이 M의 크기 자체가 연산자가 요구하는 '치유량'임
        return M[1:-1, 1:-1].diagonal()

def run_self_driven_optimization():
    arena = SelfAdaptiveArena(N)
    # 🎯 초기 상태: 아무 편견 없는 완벽한 균등 격자
    xc = np.linspace(0, 1, N)
    yc = np.linspace(0, 1, N)
    dx_uni = 1.0/(N-1)
    
    xi, yi = xc.copy(), yc.copy()
    history = []
    
    for f in range(MAX_ITER):
        # 타겟 없이 현재 상태의 위배량만 계산
        rx = arena.calc_leibniz_energy(xc)
        ry = arena.calc_leibniz_energy(yc)
        r = np.concatenate([rx, ry])
        
        err = np.max(np.abs(r)); history.append(err)
        if err < TOLERANCE: break
        
        # Jacobian: 노드 이동에 따른 연산자 비대칭성 변화 계측
        jac = np.zeros((len(r), (N-2)*2))
        for j in range(N-2):
            xe = xc.copy(); xe[j+1] += EPS
            jac[:, j] = (np.concatenate([arena.calc_leibniz_energy(xe), ry]) - r) / EPS
            ye = yc.copy(); ye[j+1] += EPS
            jac[:, N-2+j] = (np.concatenate([rx, arena.calc_leibniz_energy(ye)]) - r) / EPS
            
        du, _, _, _ = np.linalg.lstsq(jac, -r, rcond=1e-3)
        # 연산자의 명령에 따른 격자 이주
        xc[1:-1] += np.clip(du[:N-2], -dx_uni*0.2, dx_uni*0.2) * 0.5
        yc[1:-1] += np.clip(du[N-2:], -dx_uni*0.2, dx_uni*0.2) * 0.5
        xc, yc = np.sort(xc), np.sort(yc)
        
    return xi, yi, xc, yc, history

# 실행
xi, yi, xf, yf, hist = run_self_driven_optimization()

# --- 🎯 [격자 시각화] 가로/세로 모든 선 표시 ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

def draw_grid(ax, x, y, title, color):
    for xv in x: ax.plot([xv, xv], [y[0], y[-1]], color=color, lw=0.6, alpha=0.4)
    for yv in y: ax.plot([x[0], x[-1]], [yv, yv], color=color, lw=0.6, alpha=0.4)
    ax.set_title(title); ax.set_aspect('equal')

draw_grid(axes[0], xi, yi, "1. Initial Uniform Mesh", 'blue')
draw_grid(axes[1], xf, yf, "2. Operator-Demanded Mesh (Adaptive)", 'red')

# 라이프니츠 잔차 감소량 (연산자의 '고통'이 줄어드는 과정)
axes[2].semilogy(hist, 'g-o', markersize=3)
axes[2].set_title("3. Minimizing Leibniz Violation (M)"); axes[2].grid(True)

plt.tight_layout(); plt.savefig('True_Operator_Driven_Grid.png')
print("▶ 연산자 자율 적응 리포트 생성 완료")