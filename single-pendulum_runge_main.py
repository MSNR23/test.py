import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定数
g = 9.80  # 重力加速度 
l = 1.0   # 振り子の長さ 
theta0 = np.pi / 2.0  # 初期の振り子の角度（90度にする）
omega0 = 0.0  # 初期の振り子の角速度

# 時間パラメータ
dt = 0.1  # 時間刻み幅
t_end = 10.0  # シミュレーション終了時間

# 運動方程式（ルンゲクッタ）
def runge_kutta(theta, omega, dt):
    k1_theta = omega
    k1_omega = - (g / l) * np.sin(theta)

    k2_theta = omega + 1/2 * dt * k1_omega
    k2_omega = - (g / l) * np.sin(theta + 0.5 * dt * k1_theta)

    k3_theta = omega + 0.5 * dt * k2_omega
    k3_omega = - (g / l) * np.sin(theta + 0.5 * dt * k2_theta)

    k4_theta = omega + dt * k3_omega
    k4_omega = - (g / l) * np.sin(theta + dt * k3_theta)

    theta_new = theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_new = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

    return theta_new, omega_new

# シミュレーションの初期化
t_values = np.arange(0, t_end, dt)
theta_values = []
omega_values = []

# 初期条件
theta = theta0
omega = omega0

# シミュレーション実行
for t in t_values:
    theta_values.append(theta)
    omega_values.append(omega)
    theta, omega = runge_kutta(theta, omega, dt)

# アニメーションの初期化
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)

# アニメーションの更新関数
def update(frame):
    x = l * np.sin(theta_values[frame])
    y = -l * np.cos(theta_values[frame])
    line.set_data([0, x], [0, y])
    return line,

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(t_values), blit=True)

# アニメーションの表示
plt.show()

# main関数
def main():
    plt.show()

if __name__ == "__main__":
    main()