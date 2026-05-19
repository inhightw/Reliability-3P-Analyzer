import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

def generate_probability_plot(r, total_n, f_t_points, F_vals, f_t, s_t):
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    
    # 決定分布類型與變換
    is_weibull = "Weibull" in r["Method"]
    gamma = r.get("Gamma(Loc)", 0.0)
    if gamma is None:
        gamma = 0.0
        
    # 定義機率尺標變換
    if is_weibull:
        def trans(F): return np.log(-np.log(1 - F))
        dist_name = "3P-Weibull" if gamma > 0 else "2P-Weibull"
    else:
        def trans(F): return norm.ppf(F)
        dist_name = "3P-Lognormal" if gamma > 0 else "2P-Lognormal"
        
    # 數據變換
    y_data = trans(F_vals)
    x_unadj = f_t_points
    x_adj = f_t_points - gamma
    
    # 繪製數據點
    if gamma > 0:
        ax.scatter(x_adj, y_data, color="blue", marker="o", s=70, label="Adj Data Points (t - γ)", zorder=5)
        ax.scatter(x_unadj, y_data, color="black", marker="d", s=70, label="Unadj Data Points (t)", zorder=4)
    else:
        ax.scatter(x_unadj, y_data, color="black", marker="d", s=70, label="Data Points (t)", zorder=4)
        
    # 繪製擬合線
    # 生成平滑的 t_adj 網路，避開 0 或負值
    min_x = max(1e-2, min(x_adj) * 0.1)
    max_x = max(x_unadj) * 1.5
    t_adj_grid = np.geomspace(min_x, max_x - gamma if gamma > 0 else max_x, 200)
    
    if is_weibull:
        beta = r["Beta(Shape)"]
        eta = r["Eta(Scale)"]
        y_fit = beta * np.log(t_adj_grid) - beta * np.log(eta)
    else:
        sigma = r["Sigma"]
        scale_l = r["Eta(Scale)"]
        y_fit = (np.log(t_adj_grid) - np.log(scale_l)) / sigma
        
    if gamma > 0:
        ax.plot(t_adj_grid, y_fit, color="blue", linestyle="-", linewidth=2.5, label="Adjusted Line (t - γ)")
        ax.plot(t_adj_grid + gamma, y_fit, color="black", linestyle="-", linewidth=2.5, label="Unadjusted Line (t)")
    else:
        ax.plot(t_adj_grid, y_fit, color="black", linestyle="-", linewidth=2.5, label="Fitted Line (t)")
        
    # 設定 X 軸為對數標尺
    ax.set_xscale("log")
    ax.set_xlabel("Time (t) (hr)", fontsize=11, color="red", fontweight="bold")
    
    # 設定自訂 Y 軸機率標尺 ticks
    pcts = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
    pct_labels = ["1.0%", "5.0%", "10.0%", "20.0%", "30.0%", "50.0%", "70.0%", "80.0%", "90.0%", "95.0%", "99.0%"]
    y_ticks = trans(pcts)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(pct_labels)
    ax.set_ylim(trans(0.005), trans(0.995))
    ax.set_ylabel("Unreliability, F(t) = 100-R(t) (%)", fontsize=11, color="red", fontweight="bold")
    
    # 標題
    ax.set_title(f"Probability - {('Weibull' if is_weibull else 'Lognormal')}", fontsize=14, color="red", fontweight="bold", pad=15)
    
    # 格線與外觀
    ax.grid(True, which="both", linestyle="-", color="#d3d3d3", alpha=0.7)
    ax.set_facecolor("#fafafa")
    
    # 右上角資訊框 (Weibull++ 風格)
    info_text = f"Probability\nData1\n{dist_name}\nMethod: {r['Method'].split('(')[-1].replace(')', '')}\nF={len(f_t)}/S={len(s_t)}"
    ax.text(1.02, 0.95, info_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))
            
    # 左下角參數標註
    if is_weibull:
        param_text = f"Beta = {r['Beta(Shape)']:.6f}, Eta = {r['Eta(Scale)']:.6f}, Gamma = {gamma:.6f}"
    else:
        param_text = f"Sigma = {r['Sigma']:.6f}, Mu = {r['Mu']:.6f}, Gamma = {gamma:.6f}"
    ax.text(0.02, -0.12, param_text, transform=ax.transAxes, fontsize=9, fontweight="bold", color="black")
    
    # 右下角簽名與時間 (對標 Moxa/Henry Luo 簽章風格)
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sig_text = f"Henry Luo\nMoxa\n{time_str}"
    ax.text(0.82, -0.15, sig_text, transform=ax.transAxes, fontsize=8, color="#555555")
    
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig
