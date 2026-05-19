import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min, norm
import scipy.optimize as opt
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Reliability 3P Analyzer", layout="wide")

def load_data(raw_text):
    all_t = []
    for line in raw_text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3:
            all_t.extend([(float(parts[2]), parts[1])] * int(parts[0]))
    all_t.sort(key=lambda x: x[0])
    
    total_n = len(all_t)
    unique_t = sorted(list(set(t for t, st in all_t)))
    km_results = []
    R_c = 1.0
    alive = total_n
    for t in unique_t:
        d = sum(1 for (time, st) in all_t if time == t and st == 'F')
        s = sum(1 for (time, st) in all_t if time == t and st == 'S')
        if alive > 0: R_c = R_c * (alive - d) / alive
        else: R_c = 0.0
        km_results.append((t, d, s, 1 - R_c))
        alive -= (d + s)

    f_t_points = np.array([res[0] for res in km_results if res[1] > 0])
    F_vals = np.array([res[3] for res in km_results if res[1] > 0])
    F_vals = np.clip(F_vals, 1e-5, 0.99999)

    f_t = np.array([t for t, st in all_t if st == 'F'], dtype=float)
    s_t = np.array([t for t, st in all_t if st == 'S'], dtype=float)
    f_c = np.ones_like(f_t)
    s_c = np.ones_like(s_t)
    
    return total_n, f_t_points, F_vals, f_t, f_c, s_t, s_c

def calc_aic_bic(n, k, ll):
    aic = 2*k - 2*ll
    bic = k*np.log(n) - 2*ll
    return aic, bic

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


def fit_weibull_3p_gamma_gt0(total_n, f_t, s_t):
    f_t_u, f_c_u = np.unique(f_t, return_counts=True)
    s_t_u, s_c_u = np.unique(s_t, return_counts=True)
    def nll(params):
        shape, loc, scale = params
        min_t = np.min(np.concatenate([f_t_u, s_t_u])) if len(s_t_u) > 0 else np.min(f_t_u)
        if shape <= 0 or scale <= 0 or loc >= min_t: return np.inf
        ll = 0.0
        if len(f_t_u) > 0: ll += np.sum(f_c_u * np.log(np.maximum(weibull_min.pdf(f_t_u, c=shape, loc=loc, scale=scale), 1e-300)))
        if len(s_t_u) > 0: ll += np.sum(s_c_u * np.log(np.maximum(weibull_min.sf(s_t_u, c=shape, loc=loc, scale=scale), 1e-300)))
        return -ll

    min_t = np.min(f_t_u)
    bounds = [(1e-5, None), (0.0, min_t - 1e-5), (1e-5, None)]
    best_ll = np.inf
    best_res = None
    for init_loc in [0.0, min_t*0.5, min_t*0.8, min_t*0.9]:
        res = opt.minimize(nll, [2.0, init_loc, np.mean(f_t_u)], method='L-BFGS-B', bounds=bounds)
        if res.fun < best_ll:
            best_ll = res.fun
            best_res = res
            
    beta, gamma, eta = best_res.x
    ll = -best_ll
    aic, bic = calc_aic_bic(total_n, 3, ll)
    return {"Method": "Weibull 3P (Gamma >= 0)", "Beta(Shape)": beta, "Eta(Scale)": eta, "Sigma": None, "Mu": None, "Gamma(Loc)": gamma, "LL": ll, "AIC": aic, "BIC": bic}

def fit_weibull_3p_hybrid(total_n, f_t_points, F_vals, f_t_u, f_c_u, s_t_u, s_c_u):
    Y_w = np.log(-np.log(1 - F_vals))
    def corr_w(gamma):
        if gamma >= min(f_t_points): return np.inf
        X = np.log(f_t_points - gamma)
        if np.std(X) == 0: return 0
        r = np.corrcoef(X, Y_w)[0, 1]
        return -abs(r)
    res_w = opt.minimize_scalar(corr_w, bounds=(0, min(f_t_points)-0.01), method='bounded')
    gamma = res_w.x if res_w.success else 0.0

    def nll(params):
        shape, scale = params
        if shape <= 0 or scale <= 0: return np.inf
        ll = 0.0
        valid_f = f_t_u - gamma
        if len(valid_f) > 0: ll += np.sum(f_c_u * np.log(np.maximum(weibull_min.pdf(valid_f, c=shape, scale=scale), 1e-300)))
        valid_s = s_t_u - gamma
        if len(valid_s) > 0: ll += np.sum(s_c_u * np.log(np.maximum(weibull_min.sf(valid_s, c=shape, scale=scale), 1e-300)))
        return -ll

    mle_w = opt.minimize(nll, [2.0, 100.0], method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])
    beta, eta = mle_w.x
    ll = -mle_w.fun
    aic, bic = calc_aic_bic(total_n, 3, ll)
    return {"Method": "Weibull 3P (Hybrid)", "Beta(Shape)": beta, "Eta(Scale)": eta, "Sigma": None, "Mu": None, "Gamma(Loc)": gamma, "LL": ll, "AIC": aic, "BIC": bic}

def fit_lognormal_3p_gamma_gt0(total_n, f_t, s_t):
    f_t_u, f_c_u = np.unique(f_t, return_counts=True)
    s_t_u, s_c_u = np.unique(s_t, return_counts=True)
    def nll(params):
        shape, loc, scale = params
        min_t = np.min(np.concatenate([f_t_u, s_t_u])) if len(s_t_u) > 0 else np.min(f_t_u)
        if shape <= 0 or scale <= 0 or loc >= min_t: return np.inf
        ll = 0.0
        if len(f_t_u) > 0: ll += np.sum(f_c_u * np.log(np.maximum(lognorm.pdf(f_t_u, s=shape, loc=loc, scale=scale), 1e-300)))
        if len(s_t_u) > 0: ll += np.sum(s_c_u * np.log(np.maximum(lognorm.sf(s_t_u, s=shape, loc=loc, scale=scale), 1e-300)))
        return -ll

    min_t = np.min(f_t_u)
    bounds = [(1e-5, None), (0.0, min_t - 1e-5), (1e-5, None)]
    best_ll = np.inf
    best_res = None
    for init_loc in [0.0, min_t*0.5, min_t*0.8, min_t*0.9]:
        res = opt.minimize(nll, [1.0, init_loc, np.mean(f_t_u)], method='L-BFGS-B', bounds=bounds)
        if res.fun < best_ll:
            best_ll = res.fun
            best_res = res
            
    sigma, gamma, scale_l = best_res.x
    mu = np.log(scale_l)
    ll = -best_ll
    aic, bic = calc_aic_bic(total_n, 3, ll)
    return {"Method": "Lognormal 3P (Gamma >= 0)", "Beta(Shape)": None, "Eta(Scale)": scale_l, "Sigma": sigma, "Mu": mu, "Gamma(Loc)": gamma, "LL": ll, "AIC": aic, "BIC": bic}

def fit_lognormal_3p_hybrid(total_n, f_t_points, F_vals, f_t_u, f_c_u, s_t_u, s_c_u):
    Y_l = norm.ppf(F_vals)
    def corr_l(gamma):
        if gamma >= min(f_t_points): return np.inf
        X = np.log(f_t_points - gamma)
        if np.std(X) == 0: return 0
        r = np.corrcoef(X, Y_l)[0, 1]
        return -abs(r)
    res_l = opt.minimize_scalar(corr_l, bounds=(0, min(f_t_points)-0.01), method='bounded')
    gamma = res_l.x if res_l.success else 0.0

    def nll(params):
        shape, scale = params
        if shape <= 0 or scale <= 0: return np.inf
        ll = 0.0
        valid_f = f_t_u - gamma
        if len(valid_f) > 0: ll += np.sum(f_c_u * np.log(np.maximum(lognorm.pdf(valid_f, s=shape, scale=scale), 1e-300)))
        valid_s = s_t_u - gamma
        if len(valid_s) > 0: ll += np.sum(s_c_u * np.log(np.maximum(lognorm.sf(valid_s, s=shape, scale=scale), 1e-300)))
        return -ll

    mle_l = opt.minimize(nll, [1.0, 100.0], method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])
    sigma, scale_l = mle_l.x
    mu = np.log(scale_l)
    ll = -mle_l.fun
    aic, bic = calc_aic_bic(total_n, 3, ll)
    return {"Method": "Lognormal 3P (Hybrid)", "Beta(Shape)": None, "Eta(Scale)": scale_l, "Sigma": sigma, "Mu": mu, "Gamma(Loc)": gamma, "LL": ll, "AIC": aic, "BIC": bic}

def main():
    st.title("📈 3-Parameter 統計分布分析工具 (MLE & Hybrid)")
    st.markdown("透過純數學最大概似估計 (MLE) 與商用軟體混合法 (Hybrid Method)，將失效資料配適 3-Parameter 的 Weibull 以及 Lognormal 分布的參數估計。")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📥 資料輸入區")
        default_data = "1\tF\t144\n1\tS\t192\n1\tS\t192\n1\tS\t192\n1\tS\t192\n1\tS\t192\n1\tF\t144\n1\tF\t192\n1\tF\t168\n1\tF\t192\n1\tS\t192\n1\tS\t192\n1\tF\t192\n1\tS\t192\n1\tS\t192\n1\tF\t192\n1\tS\t192\n1\tS\t192\n1\tS\t192\n1\tS\t192\n1\tF\t192\n1\tS\t192\n1\tF\t168\n1\tS\t192\n1\tF\t168\n1\tF\t192\n1\tF\t192\n1\tF\t168\n1\tF\t168\n1\tF\t168"
        user_input = st.text_area("請貼上您的資料 (格式: 數量 F/S 時間)", value=default_data, height=450)
        analyze_btn = st.button("🚀 開始執行全模型分析", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 分析結果與推薦模型")
        if analyze_btn:
            with st.spinner("正在執行 L-BFGS-B 與 Rank Regression 尋優演算法..."):
                try:
                    warnings.filterwarnings('ignore')
                    total_n, f_t_points, F_vals, f_t, f_c, s_t, s_c = load_data(user_input)
                    if len(f_t) < 3:
                        st.error("錯誤：至少需要 3 筆失效 (F) 資料才能收斂 3-Parameter 模型！")
                        st.stop()
                        
                    f_t_u, f_c_u = np.unique(f_t, return_counts=True)
                    s_t_u, s_c_u = np.unique(s_t, return_counts=True)

                    results = []
                    results.append(fit_weibull_3p_gamma_gt0(total_n, f_t, s_t))
                    results.append(fit_weibull_3p_hybrid(total_n, f_t_points, F_vals, f_t_u, f_c_u, s_t_u, s_c_u))
                    results.append(fit_lognormal_3p_gamma_gt0(total_n, f_t, s_t))
                    results.append(fit_lognormal_3p_hybrid(total_n, f_t_points, F_vals, f_t_u, f_c_u, s_t_u, s_c_u))
                    
                    # 畫面輸出整理
                    st.success("✔ 分析完成！請參考下方四個模型的獨立參數。")
                    
                    df = pd.DataFrame(results)
                    # format display
                    styled_df = df.style.format(precision=4)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    lowest_aic = np.inf
                    best_model = None
                    for r in results:
                        # 迴避數學無限大陷阱 (Beta<1且Gamma逼近t1)
                        is_trap = ("Weibull" in r['Method'] and r.get('Beta(Shape)', 10) < 1 and r.get('Gamma(Loc)', 0) > min(f_t)-1)
                        if not is_trap:
                            if r['AIC'] < lowest_aic:
                                lowest_aic = r['AIC']
                                best_model = r
                                
                    st.markdown("---")
                    st.markdown("### 🏆 系統綜合判斷 (自動決策引擎)")
                    if best_model:
                        st.info(f"**最佳工程模型（基於 AIC 懲罰最少且合乎物理事實）：** \n### ✨ {best_model['Method']} ✨")
                        st.write("該模型在資料解釋力 (Log-Likelihood) 與模型複雜度的懲罰取捨中，取得了最完美的平衡。")
                    
                    
                    # Details
                    cols = st.columns(4)
                    for i, r in enumerate(results):
                        with cols[i]:
                            st.markdown(f"**{r['Method']}**")
                            st.caption(f"LL: {r['LL']:.3f} | AIC: {r['AIC']:.3f}")
                            if r['Sigma'] is not None:
                                st.code(f"Sigma: {r['Sigma']:.4f}\nMu: {r['Mu']:.4f}\nGamma: {r['Gamma(Loc)']:.4f}")
                            else:
                                st.code(f"Beta: {r['Beta(Shape)']:.4f}\nEta: {r['Eta(Scale)']:.4f}\nGamma: {r['Gamma(Loc)']:.4f}")
                                
                    st.markdown("---")
                    st.markdown("### 📈 累積機率配適圖 (Probability Plot)")
                    model_names = [r["Method"] for r in results]
                    selected_model_name = st.selectbox(
                        "選擇要繪製配適圖的模型：", 
                        model_names, 
                        index=model_names.index(best_model["Method"]) if best_model else 0
                    )
                    selected_r = next(r for r in results if r["Method"] == selected_model_name)
                    
                    with st.spinner("正在產生 ReliaSoft Weibull++ 風格之高解析度機率配適圖..."):
                        try:
                            fig = generate_probability_plot(selected_r, total_n, f_t_points, F_vals, f_t, s_t)
                            st.pyplot(fig, clear_figure=True)
                        except Exception as plot_err:
                            st.error(f"配適圖產生失敗：{plot_err}")

                except Exception as e:
                    st.error(f"分析失敗，請檢查資料格式是否正確。系統回報：{e}")
        else:
            st.info("👈 請在左側確認您的資料後，點擊「開始執行全模型分析」。")

    st.markdown("---")
    st.markdown("### 📚 演算法與技術參考來源 (References)")
    st.markdown("1. Python `reliability` 套件說明：[Fit_Lognormal_3P](https://reliability.readthedocs.io/en/latest/API/Fitters/Fit_Lognormal_3P.html#reliability.Fitters.Fit_Lognormal_3P)")
    st.markdown("2. ReliaSoft Weibull++ 官方白皮書：[Discussion of Maximum Likelihood Estimation for the 3-Parameter Weibull Distribution](https://help.reliasoft.com/articles/content/hotwire/issue148/hottopics148.htm)")

if __name__ == '__main__':
    main()
