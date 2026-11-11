import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas_ta as ta

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingClassifier


# ==============================================================================
# STAGE 1: DATA INGESTION & 130+ INDICATOR GENERATION
# ==============================================================================
def load_and_generate_data(ticker="RELIANCE.NS", vol_ticker="^INDIAVIX", start="2018-01-01", end="2026-10-01"):
    print(f"[*] Downloading {ticker} and {vol_ticker} from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    vix = yf.download(vol_ticker, start=start, end=end, progress=False)[['Close']]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    print("[*] Calculating indicators across categories...")
    indicator_funcs = [
        # Momentum & Oscillators
        ('rsi', {'length': 14}), ('rsi', {'length': 7}), ('rsi', {'length': 21}),
        ('macd', {'fast': 12, 'slow': 26, 'signal': 9}),
        ('stoch', {'k': 14, 'd': 3}), ('cci', {'length': 14}), ('cci', {'length': 20}),
        ('willr', {'length': 14}), ('cmo', {'length': 14}), ('roc', {'length': 5}),
        ('roc', {'length': 10}), ('roc', {'length': 20}), ('mom', {'length': 10}),
        ('uo', {}), ('tsi', {}), ('ao', {}), ('bop', {}),
        # Trend & Moving Averages
        ('sma', {'length': 10}), ('sma', {'length': 20}), ('sma', {'length': 50}),
        ('ema', {'length': 9}), ('ema', {'length': 21}), ('ema', {'length': 50}),
        ('wma', {'length': 20}), ('hma', {'length': 20}), ('dema', {'length': 20}),
        ('adx', {'length': 14}), ('aroon', {'length': 14}), ('vortex', {'length': 14}),
        ('supertrend', {}), ('psar', {}),
        # Volatility & Bands
        ('bbands', {'length': 20, 'std': 2}), ('kc', {'length': 20}),
        ('donchian', {'lower_length': 10, 'upper_length': 20}),
        ('atr', {'length': 14}), ('natr', {'length': 14}),
        ('massi', {}), ('true_range', {}),
        # Volume & Flow
        ('obv', {}), ('mfi', {'length': 14}), ('cmf', {'length': 20}),
        ('vwap', {}), ('ad', {}), ('adosc', {}), ('eom', {'length': 14}),
        ('pvt', {}),
        # Statistics
        ('entropy', {'length': 10}), ('skew', {'length': 30}),
        ('kurtosis', {'length': 30}), ('variance', {'length': 20}),
        ('zscore', {'length': 20})
    ]

    generated_series = []
    for name, params in indicator_funcs:
        try:
            func = getattr(df.ta, name, None)
            if func is not None:
                res = func(**params)
                if isinstance(res, pd.DataFrame):
                    generated_series.append(res)
                elif isinstance(res, pd.Series):
                    generated_series.append(res.to_frame())
        except Exception:
            continue

    if generated_series:
        df = pd.concat([df] + generated_series, axis=1)

    # De-duplicate any duplicated column names
    df = df.loc[:, ~df.columns.duplicated()]

    # Option / Implied Volatility Features
    df['iv_level'] = vix['Close'].reindex(df.index).ffill()
    df['iv_rank_60'] = df['iv_level'].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1])
    realized_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['vrp_spread'] = df['iv_level'] - realized_vol

    # Forward 5-day return target for swing trading
    target_5d = df['close'].pct_change(5).shift(-5)

    # Filter out raw price levels and high-NaN columns
    raw_cols = ['open', 'high', 'low', 'close', 'adj close', 'volume']
    feature_cols = [c for c in df.columns if c not in raw_cols]
    X = df[feature_cols].copy()
    X = X.loc[:, X.isna().mean() < 0.20]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    common_idx = X.index.intersection(target_5d.dropna().index)
    print(f"[+] Feature matrix constructed: {X.shape[1]} indicators across {len(common_idx)} trading days.")
    return df.loc[common_idx, 'close'], X.loc[common_idx], target_5d.loc[common_idx]


# ==============================================================================
# STAGE 2: DBSCAN COMPRESSION & 90% INFORMATION CAPPING
# ==============================================================================
def compress_and_cap_indicators(X: pd.DataFrame, y: pd.Series, corr_threshold=0.75, info_cap=0.90, min_gain=0.008):
    print("[*] Computing non-linear Mutual Information with swing target...")
    X_ranked = X.rank(pct=True).fillna(0.5)
    mi_scores = pd.Series(mutual_info_regression(X_ranked, y, random_state=42), index=X.columns)

    print("[*] Calculating pairwise Spearman distance matrix...")
    corr_matrix = X.corr(method='spearman').abs().fillna(0)
    dist_arr = (1.0 - corr_matrix).clip(lower=0.0).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0.0)
    dist_matrix = pd.DataFrame(dist_arr, index=X.columns, columns=X.columns)

    print(f"[*] Running DBSCAN (eps={1.0 - corr_threshold:.2f}, min_samples=2)...")
    db = DBSCAN(eps=1.0 - corr_threshold, min_samples=2, metric='precomputed')
    labels = db.fit_predict(dist_matrix.values)
    labels = db.fit_predict(dist_matrix)

    summary = pd.DataFrame({'Indicator': X.columns, 'Cluster': labels, 'MI': mi_scores.values})

    # Pick 1 champion per cluster + keep orthogonal noise (-1) signals
    candidates = []
    for cid, group in summary[summary['Cluster'] != -1].groupby('Cluster'):
        candidates.append(group.sort_values('MI', ascending=False).iloc[0])
    for _, row in summary[summary['Cluster'] == -1].iterrows():
        candidates.append(row)

    candidates_df = pd.DataFrame(candidates).sort_values('MI', ascending=False)
    ordered_pool = candidates_df['Indicator'].tolist()

    print("[*] Performing greedy subspace reconstruction capping...")
    selected = []
    prev_r2 = 0.0
    history = []

    for ind in ordered_pool:
        trial_set = selected + [ind]
        model = Ridge(alpha=1.0).fit(X_ranked[trial_set], X_ranked)
        curr_r2 = max(0.0, r2_score(X_ranked, model.predict(X_ranked[trial_set]), multioutput='uniform_average'))
        gain = curr_r2 - prev_r2

        if gain >= min_gain or len(selected) < 3:
            selected.append(ind)
            prev_r2 = curr_r2
            history.append({
                'Step': len(selected),
                'Indicator': ind,
                'Cumulative_Info_%': round(curr_r2 * 100, 2),
                'Marginal_Gain_%': round(gain * 100, 2)
            })

        if curr_r2 >= info_cap:
            break

    return selected, pd.DataFrame(history), corr_matrix, labels


# ==============================================================================
# STAGE 3: VISUALIZATIONS (HEATMAP & RETENTION CURVE)
# ==============================================================================
def plot_results(corr_matrix, db_labels, history_df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Plot Clustered Heatmap
    # 1. Plot Clustered Heatmap
    dist_arr = (1.0 - corr_matrix).clip(lower=0.0).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0.0)
    condensed_dist = squareform(dist_arr)
    hc_linkage = linkage(condensed_dist, method='average')

    palette = sns.color_palette("tab20", len(set(db_labels)) + 1)
    color_map = {c: palette[i] if c != -1 else (0.2, 0.2, 0.2) for i, c in enumerate(sorted(set(db_labels)))}
    row_colors = pd.Series(db_labels, index=corr_matrix.columns).map(color_map)

    g = sns.clustermap(
        corr_matrix,
        row_linkage=hc_linkage,
        col_linkage=hc_linkage,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        figsize=(12, 12),
        xticklabels=False,
        yticklabels=False,
        cbar_pos=(0.02, 0.82, 0.03, 0.15)
    )
    g.ax_heatmap.set_title("Hierarchically Clustered Spearman Matrix & DBSCAN Clusters", pad=20, fontweight='bold')
    plt.show()

    # 2. Plot Information Retention Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['Step'], history_df['Cumulative_Info_%'], marker='o', color='#1f77b4', linewidth=2.5)
    plt.axhline(90, color='crimson', linestyle='--', label='90% Target Retention')
    plt.xticks(history_df['Step'], history_df['Indicator'], rotation=40, ha='right')
    plt.ylabel("Universe Variance Explained (%)")
    plt.xlabel("Selected DBSCAN Indicators (Added Sequentially)")
    plt.title(f"Subspace Compression: {len(history_df)} Indicators Reconstruct Full Universe", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# STAGE 4: LEAK-FREE NORMALIZATION & WALK-FORWARD BACKTEST
# ==============================================================================
def rolling_robust_norm(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    rolling_med = df.rolling(window=window, min_periods=20).median()
    q75 = df.rolling(window=window, min_periods=20).quantile(0.75)
    q25 = df.rolling(window=window, min_periods=20).quantile(0.25)
    rolling_iqr = (q75 - q25).replace(0, np.nan).ffill().fillna(1.0)
    return ((df - rolling_med) / rolling_iqr).clip(-4.0, 4.0).dropna()


def run_swing_trade_benchmark(X_norm, y_target, selected_cols, cost_bps=10):
    common_idx = X_norm.index.intersection(y_target.index)
    X_all = X_norm.loc[common_idx]
    X_sub = X_all[selected_cols]
    y_ret = y_target.loc[common_idx]

    # Binary swing label: 1 if forward 5-day return > 0.5% hurdle
    y_bin = (y_ret > 0.005).astype(int)

    # 70% Train / 30% Test with a 5-day embargo
    split_idx = int(len(X_all) * 0.70)
    embargo = 5

    X_all_tr, X_all_te = X_all.iloc[:split_idx], X_all.iloc[split_idx + embargo:]
    X_sub_tr, X_sub_te = X_sub.iloc[:split_idx], X_sub.iloc[split_idx + embargo:]
    y_tr, y_te_ret = y_bin.iloc[:split_idx], y_ret.iloc[split_idx + embargo:]

    # Strategy 1: Rule-Based Linear Composite
    train_corrs = np.sign(X_sub_tr.corrwith(y_ret.iloc[:split_idx]))
    comp_score = (X_sub_te * train_corrs).mean(axis=1)
    sig_rules = np.where(comp_score > 0.5, 1, np.where(comp_score < -0.5, -1, 0))

    # Strategy 2: ML on Full 130+ Indicators (Unpruned)
    clf_all = HistGradientBoostingClassifier(max_depth=4, l2_regularization=1.0, random_state=42)
    clf_all.fit(X_all_tr, y_tr)
    prob_all = clf_all.predict_proba(X_all_te)[:, 1]
    sig_ml_all = np.where(prob_all > 0.55, 1, np.where(prob_all < 0.45, -1, 0))

    # Strategy 3: ML on DBSCAN Compressed Subset
    clf_sub = HistGradientBoostingClassifier(max_depth=4, l2_regularization=1.0, random_state=42)
    clf_sub.fit(X_sub_tr, y_tr)
    prob_sub = clf_sub.predict_proba(X_sub_te)[:, 1]
    sig_ml_sub = np.where(prob_sub > 0.55, 1, np.where(prob_sub < 0.45, -1, 0))

    # Evaluate 5-day holding periods
    step = 5
    cost = cost_bps / 10000.0
    results = {}

    for name, sig in [
        ("1. Rule-Based (Top Indicators)", sig_rules),
        ("2. ML Full Universe (All Indicators)", sig_ml_all),
        ("3. ML DBSCAN Compressed Subset", sig_ml_sub)
    ]:
        s_p = pd.Series(sig, index=y_te_ret.index).iloc[::step]
        r_p = y_te_ret.iloc[::step]
        trades = s_p.abs()
        net_rets = (s_p * r_p) - (trades * cost)

        ann_sharpe = (net_rets.mean() / (net_rets.std() + 1e-9)) * np.sqrt(252 / step)
        cum_ret = (1 + net_rets).cumprod()
        mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()

        results[name] = {
            "Total_Return_%": round((cum_ret.iloc[-1] - 1) * 100, 2),
            "Annualized_Sharpe": round(ann_sharpe, 2),
            "Max_Drawdown_%": round(mdd * 100, 2),
            "Trade_Count": int(trades.sum())
        }

    return pd.DataFrame(results).T


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    close_prices, X_raw, y_target = load_and_generate_data()

    selected_cols, history, corr_mat, db_labels = compress_and_cap_indicators(
        X_raw, y_target, corr_threshold=0.75, info_cap=0.90
    )

    print("\n--- Subspace Capping Summary ---")
    print(history.to_string(index=False))

    print("\n[*] Applying rolling robust Z-score normalization...")
    X_norm = rolling_robust_norm(X_raw)

    print("[*] Running out-of-sample walk-forward backtest...")
    comparison = run_swing_trade_benchmark(X_norm, y_target, selected_cols)

    print("\n================ OUT-OF-SAMPLE BENCHMARK RESULTS ================")
    print(comparison)
    print("=================================================================")

    plot_results(corr_mat, db_labels, history)