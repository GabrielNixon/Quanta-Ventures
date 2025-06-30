# ============================================================
#   RL PORTFOLIO – QUANTA FELLOWSHIP SUBMISSION BLOCK
#   In-sample: 2020-01-01 … 2024-12-31
#   Out-sample: 2025-01-01 … 2025-12-31
# ============================================================
import numpy as np, pandas as pd, yfinance as yf, gym, matplotlib.pyplot as plt
from gym import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------- 1. LOAD DAILY DATA ------------------------------
TICKERS = {"SPY": "S&P", "LQD": "BOND", "BIL": "TBILL", "GC=F": "GOLD"}
START, END = "2019-12-31", "2025-12-31"

px = yf.download(list(TICKERS.keys()), START, END,
                 auto_adjust=True, progress=False)["Close"]
px.columns = list(TICKERS.values())
ret = px.pct_change().dropna()                                    # daily returns

# ---------- 2. REGIME FLAG -----------------------------------
vol = ret.rolling(20).std().mean(axis=1)
spread = ret["BOND"] - ret["TBILL"]
risk_off = ((vol > vol.median()) & (spread > spread.median())).astype(int)
data = pd.concat([ret, risk_off.rename("REGIME")], axis=1).dropna()

# ---------- 3. TRAIN / TEST SPLIT ----------------------------
train_df = data.loc["2020":"2024"].copy()
test_df  = data.loc["2025"].copy()

# ---------- 4. GYM ENVIRONMENT -------------------------------
class RecurrentEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.r = df.iloc[:, :-1].values          # returns matrix
        self.reg = df["REGIME"].values           # regime flag
        self.n_assets = self.r.shape[1]
        self.action_space = spaces.Box(0, 1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.n_assets + 1,), dtype=np.float32)
        self.reset()

    # --- helpers
    def _obs(self):
        return np.concatenate([self.r[self.t - 1], [self.reg[self.t]]]).astype(np.float32)

    # --- Gym API
    def reset(self, *, seed=None, options=None):
        self.t, self.nav = 1, 1.0
        return self._obs(), {}

    def step(self, w):
        w = np.nan_to_num(w); w /= w.sum() + 1e-8        # normalise weights
        r_t = np.dot(w, self.r[self.t])                  # portfolio return
        self.nav *= (1 + r_t)
        self.t += 1
        done = (self.t >= len(self.r) - 1)
        info = {"nav": self.nav}
        return self._obs(), r_t, done, False, info

# ---------- 5. TRAIN RECURRENT PPO ---------------------------
TOTAL_STEPS = 250_000
train_env = DummyVecEnv([lambda: RecurrentEnv(train_df)])

model = RecurrentPPO(
    MlpLstmPolicy,
    train_env,
    learning_rate=1e-4,
    n_steps=256,
    batch_size=64,
    gamma=0.99,
    verbose=0,
)
model.learn(total_timesteps=TOTAL_STEPS)
model.save("quanta_rl_model")

# ---------- 6. HELPER: RUN STRATEGY & COLLECT NAV ------------
def run_strategy(env, agent):
    obs, _ = env.reset()
    navs = [1.0]
    done = False
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
        navs.append(info["nav"])
    return np.array(navs)

nav_train = run_strategy(RecurrentEnv(train_df), model)
nav_test  = run_strategy(RecurrentEnv(test_df),  model)

# ---------- 7. METRIC CALCULATOR -----------------------------
def perf_metrics(navs, periods_per_year=252):
    rets = navs[1:] / navs[:-1] - 1
    years = len(rets) / periods_per_year
    cagr = (navs[-1] / navs[0]) ** (1 / years) - 1
    max_dd = np.max((np.maximum.accumulate(navs) - navs) / np.maximum.accumulate(navs))
    calmar = cagr / (max_dd + 1e-8)
    sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(periods_per_year)
    return cagr, max_dd, calmar, sharpe

cagr_tr, dd_tr, calmar_tr, sharpe_tr = perf_metrics(nav_train)
cagr_te, dd_te, calmar_te, sharpe_te = perf_metrics(nav_test)

# ---------- 8. PRINT SUBMISSION METRICS ----------------------
print("\n=== OUT-OF-SAMPLE PERFORMANCE 2025 ===")
print(f"CAGR         : {cagr_te: .2%}")
print(f"Max Drawdown : {dd_te  : .2%}")
print(f"Calmar Ratio : {calmar_te: .2f}")
print(f"Sharpe Ratio : {sharpe_te: .2f}")
