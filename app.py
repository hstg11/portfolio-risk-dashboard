import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Portfolio Performance & Risk Dashboard")
st.write("Analyze returns, risk, and benchmark performance in one place.")


st.sidebar.header("Portfolio Inputs")

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    value="AAPL,MSFT, AMZN, META, GOOG"
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Convert input to list
tickers_list = [t.strip() for t in tickers.split(",")]

st.sidebar.subheader("🎚 Portfolio Weights (in %)")

weights = []
for t in tickers_list:
    w = st.sidebar.number_input(
        f"Weight for {t}",
        min_value=0.0,
        max_value=100.0,
        value=round(100 / len(tickers_list), 2)
    )
    weights.append(w)

# convert to decimals and normalize (e.g., 40 → 0.40)
weights = [w / 100 for w in weights]

if abs(sum(weights) - 1) > 0.001:
    st.warning("⚠️ Weights must add up to 100%. Adjust them to continue.")


# Download prices
raw = yf.download(
    tickers_list,
    start=start_date,
    end=end_date,
    auto_adjust=True,      # IMPORTANT for Indian stocks
    progress=False
)



tab_overview, tab_risk, tab_benchmark, tab_monte = st.tabs(
    ["📊 Overview", "⚠️ Risk", "📈 Benchmark", "🎲 Monte Carlo"]
)

with tab_overview:
    st.subheader("📈 Price History")
    # Always use Close, auto-adjusted avoids Adj Close issues
    data = raw["Close"]

    # Use Adjusted if available, else Close
    if "Adj Close" in raw.columns:
        data = raw["Adj Close"]
    else:
        data = raw["Close"]

    data = data.dropna()
    import altair as alt

    price_df = data.reset_index().melt('Date', var_name='Ticker', value_name='Price')

    chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X(
                'Date:T',
                axis=alt.Axis(
                    format='%b %Y',
                    labelAngle=0,
                    labelOverlap=True
                )
            ),
            y=alt.Y('Price:Q'),
            color=alt.Color(
                'Ticker:N',
                legend=alt.Legend(orient='bottom', title=None)   # <-- move legend below
            )
        )
        .properties(
            height=420,
            width=1000
        )
    )

    st.altair_chart(chart, use_container_width=False)

    st.subheader("📊 Daily Returns")

    returns = data.pct_change().dropna()

    # show only last rows, formatted as %
    st.dataframe(
        returns.tail().style.format("{:.2%}")
    )

if returns.empty:
    st.warning("No price data available for the selected tickers/date range.")
    st.stop()


    # if multiple stocks → equal weight for now
    # weighted portfolio return (matrix multiplication style)
if returns.shape[1] != len(weights):
    weights = np.repeat(1/returns.shape[1], returns.shape[1])

    portfolio_return = returns.dot(weights)

    cumulative = (1 + portfolio_return).cumprod()

    st.subheader("📈 Weighted Cumulative Return (%)")

    cumulative_pct = (cumulative - 1) * 100

    import altair as alt

    cum_df = cumulative_pct.reset_index()
    cum_df.columns = ["Date", "Cumulative %"]

    chart_cum = (
        alt.Chart(cum_df)
        .mark_line()
        .encode(
            x=alt.X(
                "Date:T",
                axis=alt.Axis(
                    format="%b %Y",      # Month + Year
                    labelAngle=0,
                    labelOverlap=True
                )
            ),
            y=alt.Y("Cumulative %:Q"),
        )
        .properties(
            height=420,
            width=1000
        )
    )

    st.altair_chart(chart_cum, use_container_width=False)
# ⚠️ make sure this runs BEFORE ANY RISK METRICS OR MONTE CARLO

portfolio_return = returns.dot(weights)
# ---- PORTFOLIO RETURN SERIES ----
portfolio_return = returns.dot(weights)

# ---- CUMULATIVE GROWTH INDEX ----
cumulative = (1 + portfolio_return).cumprod()


with tab_risk:
    st.header("📉 Portfolio Risk Metrics")
    trading_days = 252
    volatility = (portfolio_return.std() * (trading_days ** 0.5)) * 100
    sharpe = (portfolio_return.mean() / portfolio_return.std()) * (trading_days ** 0.5)

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    var_95 = portfolio_return.quantile(0.05) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Volatility (Annualized)", f"{volatility:.2f}%")
    col1.caption("Measures how **choppy** returns are. Higher = riskier.")

    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.caption("Return earned **per unit of risk**. >1 is generally strong.")

    col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    col3.caption("Worst peak-to-trough fall — shows **pain during crashes**.")

    col4.metric("VaR (95%)", f"{var_95:.2f}%")
    col4.caption("On a bad day, you may lose **about this much or worse** (5% chance).")



    st.subheader("🔄 Rolling Risk (Volatility & Sharpe)")
    st.caption("Rolling metrics show **how risk and performance change over time**, instead of one static value.")

    st.sidebar.subheader("📐 Rolling Window Settings")

    window = st.sidebar.slider(
        "Rolling window (trading days)",
        min_value=20,
        max_value=252,
        value=60,
        step=5
    )



    rolling_vol = (portfolio_return.rolling(window).std() * (252 ** 0.5)) * 100

    rolling_sharpe = (
        portfolio_return.rolling(window).mean() /
        portfolio_return.rolling(window).std()
    ) * (252 ** 0.5)

    import altair as alt

    # Rolling Volatility Chart
    vol_df = rolling_vol.reset_index()
    vol_df.columns = ["Date", "Rolling Volatility (%)"]

    chart_vol = (
        alt.Chart(vol_df)
        .mark_line(color="#e63946")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y="Rolling Volatility (%):Q"
        )
        .properties(title="Rolling Annualized Volatility", height=320, width=950)
    )

    st.altair_chart(chart_vol, use_container_width=False)


    # Rolling Sharpe Chart
    sharpe_df = rolling_sharpe.reset_index()
    sharpe_df.columns = ["Date", "Rolling Sharpe"]

    chart_sharpe = (
        alt.Chart(sharpe_df)
        .mark_line(color="#1d3557")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y="Rolling Sharpe:Q"
        )
        .properties(title="Rolling Sharpe Ratio", height=320, width=950)
    )

    st.altair_chart(chart_sharpe, use_container_width=False)

    with st.expander("🔍 What does Rolling Volatility & Sharpe mean?"):
        st.markdown("""
    **Rolling Volatility:**  
    Shows when markets shifted from **calm** to **turbulent**.

    **Rolling Sharpe:**  
    Reveals when the portfolio delivered **good vs poor risk-adjusted returns**.
        """)



with tab_benchmark:
    st.header("📊 Portfolio vs Benchmark")
    st.sidebar.subheader("📌 Benchmark Comparison")

    benchmark_symbol = st.sidebar.selectbox(
        "Select benchmark index:",
        ("^NSEI", "^GSPC", "^NDX", "^BSESN"),   # Nifty, S&P, Nasdaq, Sensex
        index=0)

    benchmark_names = {
        "^NSEI": "Nifty 50",
        "^GSPC": "S&P 500",
        "^NDX": "Nasdaq 100",
        "^BSESN": "Sensex"
    }

    benchmark_label = benchmark_names.get(benchmark_symbol, "Benchmark")


    benchmark = yf.download(
        benchmark_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()


    # --- make both Series, then turn into DataFrames with names ---

    # --- Portfolio column ---
    if isinstance(cumulative, pd.DataFrame):
        portfolio_df = cumulative.copy()
        portfolio_df.columns = ["Portfolio"]
    else:
        portfolio_df = cumulative.to_frame(name="Portfolio")

    # --- Benchmark column (with readable name) ---
    if isinstance(benchmark_cumulative, pd.DataFrame):
        benchmark_df = benchmark_cumulative.copy()
        benchmark_df.columns = [benchmark_label]
    else:
        benchmark_df = benchmark_cumulative.to_frame(name=benchmark_label)

    compare_df = (
        pd.concat([portfolio_df, benchmark_df], axis=1)
        .dropna()
        .reset_index()
    )


    import altair as alt

    chart_compare = (
        alt.Chart(compare_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y=alt.Y("value:Q", title="Growth Index"),
            color=alt.Color("variable:N", legend=alt.Legend(title=None))
        )
        .transform_fold(
            ["Portfolio", benchmark_label],
            as_=["variable", "value"]
    )

        .properties(
            height=420,
            width=1000,
            title="Portfolio vs Benchmark Performance"
        )
    )


    st.altair_chart(chart_compare, use_container_width=False)

    # make sure both are 1-D
    portfolio_total = cumulative.squeeze().iloc[-1] - 1
    benchmark_total = benchmark_cumulative.squeeze().iloc[-1] - 1

    alpha = (portfolio_total - benchmark_total) * 100
    alpha = float(alpha)
    color = "green" if alpha >= 0 else "red"

    st.markdown(
        f"""
        <h4>📈 Alpha vs {benchmark_label}: 
        <span style='color:{color}'>{alpha:.2f}%</span>
        </h4>
        """,
        unsafe_allow_html=True
    )

with tab_monte:
    
    st.header("🎲 Monte Carlo Simulation")

    st.write("""
    We simulate thousands of possible future paths for this portfolio
    using historical returns and volatility.  
    This helps estimate **best-case, worst-case, and typical outcomes**.
    """)

    # daily portfolio returns (already computed earlier)
    daily_returns = portfolio_return

    mu = daily_returns.mean()
    sigma = daily_returns.std()

    st.subheader("Simulation Settings")

    num_days = st.slider(
        "Number of days to simulate (trading days)",
        min_value=60,
        max_value=756,
        value=252,
        step=10
    )

    num_sims = st.slider(
        "Number of simulations",
        min_value=100,
        max_value=3000,
        value=200,
        step=100
    )

    @st.cache_data(show_spinner=False)
    def run_simulation(mu, sigma, num_days, num_sims):
        start_value = 1
        sim_results = np.zeros((num_days, num_sims))

        for s in range(num_sims):
            daily_random_returns = np.random.normal(mu, sigma, num_days)
            sim_path = start_value * np.cumprod(1 + daily_random_returns)
            sim_results[:, s] = sim_path

        return sim_results


    # ---- CALL ONLY ONCE ----
    sim_results = run_simulation(mu, sigma, num_days, num_sims)

    st.subheader("Simulated Future Portfolio Paths")

    sample = min(60, num_sims)
    sim_df = pd.DataFrame(sim_results[:, :sample])
    sim_df["Day"] = np.arange(1, num_days + 1)

    sim_melted = sim_df.melt(id_vars="Day", var_name="Simulation", value_name="Value")

    p5  = np.percentile(sim_results, 5, axis=1)
    p50 = np.percentile(sim_results, 50, axis=1)
    p95 = np.percentile(sim_results, 95, axis=1)

    percentile_df = pd.DataFrame({
        "Day": np.arange(1, num_days + 1),
        "P5": p5,
        "P50": p50,
        "P95": p95
    })

    # --- Band (correct shading) ---
    # LOWER band line (P5)
    lower_line = (
        alt.Chart(percentile_df)
        .mark_line(color="#d61f1f")      # red dashed
        .encode(
            x="Day:Q",
            y=alt.Y("P5:Q", axis=alt.Axis(title="Portfolio Value (x)"))
        )
    )

    # UPPER band line (P95)
    upper_line = (
        alt.Chart(percentile_df)
        .mark_line(color="#4ade80")      # green dashed
        .encode(
            x="Day:Q",
            y="P95:Q"
        )
    )

    # MEDIAN line (still solid orange)
    median_line = (
        alt.Chart(percentile_df)
        .mark_line(color="orange", size=2)
        .encode(
            x="Day:Q",
            y="P50:Q"
        )
    )


    # --- Sample simulation paths ---
    chart_mc = (
        alt.Chart(sim_melted)
        .mark_line(opacity=0.1)
        .encode(
            x="Day:Q",
            y="Value:Q"
        )
        .properties(height=420, width=900)
    )

    final_chart = lower_line + upper_line + chart_mc + median_line
    st.altair_chart(final_chart, use_container_width=False)



    st.subheader("Monte Carlo Summary (End of Period)")

    final_values = sim_results[-1, :]
    p5  = np.percentile(final_values, 5)
    p50 = np.percentile(final_values, 50)
    p95 = np.percentile(final_values, 95)

    col1, col2, col3 = st.columns(3)

    # Bad Case (P5) — RED
    col1.markdown(
        f"""
        <div style="
            padding:14px;
            border-radius:12px;
            background-color:#111827;
            border:1px solid #2d2d2d;
        ">
            <div style="font-size:14px;opacity:0.85;">📉 Bad Case (5th %ile)</div>
            <div style="font-size:30px;font-weight:200;color:#d61f1f;">
                {p5:.2f}x
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Median (P50) — BLUE
    col2.markdown(
        f"""
        <div style="
            padding:14px;
            border-radius:12px;
            background-color:#111827;
            border:1px solid #2d2d2d;
        ">
            <div style="font-size:14px;opacity:0.85;">📊 Median (50th %ile)</div>
            <div style="font-size:30px;font-weight:200;color:orange;">
                {p50:.2f}x
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Good Case (P95) — GREEN
    col3.markdown(
        f"""
        <div style="
            padding:14px;
            border-radius:12px;
            background-color:#111827;
            border:1px solid #2d2d2d;
        ">
            <div style="font-size:14px;opacity:0.85;">📈 Good Case (95th %ile)</div>
            <div style="font-size:30px;font-weight:200;color:#4ade80;">
                {p95:.2f}x
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



    st.write(f"""
    ➡ With 90% confidence, your portfolio may end between **{p5:.2f}x** and **{p95:.2f}x** 
    over the next {num_days} days.
    """)

    st.caption("""
    These estimates come from Monte Carlo simulations based on historical volatility 
    and returns. They do **not** guarantee outcomes, but help visualize risk and uncertainty.
    """)




st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 70px;
        right: 15px;

        font-size: 13px;
        font-weight: 600;
        letter-spacing: .3px;

        color: #b00020;                  /* elegant deep red */
        border: 1.5px solid #b00020;     /* thin classy outline */
        padding: 5px 10px;
        border-radius: 999px;            /* pill shape */

        background: transparent;         /* NO fill */
        backdrop-filter: blur(2px);      /* subtle glassy feel */
        box-shadow: 0 2px 6px rgba(0,0,0,.06);

        z-index: 1000;
    }
    </style>

    <div class="footer">
        Harbhajan The Great
    </div>
    """,
    unsafe_allow_html=True
)

