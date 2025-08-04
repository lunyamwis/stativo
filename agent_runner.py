# --- Imports and Environment Setup ---
import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from smolagents import tool, HfApiModel, ToolCallingAgent
import warnings
warnings.filterwarnings("ignore")

_ = load_dotenv(find_dotenv())
login(os.getenv("HUGGINGFACE_API_KEY"))

# --- Sample Booksy acquisition channels data ---
channels_data = {
    "channel_name": [
        "Facebook Ads",
        "Google Ads",
        "YouTube Influencers",
        "Referral Program",
        "Event Sponsorships"
    ],
    "location": [
        "North America",
        "Europe",
        "North America",
        "Global",
        "Local"
    ],
    "reach_distance_km": [300, 450, 120, 80, 50],
    "is_domestic": [True, False, True, True, True],
    "cost_per_lead": [8.50, 7.90, 10.00, 3.50, 12.00],
    "engagement_cost": [15.00, 20.00, 30.00, 5.00, 25.00],
}
channels_df = pd.DataFrame(channels_data)

# --- Tools ---

@tool
def calculate_transport_cost(distance_km: float, order_volume: float) -> float:
    """
    Calculate delivery or engagement cost based on distance and order volume.

    Args:
        distance_km (float): Distance to customer location in kilometers.
        order_volume (float): Number of accounts or units served per day.

    Returns:
        float: Total cost to serve.
    """
    trucks_needed = np.ceil(order_volume / 300)
    cost_per_km = 1.20
    return distance_km * cost_per_km * trucks_needed

@tool
def calculate_tariff(base_cost: float, is_international: bool) -> float:
    """
    Apply international regulatory overhead to cost.

    Args:
        base_cost (float): The base acquisition cost before overhead.
        is_international (bool): Whether the channel is international.

    Returns:
        float: Regulatory overhead cost.
    """
    if is_international:
        return base_cost * 0.075
    return 0.0

@tool
def simulate_revenue_metrics_explicit(
    cost_per_lead: float,
    reach_distance_km: float,
    is_domestic: bool,
    engagement_cost: float,
    daily_volume: int = 15,
    contract_length_days: int = 365
) -> dict:
    """
    Simulate CAC, LTV, and LTV:CAC ratio for a SaaS business.

    Args:
        cost_per_lead (float): Cost to acquire each lead.
        reach_distance_km (float): Distance in kilometers for serving users.
        is_domestic (bool): Whether the campaign is domestic.
        engagement_cost (float): Additional fixed cost to engage audience.
        daily_volume (int): Daily client acquisition volume.
        contract_length_days (int): Contract duration in days.

    Returns:
        dict: Dictionary of CAC components and SaaS efficiency metrics.
    """
    base_cost = cost_per_lead * daily_volume
    cost_to_serve = calculate_transport_cost(reach_distance_km, daily_volume)
    regulatory_cost = calculate_tariff(base_cost, not is_domestic)
    cac = base_cost + cost_to_serve + regulatory_cost + engagement_cost

    arpu_daily = 12.00
    gross_margin = 0.8
    churn_rate = 0.2

    arpu_annual = arpu_daily * daily_volume * contract_length_days
    ltv = (arpu_annual * gross_margin) / churn_rate
    ltv_cac = ltv / cac if cac > 0 else np.nan

    return {
        "Base CAC": round(base_cost, 2),
        "Cost to Serve": round(cost_to_serve, 2),
        "Regulatory Cost": round(regulatory_cost, 2),
        "Engagement Cost": round(engagement_cost, 2),
        "Total CAC": round(cac, 2),
        "Estimated LTV": round(ltv, 2),
        "LTV:CAC": round(ltv_cac, 2),
    }

@tool
def plot_channel_metrics_chart(
    metric_name: str = "LTV:CAC",
    top_n: int = 5
) -> str:
    """
    Plot a bar chart for a given metric across acquisition channels.

    Args:
        metric_name (str): The name of the metric to visualize (e.g. 'LTV:CAC', 'Total CAC').
        top_n (int): Number of top channels to display.

    Returns:
        str: Base64-encoded PNG image of the chart.
    """
    if metric_name not in booksy_df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in dataset.")

    sorted_df = booksy_df.sort_values(by=metric_name, ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(sorted_df["channel_name"], sorted_df[metric_name], color="skyblue")
    plt.xlabel("Channel")
    plt.ylabel(metric_name)
    plt.title(f"Top {top_n} Channels by {metric_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# --- Data Enrichment ---
def enrich_dataframe_explicit(df: pd.DataFrame) -> pd.DataFrame:
    enriched_rows = []
    for _, row in df.iterrows():
        metrics = simulate_revenue_metrics_explicit(
            cost_per_lead=row["cost_per_lead"],
            reach_distance_km=row["reach_distance_km"],
            is_domestic=row["is_domestic"],
            engagement_cost=row["engagement_cost"],
            daily_volume=15,
            contract_length_days=365
        )
        enriched_rows.append(metrics)
    return df.join(pd.DataFrame(enriched_rows))

booksy_df = enrich_dataframe_explicit(channels_df)

# --- Agent Setup ---
model = HfApiModel(
    model="Qwen/Qwen2.5-72B-Instruct",
    provider="together",
    max_tokens=4096,
    temperature=0.1,
)

agent = ToolCallingAgent(
    model=model,
    tools=[
        calculate_transport_cost,
        calculate_tariff,
        simulate_revenue_metrics_explicit,
        plot_channel_metrics_chart,
    ],
    max_steps=5,
)

agent.logger.console.width = 72

# --- Task for Agent ---
task = """
Using the Booksy SaaS dataset, identify the top two channels based on LTV:CAC ratio. 
Explain why they perform better. Then generate a bar chart of LTV:CAC ratios.
"""

# --- Run Agent ---
output = agent.run(task, additional_args={"suppliers_data": booksy_df})

# --- Output ---
print("\n📊 Enriched Booksy Data:")
print(booksy_df)

print("\n🔍 Agent Insight:")
print(output)
