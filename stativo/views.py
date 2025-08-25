import re
import shutil
from django.shortcuts import render

# Create your views here.
# views.py
import os
import glob
import uuid
import random
import pandas as pd
import matplotlib.pyplot as plt

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from .serializers import NaturalLanguageQuerySerializer, NaturalLanguageResponseSerializer
from .firebase_agent import FirestoreAgentService

from smolagents import tool, HfApiModel, CodeAgent
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

login(os.getenv("HUGGINGFACE_API_KEY"))


def home(request):
    """
    Render the home page.
    """
    return render(request, "dashboard/home.html")

# --- Tools --- #
@tool
def simulate_revenue_metrics_explicit(n: int = 100) -> pd.DataFrame:
    """
    Simulate SaaS revenue metrics for multiple acquisition channels.

    Args:
        n (int): Number of simulated rows to generate.

    Returns:
        pd.DataFrame: A DataFrame containing simulated SaaS marketing metrics including LTV, CAC, and LTV:CAC.
    """
    channels = [
        "Facebook Ads", "Instagram Ads", "Google Ads",
        "Email Campaign", "Referral", "Influencer", "Affiliate"
    ]
    data = []
    for _ in range(n):
        channel = random.choice(channels)
        spend = round(random.uniform(100, 2000), 2)
        leads = random.randint(10, 200)
        signups = int(leads * random.uniform(0.1, 0.5))
        avg_contract = round(random.uniform(50, 900), 2)
        retention_days = random.randint(30, 365)
        cost_per_lead = spend / leads
        cac = spend / (signups if signups != 0 else 1)
        ltv = (avg_contract * retention_days) / 30
        ltv_cac = ltv / (cac if cac != 0 else 1)

        data.append({
            "channel_name": channel,
            "ad_spend": spend,
            "leads": leads,
            "signups": signups,
            "avg_contract": avg_contract,
            "retention_days": retention_days,
            "cost_per_lead": cost_per_lead,
            "CAC": cac,
            "LTV": ltv,
            "LTV:CAC": ltv_cac
        })

    return pd.DataFrame(data)

# @tool
# def plot_channel_metrics_chart(df: pd.DataFrame, metric: str = "LTV:CAC", top_n: int = 5) -> str:
#     """
#     Plot a bar chart of acquisition channels by a given metric (e.g. LTV:CAC), save locally, and return its URL.

#     Args:
#         df (pd.DataFrame): DataFrame with channel metrics.
#         metric (str): The column/metric to plot (e.g. "LTV:CAC").
#         top_n (int): Number of top rows to display.

#     Returns:
#         str: URL path to the saved chart image.
#     """
#     top = df.sort_values(by=metric, ascending=False).head(top_n)

#     plt.figure(figsize=(10, 5))
#     bars = plt.bar(top["channel_name"], top[metric], color='skyblue')
#     plt.title(f"Top {top_n} Channels by {metric}")
#     plt.xlabel("Channel")
#     plt.ylabel(metric)
#     plt.xticks(rotation=30)
#     plt.tight_layout()

#     for bar in bars:
#         height = bar.get_height()
#         plt.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
#                      xytext=(0, 5), textcoords="offset points", ha='center')

#     chart_id = uuid.uuid4().hex
#     chart_path = os.path.join(settings.MEDIA_ROOT, "charts")
#     os.makedirs(chart_path, exist_ok=True)
#     file_path = os.path.join(chart_path, f"{chart_id}.png")

#     plt.savefig(file_path)
#     plt.close()

#     return {
#         "url":f"https://stativiz.com{settings.MEDIA_URL}charts/{chart_id}.png",
#         "key_info": {
#             "top_channels": top[["channel_name", metric]].to_dict(orient="records"),
#             "metric": metric
#         },
#         "explanation": "",
#     }



def move_png_files_glob(source_dir, destination_dir):
    """Move all PNG files using glob pattern matching"""
    os.makedirs(destination_dir, exist_ok=True)
    
    # Find all PNG files (case insensitive)
    png_files = glob.glob(os.path.join(source_dir, "*.png")) + \
                glob.glob(os.path.join(source_dir, "*.PNG"))
    
    for png_file in png_files:
        filename = os.path.basename(png_file)
        dest_path = os.path.join(destination_dir, filename)
        shutil.move(png_file, dest_path)
        print(f"Moved: {filename}")
    
    return [os.path.basename(f) for f in png_files]


# --- Django View --- #
class BooksyLLMAnalysisAPIView(APIView):

    def post(self, request):
        user_query = request.data.get("query", "")
        if not user_query:
            return Response({"error": "Query is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Simulate data
            booksy_df = simulate_revenue_metrics_explicit()

            # Agent setup
            model = HfApiModel(
                model="Qwen/Qwen2.5-72B-Instruct",
                provider="together",
                max_tokens=4096,
                temperature=0.1,
            )

            agent = CodeAgent(
                model=model,
                tools=[simulate_revenue_metrics_explicit
                       ],
                additional_authorized_imports=[
                    "matplotlib.pyplot",
                    "pandas",
                    "numpy",
                ],
                max_steps=10,
            )

            # Run agent
            task = user_query
            output = agent.run(task, additional_args={"suppliers_data": booksy_df})
            generated_chart_list = move_png_files_glob(settings.BASE_DIR, os.path.join(settings.MEDIA_ROOT, "charts"))
                

            # import pdb;pdb.set_trace()
            # Return output
            return Response({
                "query": user_query,
                "result": output,
                "generated_chart_list": [f"https://stativiz.com/media/charts/{url}" for url in generated_chart_list]
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        


class FirestoreQueryAPIView(APIView):
    """
    POST endpoint that accepts natural language query in JSON payload
    and returns natural language response from Firestore AI agent.
    """

    def post(self, request):
        serializer = NaturalLanguageQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        query = serializer.validated_data["query"]

        try:
            # Initialize agent service lazily inside request method to avoid __signature__ problem
            agent_service = FirestoreAgentService()

            response_text = agent_service.ask(query)
            response_serializer = NaturalLanguageResponseSerializer(data={"response": response_text})
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            # Log the exception in production code
            return Response(
                {"detail": f"Error processing query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
