# urls.py
from django.urls import path
from .views import BooksyLLMAnalysisAPIView

urlpatterns = [
    path("metrics/agent/", BooksyLLMAnalysisAPIView.as_view(), name="booksy-agent"),
]
