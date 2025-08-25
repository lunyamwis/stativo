# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("",views.home, name="home"),
    path("metrics/agent/", views.BooksyLLMAnalysisAPIView.as_view(), name="booksy-agent"),
    path('firestore-agent/', views.FirestoreQueryAPIView.as_view(), name='firestore-agent'),
]
