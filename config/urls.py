from django.contrib import admin
from django.urls import path
from analytics.views import dashboard_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', dashboard_view, name='dashboard'), # This points the homepage to your AI view!
]