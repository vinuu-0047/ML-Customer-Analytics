from django.contrib import admin
from django.urls import path
from analytics.views import dashboard_view
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', dashboard_view, name='dashboard'), # This points the homepage to your AI view!
    path('', RedirectView.as_view(url='/analytics/')), # Change /analytics/ to your actual app path
]