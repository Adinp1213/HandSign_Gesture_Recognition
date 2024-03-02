from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .import views


urlpatterns = [
				
				path('',views.Home,name='Home'),
				path('Detect_Alphabet/',views.Detect_Alphabet,name='Detect_Alphabet'),
				path('Detect/',views.Detect,name='Detect'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)