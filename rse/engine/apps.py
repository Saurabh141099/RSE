from django.apps import AppConfig
from engine.views import load


class EngineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'engine'
    
    def ready(self):
        load()