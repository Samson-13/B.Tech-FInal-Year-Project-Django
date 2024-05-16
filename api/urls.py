from django.urls import path
from .views import *
urlpatterns = [
    path('mymodels/', mymodel_list, name='mymodel-list'),
    path('trainmodel/', train_and_save_model, name='train-model'),
]
