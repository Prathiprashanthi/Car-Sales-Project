"""
URL configuration for car_sales_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mainapp import views as main_views
from userapp import views as user_views
from adminapp import views as admin_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    #main
    path('admin/', admin.site.urls),
    path('',main_views.home,name='home'),
    path('user_login',main_views.user_login,name='user_login'),
    path('admin_login',main_views.admin_login,name='admin_login'),
    path('about_us',main_views.about_us,name='about_us'),
    path('contact_us',main_views.contact_us,name='contact_us'),
    path('register',main_views.register,name='register'),
    path('otp',main_views.otp,name='otp'),
   path('forgotpassword', main_views.forgotpassword, name='forgotpassword'),
    
    #user
    path('car_predict', user_views.car_predict, name = 'car_predict' ),
    path('car_model',user_views.car_model,name='car_model'),
    path('userdashboard',user_views.userdashboard,name ='userdashboard' ),
    path('profile', user_views.profile,name = 'profile' ),
    path('feedback',user_views.feedback,name='feedback'),
    path('predict_result', user_views.predict_result, name = 'predict_result'),
    path('model_result', user_views.model_result, name = 'model_result'),
    path('userlogout', user_views.userlogout, name = 'userlogout'),
    
    
    #admin
    path('admin_dashboard',admin_views.admin_dashboard,name='admin_dashboard'),
    path('pending_users',admin_views.pending_users,name='pending_users'),
    path('all_users',admin_views.all_users,name='all_users'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('delete-user/<int:id>', admin_views.delete_user, name = 'delete_user'),
    path('delete-dataset/<int:id>', admin_views.delete_dataset, name = 'delete_dataset'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),
    path('train_test',admin_views.train_test,name='train_test'),
    path('admin_traintest_btn',admin_views.admin_traintest_btn,name='admin_traintest_btn'),
    path('admin_cnn_btn',admin_views.admin_cnn_btn,name='admin_cnn_btn'),
    path('cnn_model',admin_views.cnn_model,name='cnn_model'),
    path('upload_dataset',admin_views.upload_dataset,name="upload_dataset"),
    path('view_dataset', admin_views.viewdataset, name = 'viewdataset'),
    path('view_view', admin_views.view_view, name='view_view'),
    path('xgb-algm', admin_views.xgbalgm, name = 'xgbalgm'),
    path('XGBOOST_btn', admin_views.XGBOOST_btn, name='XGBOOST_btn'),
    path('adab-algm', admin_views.adabalgm, name = 'adabalgm'),
    path('ADABoost_btn', admin_views.ADABoost_btn, name='ADABoost_btn'),
    path('knn-algm', admin_views.knnalgm, name = 'knnalgm'),
    path('KNN_btn', admin_views.KNN_btn, name='KNN_btn'),
    path('linear_regression', admin_views.linear_regression, name = 'linear_regression'),
    path('linear_regression_btn', admin_views.linear_regression_btn, name='linear_regression_btn'),
    path('random', admin_views.random, name = 'random'),
    path('randomforest_btn', admin_views.randomforest_btn, name='randomforest_btn'),
    path('dt-algm', admin_views.dtalgm, name = 'dtalgm'),
    path('Decisiontree_btn', admin_views.Decisiontree_btn, name='Decisiontree_btn'),
    path('svr-alg', admin_views.svralgm, name = 'svralgm'),
    path('SVR_btn', admin_views.SVR_btn, name='SVR_btn'),
    path('admin_graph', admin_views.admin_graph, name = 'admin_graph'),
    path('user_feedbacks',admin_views.user_feedbacks,name='user_feedbacks'),
    path('user_sentiment',admin_views.user_sentiment,name='user_sentiment'),
    path('user_graph',admin_views.user_graph,name='user_graph'),
    
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
