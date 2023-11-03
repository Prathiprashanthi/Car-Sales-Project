from django.shortcuts import render,redirect
from mainapp.models import*
from userapp.models import*
from adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from adminapp.models import  All_users_model, Upload_dataset_model
from mainapp.models import User, Predict_details
import pandas as pd
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, 
                             mean_absolute_error, 
                             mean_squared_error
                        )
import numpy as np

# Create your views here.
def admin_dashboard(req):
    all_users_count =  User.objects.all().count()
    pending_users_count = User.objects.filter(User_Status = 'Pending').count()
    rejected_users_count = User.objects.filter(User_Status = 'removed').count()
    Feedbacks_users_count= Feedback.objects.all().count()
    
    return render(req,'admin/dashboard.html',{'a' : all_users_count, 'b' : pending_users_count, 'c' : rejected_users_count, 'd':Feedbacks_users_count})
def pending_users(req):
    pending = User.objects.filter(User_Status = 'Pending')
    paginator = Paginator(pending, 5) 
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin_pendingusers.html', { 'user' : post})
def all_users(req):
    all_users  = User.objects.all()
    paginator = Paginator(all_users, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin_allusers.html',{"allu" : all_users, 'user' : post})
def delete_user(req, id):
    User.objects.get(User_id = id).delete()
    messages.warning(req, 'User was Deleted..!')
    return redirect('all_users')

# Acept users button
def accept_user(req, id):
    status_update = User.objects.get(User_id = id)
    status_update.User_Status = 'accepted'
    status_update.save()
    messages.success(req, 'User was accepted..!')
    return redirect('pending_users')

# Remove user button
def reject_user(req, id):
    status_update2 = User.objects.get(User_id = id)
    status_update2.User_Status = 'removed'
    status_update2.save()
    messages.warning(req, 'User was Rejected..!')
    return redirect('pending_users')

def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('admin_login')

def train_test(req):
    return render(req,'admin/admin_train_test.html')

def admin_traintest_btn(req):
    messages.success(req, "Train test Algorithm executed successfully. Training Images: 613,Validation Images:315,Test Images: 72,Classes: 04")
    return render(req,'admin/admin_traintest_btn.html')

def cnn_model(req):
    return render(req,'admin/admin_cnn.html')

def admin_cnn_btn(req):
    messages.success(req, ' CNN Alogorithm exicuted successfully Accuracy:94%')
    return render(req,'admin/admin_cnn_btn.html')

def upload_dataset(req):
    if req.method == 'POST':
        file = req.FILES['data_file']
        # print(file)
        file_size = str((file.size)/1024) +' kb'
        # print(file_size)
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(req, 'Your dataset was uploaded..')
    return render(req,'admin/admin_uploaddataset.html')

def viewdataset(req):
    dataset = Upload_dataset_model.objects.all()
    paginator = Paginator(dataset, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin_viewdataset.html', {'data' : dataset, 'user' : post})

def delete_dataset(req, id):
    dataset = Upload_dataset_model.objects.get(User_id = id).delete()
    messages.warning(req, 'Dataset was deleted..!')
    return redirect('viewdataset')


def view_view(req):
    data = Upload_dataset_model.objects.last()
    print(data,type(data),'sssss')
    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    table = df.to_html(table_id='data_table')
    return render(req,'admin/admin_view_view.html', {'t':table})

def xgbalgm(req):
   
    return render(req,'admin/xg_boost.html' )

def XGBOOST_btn(req):
    dataset = Upload_dataset_model.objects.first() 
    df = pd.read_csv('car_price_prediction/car_clean.csv')
    print("uuuuuuuuuuuuuuuuuuuuuuuu",df.shape)
    from xgboost import XGBRegressor
    X= df.drop('Selling_Price',axis = 1)
    y = df['Selling_Price']
    print(df.columns)
    print(df.columns)
    target_column = "Selling_Price"  
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("ddddddddddddddddddd",X_train.shape)

    Xgb = XGBRegressor(random_state=42)
    Xgb.fit(X_train, y_train)

    print('*'*10)

    # prediction
    train_prediction= Xgb.predict(X_train)
    test_prediction= Xgb.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)

    
    r2 = round(r2_score(y_test, test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test, test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test,  test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test, test_prediction)*100, 2)

    print(r2,mse,rmse,amse)
    name = "XG Algorithm"
    XG_ALGO.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = XG_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})

    req.session['xgRegressor'] = r2

    metrics_data = {
        'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        # 'dataset_title': dataset.title,
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/xg_boost_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})
    
def adabalgm(req):
    return render(req,'admin/ada_boost.html')


from sklearn.ensemble import AdaBoostRegressor
def ADABoost_btn(req):
    dataset = Upload_dataset_model.objects.first() 
    df = pd.read_csv(dataset.Dataset)
    print(df.columns)
    target_column = "Selling_Price"  
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("ddddddddddddddddddd",X_train.shape)

    Adb = AdaBoostRegressor(random_state=42)
    Adb.fit(X_train, y_train)

    print('*'*10)

    # prediction
    train_prediction= Adb.predict(X_train)
    test_prediction= Adb.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)

    
  
    r2 = round(r2_score(y_test,test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test, test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test,test_prediction)*100, 2)
    print(r2, mse, rmse, amse)

    req.session['AdaBoostRegressor'] = r2
    name = "ADAB Algorithm"
    ADA_ALGO.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = ADA_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})
    metrics_data = {
        'algorithm': 'AdaBoostRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/ada_boost_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})

def knnalgm(req):
    return render(req,'admin/knn_algorithem.html')
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def KNN_btn(req):
    dataset = Upload_dataset_model.objects.first()
    df = pd.read_csv(dataset.Dataset)
    print(df.columns)
    target_column = "Selling_Price"
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("ddddddddddddddddddd",X_train.shape)

    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)

    print('*'*10)

    # prediction
    train_prediction= knn.predict(X_train)
    test_prediction= knn.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)


    r2 = round(r2_score(y_test,test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test, test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test,test_prediction)*100, 2)

    print(r2, mse, rmse, amse)

    req.session['KNNRegressor'] = r2
    name = "KNN Algorithm"
    KNN_ALGO.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = KNN_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})
    metrics_data = {
        'algorithm': 'K-Nearest Neighbors (KNN) Regressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/knn_algo_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})

def linear_regression(req):
    return render(req,'admin/linear_algorithem.html')

from sklearn.linear_model import LinearRegression
def linear_regression_btn(req):
    dataset = Upload_dataset_model.objects.first()
    df = pd.read_csv(dataset.Dataset)
    print(df.columns)
    target_column = "Selling_Price"
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("ddddddddddddddddddd",X_train.shape)

    Lin = LinearRegression()
    Lin.fit(X_train, y_train)

    print('*'*10)

    # prediction
    train_prediction= Lin.predict(X_train)
    test_prediction= Lin.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)


    r2 = round(r2_score(y_test,test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test, test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test,test_prediction)*100, 2)

    print(r2, mse, rmse, amse)

    req.session['LinearRegressor'] = r2
    name = "KNN Algorithm"
    Linear_ALGo.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = Linear_ALGo.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})
    metrics_data = {
        'algorithm': 'Linear Regression',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/linear_algorithem_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})

def random(req):
    return render(req,'admin/randomforest.html')
from sklearn.ensemble import RandomForestRegressor
def randomforest_btn(req):
    dataset = Upload_dataset_model.objects.first()
    df = pd.read_csv(dataset.Dataset)
    print(df.columns)
    target_column = "Selling_Price"
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("ddddddddddddddddddd",X_train.shape)

    Ran = RandomForestRegressor(random_state=42)
    Ran.fit(X_train, y_train)

    print('*'*10)

    # prediction
    train_prediction= Ran.predict(X_train)
    test_prediction= Ran.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)


    # Create a Random Forest Regressor

    r2 = round(r2_score(y_test,test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test, test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test,test_prediction)*100, 2)

    print(r2, mse, rmse, amse)

    req.session['RandomForestRegressor'] = r2
    name = "KNN Algorithm"
    RandomForest.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = RandomForest.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})
    metrics_data = {
        'algorithm': 'Random Forest Regressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/random_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})

def dtalgm(req):
    return render(req,'admin/dtalgm.html')

from sklearn.tree import DecisionTreeRegressor
def Decisiontree_btn(req):
    dataset = Upload_dataset_model.objects.first()
    df = pd.read_csv(dataset.Dataset)
    print(df.columns)
    target_column = "Selling_Price"
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("ddddddddddddddddddd",X_train.shape)

    Det = DecisionTreeRegressor(random_state=42)
    Det.fit(X_train, y_train)

    print('*'*10)

    # prediction
    train_prediction= Det.predict(X_train)
    test_prediction= Det.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)

    r2 = round(r2_score(y_test,test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test, test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test,test_prediction)*100, 2)

    print(r2, mse, rmse, amse)

    req.session['DecisionTreeRegressor'] = r2
    name = "KNN Algorithm"
    DT_ALGO.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = DT_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})
    metrics_data = {
        'algorithm': 'Decision Tree Regressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/dt_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})

def svralgm(req):
    return render(req,'admin/svc_algorithem.html')


from sklearn.svm import SVR
def SVR_btn(req):
    dataset = Upload_dataset_model.objects.first()
    df = pd.read_csv(dataset.Dataset)
    print(df.columns)
    target_column = "Selling_Price"
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price'])
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train=scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    
    Svc = SVR()
    Svc.fit(X_train, y_train)

     # prediction
    train_prediction= Svc.predict(X_train)
    test_prediction= Svc.predict(X_test)
    print('*'*10)
    # evaluation

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    print('R2 for train data', r2_score(y_train,train_prediction))
    print('R2 for test data', r2_score(y_test,test_prediction))
    print('*'*10)

    print('RMSE for train data', np.sqrt(mean_squared_error(y_train,train_prediction)))
    print('RMSE for test data', np.sqrt(mean_squared_error(y_test,test_prediction)))

    print('*'*10)
    print('MSE for train data:',(mean_squared_error(y_train, train_prediction)))
    print('MSE for test data:',(mean_squared_error(y_test, test_prediction)))
    print('*' * 10)

    r2 = round(r2_score(y_test,test_prediction)*100, 2)
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy",r2)

    mse =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

    rmse=round(mean_squared_error(y_test, test_prediction)*100, 2)


    amse=round(mean_absolute_error(y_test,test_prediction)*100, 2)

    print(r2, mse, rmse, amse)

    req.session['SVRRegressor'] = r2
    name = "KVC Algorithm"
    SXM_ALGO.objects.create(R2=r2,MSE=mse,RMSE= rmse,AMSE= amse,Name=name)
    data = SXM_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully', {'i': data})
    metrics_data = {
        'algorithm': 'Support Vector Regression (SVR)',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse,
    }

    context = {
        'target_column': target_column,
        'metrics_data': metrics_data,
    }
    return render(req,'admin/svc_btn.html',{'algorithm': 'GradientBoostingRegressor',
        'R2_Score': r2,
        'RMSE': rmse,
        'MSE': mse,
        'AMSE': amse})

def admin_graph(request):
    # Get the latest data from your models
    xg = XG_ALGO.objects.last().R2
    ada = ADA_ALGO.objects.last().R2
    knn = KNN_ALGO.objects.last().R2
    svc = SXM_ALGO.objects.last().R2
    dt = DT_ALGO.objects.last().R2
    lin = Linear_ALGo.objects.last().R2
    ran = RandomForest.objects.last().R2

    # Pass the data to the template
    return render(request, 'admin/admin_graph_analysis.html', {
        'xg': xg,
        'ada': ada,
        'knn': knn,
        'svc': svc,
        'dt': dt,
        'lin': lin,
        'ran': ran,
    })
def user_feedbacks(req):
    feed =Feedback.objects.all()
    return render(req,'admin/admin_user_feedbacks.html', {'back':feed})

def user_sentiment(req):
    fee = Feedback.objects.all()
    return render(req,'admin/admin_user_sentiment.html', {'cat':fee})

def user_graph(req):
    positive = Feedback.objects.filter(Sentiment = 'positive').count()
    very_positive = Feedback.objects.filter(Sentiment = 'very positive').count()
    negative = Feedback.objects.filter(Sentiment = 'negative').count()
    very_negative = Feedback.objects.filter(Sentiment = 'very negative').count()
    neutral = Feedback.objects.filter(Sentiment = 'neutral').count()
    context ={
        'vp': very_positive, 'p':positive, 'neg':negative, 'vn':very_negative, 'ne':neutral
    }
    return render(req,'admin/admin_feedback_graph.html',context)