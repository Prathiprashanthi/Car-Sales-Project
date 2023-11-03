from django.shortcuts import render,redirect
from  userapp.session import session_required
from mainapp.models import *
from userapp.models import *
from adminapp.models import *
from django.contrib import messages
import time
import pytz
from datetime import datetime
from django.core.paginator import Paginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.files.storage import default_storage
from django.conf import settings
from django.contrib.auth import login
import cv2
import numpy as np
from tkinter import filedialog
from keras.models import model_from_json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm  import SVC
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score, auc, roc_auc_score, roc_curve

# Create your views here.
@session_required
def userdashboard(req):
    images_count =  User.objects.all().count()
    print(images_count)
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    prediction_count =  User.objects.all().count()
    
    if user.Last_Login_Time is None:
        IST = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(IST).time()
        user.Last_Login_Time = current_time_ist
        user.save()
        messages.success(req, 'You are login SUccessfully..')
    return render(req,'user/user_dashboard.html', {'detect' : images_count, 'la' : user,'predictions' : prediction_count,})


@session_required
def profile(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    if req.method == 'POST':
        user_name = req.POST.get('userName')
        user_age = req.POST.get('userAge')
        user_phone = req.POST.get('userphone')
        user_email = req.POST.get('userEmail')
        user_address = req.POST.get("userAddress")
        
        user.Name = user_name
        user.Age = user_age
        user.Address = user_address
        user.Phone = user_phone
        user.Email=user_email
       

        if len(req.FILES) != 0:
            image = req.FILES['profilepic']
            user.Image = image
            user.Name = user_name
            user.Age = user_age
            user.Address = user_address
            user.Phone = user_phone
            user.Email=user_email
            user.Address=user_address
            
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
        else:
            user. Name = user_name
            user.Age = user_age
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
            
    context = {"i":user}
    return render(req,'user/user_profile.html',context)


@session_required
def feedback(req):
    id=req.session["User_id"]
    uusser=User.objects.get(User_id=id)
    if req.method == "POST":
        rating=req.POST.get("rating")
        review=req.POST.get("review")
        # print(sentiment)        
        # print(rating,feed)
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(review)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='very negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment='negative'
        else :
            sentiment='neutral'
        Feedback.objects.create(Rating=rating, Review=review,Sentiment=sentiment, Reviewer=uusser)
        messages.success(req,'Feedback recorded')
        return redirect('feedback')
    return render(req,'user/user_feedback.html')




@session_required
def car_predict(req):
    if req.method == 'POST':
        Age = req.POST.get('field1')
        Present_Price = req.POST.get('field2')
        Kms_Driven = req.POST.get('field3')
        Fuel_Type = req.POST.get('field4')
        Seller_Type = req.POST.get('field5')
        Transmission = req.POST.get('field6')
        Owner = req.POST.get('field7')
        print(Age,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        Age = int(Age)
        Transmission = int(Transmission)
        if Transmission == 0:
            gen = "Manual"
        else:
            gen = "Automatic"
        context = {'gen': gen}
       
        Seller_Type = int(Seller_Type)
        if Seller_Type == 0:
            get= "Dealer"
        else:
            get = "Individual"
        context = {'get': get}
       
        Fuel_Type = int(Fuel_Type)
        if Fuel_Type == 0:
            tet= "CNG"
        elif Fuel_Type==1:
            tet = "Diesel"
        else:
            tet ="Petrol"
        context = {'tet': tet}
        
        Present_Price = float(Present_Price)
       
        import pickle
        file_path = 'car_price_prediction\Xgb_car.pkl'  # Path to the saved model file

        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            # res =loaded_model.predict([[0.95,27000,2,1,0,0,6]])
            
            # print(res,'""""userdatahhhhhhhhhhhhhhhhhhhhhhhhhh""""')
            
            
            result =loaded_model.predict([[Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner,Age]])* 100000
            print(result,'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
           
            dataset = Upload_dataset_model.objects.last()
            
            df=pd.read_csv(dataset.Dataset.path)
          
            X = df.drop('Selling_Price', axis = 1)
            y = df['Selling_Price']
            

            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)

            from xgboost import XGBRegressor
            XGB = XGBRegressor()
            XGB.fit(X_train, y_train)

            # prediction
            train_prediction= XGB.predict(X_train)
            test_prediction= XGB.predict(X_test)
            print('*'*20)

            # evaluation
            from sklearn.metrics import (r2_score, 
                             mean_absolute_error, 
                             mean_squared_error
                        )
            
            
            R2 = round(r2_score(y_test,test_prediction)*100, 2)

            RMSE =round(np.sqrt(mean_squared_error(y_test,test_prediction))*100, 2)
            

            MSE=round(mean_squared_error(y_test, test_prediction)*100, 2)


            AMSE=round(mean_absolute_error(y_test, test_prediction)*100, 2)
          
            
            print(AMSE, MSE,RMSE, R2,'uuuuuuuuuuuuuuuuuuuuuuuuuuu')
            
            context = {'r2_score': R2,'mean_squared_error': MSE,'mean_squared_error_root':RMSE,'mean_absolute_error':AMSE,'res':result}
            
            
            print(type(result), 'ttttttttttttttttttttttttt')
            print(result)
            messages.success(req,'Car Price Predicted SUccessfully')
            return render(req,'user/predict_result.html',context)
        
    return render(req,'user/user_carsaledetection.html')



@session_required
def predict_result(req):
    return render(req,'user/predict_result.html')

# Load the model architecture from the JSON file
with open('car_model_prediction\model\model.json', 'r') as json_file:
    model_json = json_file.read()

# Create the Keras model from the JSON configuration
model = model_from_json(model_json)

# Load the model weights from the H5 file
model.load_weights('car_model_prediction\model\model_weights.h5')

# Define class names
class_names = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012']

# Function to predict car model from an image file
def predict_car_model(image_file_path):
    # Load and preprocess the image
    image = cv2.imread(image_file_path)
    
    if image is not None:
        img = cv2.resize(image, (64, 64))
        img = np.array(img)
        img = img.reshape(1, 64, 64, 3).astype('float32') / 255.0

        # Make a prediction using the loaded model
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]

        return predicted_class_name
    else:
        return "Error: Failed to load the image."

# Provide the path to the image file you want to predict
image_file_path = r'car model\testImages\a1.jpg'  # Replace with your image file path
predicted_model = predict_car_model(image_file_path)

# Display the predicted car model
print(f"Predicted Car Model: {predicted_model}")

# Call the prediction function


@session_required
def car_model(req):
    result = {"message": "No image uploaded"}  # Initialize the result as a dictionary
    uploaded_image_url = None

    if req.method == "POST" and 'img' in req.FILES:
        uploaded_image = req.FILES['img']
        Dataset.objects.create(Image= uploaded_image)
        file_path = default_storage.save(uploaded_image.name, uploaded_image)
        path = settings.MEDIA_ROOT + '/' + file_path
        uploaded_image_url = default_storage.url(file_path)
        result = predict_car_model(path)  # Assuming prediction() returns a dictionary
        req.session['result'] = result
        req.session['uploaded_image_url']=uploaded_image_url
        messages.success(req,'Uploaded image successfully')
        return redirect('model_result')
    
    return render(req,'user/car_model.html', {'result': result, 'uploaded_image_url': uploaded_image_url})

@session_required
def model_result(req):
    result = req.session.get('result', {"message": "No result available"})
    uploaded_image_url = req.session.get('uploaded_image_url', None)
    messages.success(req,'Car Model Predicted SUccessfully')# Provide a default value (None 
    return render(req,'user/model_result.html', {'result': result, 'uploaded_image_url': uploaded_image_url})

def userlogout(request):
    user_id = request.session["User_id"]
    user = User.objects.get(User_id = user_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(request, 'You are logged out..')
    return redirect('user_login')