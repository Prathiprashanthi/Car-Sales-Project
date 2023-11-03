from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'
        
class All_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'all_users'

# Dataset
class Upload_dataset_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    Dataset = models.FileField(null=True)
    File_size = models.CharField(max_length = 100) 
    Date_Time = models.DateTimeField(auto_now = True)
    
    class Meta:
        db_table = 'upload_dataset'




class Linear_ALGo(models.Model):
    Logistic_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'Logistic'
        
class RandomForest(models.Model):
    Random_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'RandomForest'


# XG-Boost Algo
class XG_ALGO(models.Model):
    XG_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'XG_algo'

# ADA Boost Algo
class ADA_ALGO(models.Model):
    XG_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'ADA_algo'

# KNN Algo
class KNN_ALGO(models.Model):
    XG_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'KNN_algo'

# SXM Algo
class SXM_ALGO(models.Model):
    SXM_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'SXM_algo'

# DECISION TREE Algo
class DT_ALGO(models.Model):
    DT_ID = models.AutoField(primary_key = True)
    R2 = models.TextField(max_length = 100)
    MSE = models.TextField(max_length = 100) 
    RMSE = models.TextField(max_length = 100)
    AMSE = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'DT_algo'

# dataset
class DATASET(models.Model):
    DS_ID = models.AutoField(primary_key = True)
    Age = models.IntegerField()
    Present_Price = models.TextField() 
    Kms_Driven = models.IntegerField()
    Fuel_Type = models.TextField()
    Seller_Type = models.TextField()
    Transmission = models.TextField()
    Owner = models.IntegerField() 
    

    
    class Meta:
        db_table = 'Dataset'
        




