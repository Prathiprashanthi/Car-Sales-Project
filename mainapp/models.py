from django.db import models

# Create your models here.
class User(models.Model):
    User_id = models.AutoField(primary_key=True)
    Name = models.CharField(max_length=100)
    Age = models.IntegerField(null=True)
    Email = models.EmailField(max_length=100,null=True)
    Password = models.TextField(max_length=100,null=True)
    Address = models.TextField(max_length=100,null=True)
    Phone = models.TextField(max_length=10,null=True)
    Image = models.FileField(upload_to='images/', null=True)
    Date_Time = models.DateTimeField(auto_now = True, null = True)
    User_Status = models.TextField(default = 'pending', max_length=50, null = True)
    Otp_Num = models.IntegerField(null = True)
    Otp_Status = models.TextField(default = 'pending', max_length = 60, null = True)
    Last_Login_Time = models.TimeField(null = True)
    Last_Login_Date = models.DateField(auto_now_add=True,null = True)
    No_Of_Times_Login = models.IntegerField(default = 0, null = True)
    Message =models.TextField(max_length=250,null=True)
    class Meta:
        db_table='user'
        
class Predict_details(models.Model):
    predict_id = models.AutoField(primary_key=True)
    Field_1 = models.CharField(max_length = 60, null = True)
    Field_2 = models.CharField(max_length = 60, null = True)
    Field_3 = models.CharField(max_length = 60, null = True)
    Field_4 = models.CharField(max_length = 60, null = True)
    Field_5 = models.CharField(max_length = 60, null = True)
    Field_6 = models.CharField(max_length = 60, null = True)
    Field_7 = models.CharField(max_length = 60, null = True)
    Field_8 = models.CharField(max_length = 60, null = True)
    Field_9 = models.CharField(max_length = 60, null = True)
    Field_10 = models.CharField(max_length = 60, null = True)
    
    class Meta:
        db_table = "predict_detail"
        
class Last_login(models.Model):
    Id = models.AutoField(primary_key = True)
    Login_Time = models.DateTimeField(auto_now = True, null = True)

    class Meta:
        db_table = "last_login"
