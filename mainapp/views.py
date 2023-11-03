from django.shortcuts import render,redirect
import urllib.request
import urllib.parse
import random 
import ssl
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from mainapp.models import *
from userapp.models import *

# Create your views here.
def sendSMS(user, otp, mobile):
    data = urllib.parse.urlencode({
        'username': 'Codebook',
        'apikey': '56dbbdc9cea86b276f6c',
        'mobile': mobile,
        'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
        'senderid': 'CODEBK'
    })
    data = data.encode('utf-8')
    # Disable SSL certificate verification
    context = ssl._create_unverified_context()
    request = urllib.request.Request("https://smslogin.co/v3/api.php?")
    f = urllib.request.urlopen(request, data,context=context)
    return f.read()
def home(req):
    return render(req,'main/main_home.html')

def user_login(req):
    if req.method == 'POST':
        user_email = req.POST.get('email')
        user_password = req.POST.get('password')
        print( user_email,user_password)
        
        # try:
        user_data = User.objects.get(Email = user_email)
        print(user_data)
        if user_data.Password == user_password:
            if user_data.Otp_Status == 'verified' and user_data.User_Status=='accepted':
                req.session['User_id'] = user_data.User_id
                messages.success(req, 'You are logged in..')
                user_data.No_Of_Times_Login += 1
                user_data.save()
                return redirect('userdashboard')
            elif user_data.Otp_Status == 'verified' and user_data.User_Status=='pending':
                messages.info(req, 'Your Status is in pending')
                return redirect('user_login')
            else:
                messages.warning(req, 'verifyOTP...!')
                req.session['Email'] = user_data.Email
                return redirect('otp')
        else:
            messages.error(req, 'incorrect credentials...!')
            return redirect('user_login')
    return render(req,'main/main_user.html')

def admin_login(req):
    admin_name = 'admin@123gmail.com'
    admin_pwd = 'admin'
    if req.method == 'POST':
        admin_n = req.POST.get('aname')
        admin_p = req.POST.get('apassword')
        if (admin_n == admin_name and admin_p == admin_pwd):
            messages.success(req, 'You are logged in..')
            return redirect('admin_dashboard')
        else:
            messages.error(req, 'You are trying to loging with wrong details..')
            return redirect('admin_login')
    return render(req,'main/main_admin.html')

def about_us(req):
    return render(req,'main/main_about.html')

def contact_us(req):
    if req.method == 'POST':
        name = req.POST.get('Name')
        contact = req.POST.get('PhoneNumber')
        email = req.POST.get('Email')
        message = req.POST.get('Message')
        User.objects.create(Name=name,Phone=contact,Email=email,Message=message)
        messages.success(req, 'Your message has been submitted successfully.')
        return redirect('contact_us') 
    return render(req,'main/main_contact.html')


def register(req):
    if req.method == 'POST':
        fullname = req.POST.get('fullname')
        age = req.POST.get('age')
        email = req.POST.get('email')
        password = req.POST.get('password')
        address = req.POST.get('address')
        phone = req.POST.get('phone')
        image = req.FILES['img']
        number= random.randint(1000,9999)
        try:
            data = User.objects.get(Email =  email)
            messages.warning(req, 'Email was already registered, choose another email..!')
            return redirect('register')
        except:
            sendSMS(fullname,number,phone)
            User.objects.create( Name=fullname,Email=email,Phone=phone,Age=age,Password=password,Address=address,Image=image,Otp_Num = number)
            mail_message = f'Registration Successfully\n Your 4 digit Pin is below\n {number}'
            print(mail_message)
            send_mail("User Password", mail_message , settings.EMAIL_HOST_USER, [email])
            req.session['Email']= email
            messages.success(req, 'Your account was created..')
            return redirect('otp')
    return render(req,'main/main_register.html')
    

def otp(req):
    user_email = req.session['Email']
    user_o = User.objects.get(Email = user_email)
    print(user_o.Otp_Num,'data otp')
    if req.method == 'POST':
        user_otp = req.POST.get('otp')
        u_otp = int(user_otp)
        if u_otp == user_o.Otp_Num:
            user_o.Otp_Status = 'verified'
            user_o.save()
            messages.success(req, 'OTP verification was Success. Now you can continue to login..!')
            return redirect('home')
        else:
            messages.error(req, 'OTP verification was Faild. You entered invalid OTP..!')
            return redirect('otp')
    return render(req,'main/main_otp.html')

def forgotpassword(req):
    return render(req,'main/main_forgotpwd.html')