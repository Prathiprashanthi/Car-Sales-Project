from functools import wraps
from django.shortcuts import redirect

def session_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        username = request.session.get('User_id')
        if username is not None:
            return view_func(request, *args, **kwargs)
        else:
            return redirect('user_login') 
    return _wrapped_view