/*
document.addEventListener('DOMContentLoaded', function() {
    const token = localStorage.getItem('authToken');
    
    // If we're on login or register page but already authenticated, redirect to home
    if (token) {
      const currentPath = window.location.pathname;
      if (currentPath === '/login' || currentPath === '/register') {
        window.location.href = '/home';
      }
    }
    
    // If we're on home or chatbot but not authenticated, redirect to login
    if (!token) {
      const currentPath = window.location.pathname;
      if (currentPath === '/home' || currentPath === '/chatbot') {
        window.location.href = '/login';
      }
    }
  });


function registerUser(event){
    event.preventDefault();
    console.log("registration button clicked");

    const form = event.target;
    const formData = new URLSearchParams(new FormData(form))
    console.log("Form Data:", formData);

    fetch('/register', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("response data: ",data);
        if(data.redirect){
            window.location.href = data.redirect;
        } else {
            console.error("Registration error:", data.error);
            const errorDiv = document.getElementById("registration-error");
            if (errorDiv){
                errorDiv.textContent = data.error;
            }
        }
    })
    .catch(error=> {
        console.error("network error in registration:", error);
        const errorDiv = document.getElementById("registration-error");
        if(errorDiv){
            errorDiv.textContent = "An error occured try again later. ";
        }
    });
}

function loginUser(event){
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const data = {
        email: formData.get('email'),
        password: formData.get('password')
    };
    console.log("Form data : ", formData);

    fetch ('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        console.log("response data: ", data);
        if(data.token) {
            localStorage.setItem('authToken', data.token);
            localStorage.setItem('email', data.email);
            window.location.href = data.redirect;
            updateNavigation();
        } else {
            console.error("Login error:", data.error);
            const errorDiv = document.getElementById("login-error");
            if (errorDiv) {
                errorDiv.textContent = data.error || "An unexpected error occured.";
            }

        }
    })
    .catch(error => {
        console.error("network error during login:", error);
        const errorDiv = document.getElementById("login-error");
        if(errorDiv) {
            errorDiv.textContent = "An error occured try again later. ";
        }
    });
}

window.addEventListener('navigationLoaded', () => {
    updateNavigation();
});

const registerForm = document.getElementById("registerForm");
if(registerForm){
    registerForm.addEventListener('submit',registerUser);
}

const loginForm = document.getElementById("loginForm");
if(loginForm){
    loginForm.addEventListener('submit',loginUser);
}
    */
