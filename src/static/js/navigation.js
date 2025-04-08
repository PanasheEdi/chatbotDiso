// navigation.js
document.addEventListener('DOMContentLoaded', () => {
  fetch('/templates/navigationBar.html')
    .then(response => response.text())
    .then(data => {
      document.getElementById('navbar-container').innerHTML = data;
      setupMobileMenu();
      updateNavigationState();
      window.dispatchEvent(new Event('navigationLoaded'));
    })
    .catch(error => {
      console.error("Error loading navigation:", error);
    });
});

function setupMobileMenu() {
  const menuToggle = document.getElementById('mobile-menu');
  const navMenu = document.querySelector('.nav-menu');

  if (menuToggle && navMenu) {
    menuToggle.addEventListener('click', () => {
      navMenu.classList.toggle('active');
      menuToggle.classList.toggle('active');
    });
  }
}

function updateNavigationState() {

    const noAuthElements = document.querySelectorAll('.no-auth');

    noAuthElements.forEach(el => el.style.display = 'block');
}

document.addEventListener('click', function(e){
  if(e.target && e.target.id === 'logout-link') {
    e.preventDefault();
    window.location.href = '/login';
  }
});