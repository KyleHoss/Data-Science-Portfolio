let sections = document.querySelectorAll('section');
let navLinks = document.querySelectorAll('header nav a');

window.onscroll = () => {
    sections.forEach(sec => {
        let top = window.scrollY;
        let offset = sec.offsetTop - 150;
        let height = sec.offsetHeight;
        let id = sec.getAttribute('id');

        if(top >= offset && top < offset + height) {
            navLinks.forEach(links => {
                links.classList.remove('active');
                document.querySelector('header nav a[href*=' + id + ']').classList.add('active');
            });
        };
    });
    let header = document.querySelector('header');
};

// Scroll Reveal
ScrollReveal({
    reset: true,
    distance: '80px',
    duration: 2000, 
    delay: 200
});

ScrollReveal().reveal('.home-content, .heading', { origin: 'top'});
ScrollReveal().reveal('.home-image, .skills-container, .portfolio-box',{ origin: 'bottom'});
ScrollReveal().reveal('.home-content h1', { origin: 'left'});
ScrollReveal().reveal('.home-content p', { origin: 'right'});

// Typing Animation
var typed = new Typed('.multiple-text', {
    strings: ['Computational Scientist', 'Data Scientist', 'Financial Software Engineer','Penetration Tester', 'Systems Engineer'],
    typeSpeed: 100, 
    backSpeed: 100, 
    backDelay: 1000, 
    loop: true
});