// Portfolio Website JavaScript - Fixed Version

// Project data for modal details
const projectData = {
  1: {
    title: "Intelligent AutoML Platform",
    category: "AutoML System",
    problem: "Data scientists spend 80% of their time on repetitive model selection and tuning tasks instead of solving business problems",
    challenge: "Building a system that can automatically handle different data types, missing values, feature engineering, and model selection while maintaining transparency",
    solution: "Developed a comprehensive AutoML platform using Python and Streamlit with automated preprocessing, multiple algorithm evaluation, and intelligent hyperparameter optimization",
    technologies: ["Python", "Scikit-learn", "Streamlit", "Pandas", "NumPy", "Plotly", "Docker"],
    impact: "Reduced model development time by 75% and achieved 92% accuracy on multiclass classification tasks",
    features: [
      "Automated data preprocessing with intelligent handling of missing values",
      "Smart feature engineering including scaling, encoding, and selection",
      "Multiple algorithm comparison (Random Forest, SVM, Neural Networks)",
      "Hyperparameter optimization using grid search and random search",
      "Model interpretability with feature importance and SHAP values",
      "Real-time results visualization with interactive charts"
    ],
    demo: "https://automl-platform-demo.streamlit.app",
    github: "https://github.com/alex-kumar-dev/intelligent-automl"
  },
  2: {
  "title": "mysh - Custom Unix Shell Implementation",
  "category": "Systems Programming / Operating Systems",
  "problem": "Traditional Unix shells are complex and opaque to many learners. This project aims to demystify the internal workings of command-line interfaces by building a custom shell from scratch.",
  "challenge": "Handling command parsing, I/O redirection, piping, background execution, and integrating system calls like fork, exec, dup, and wait reliably across edge cases.",
  "solution": "Developed a lightweight C-based Unix shell (`mysh`) supporting execution of system commands with features like input/output redirection, piping, background process handling, and custom prompt.",
  "technologies": ["C", "Linux System Calls", "UNIX", "Process Control", "File Descriptors"],
  "impact": "Served as a foundational systems-level project for understanding OS-level process management and terminal behavior. Helped peers debug shell behavior and improved understanding of low-level I/O redirection.",
  "features": [
    "Supports standard command execution via `execvp` and `execlp`",
    "Implements input (`<`) and output (`>`, `>>`) redirection using `dup2` and `open`",
    "Handles background execution using `&` symbol",
    "Supports command piping with `|` via `pipe()` and dual `fork()` execution",
    "Built-in commands: `cd`, `clear`, and `exit`",
    "Custom shell prompt and clean command parsing using `strtok`"
  ],
  "demo": "N/A",
  "github": "https://github.com/your-username/mysh"
},
  3: {
    title: "Hotel Management System",
    category: "Desktop Application",
    problem: "Manual management of hotel operations is error-prone and inefficient",
    challenge: "Integrate secure data storage and user-friendly interface",
    solution: "Developed a desktop application to manage hotel operations including check-ins, check-outs, room availability, and billing using Java and MySQL",
    technologies: ["Java", "MySQL"],
    impact: "Automated booking, room management, and billing leading to reduced human errors and improved operational efficiency",
    features: [
      "Room availability tracking",
      "Customer check-in and check-out",
      "Automated billing and invoicing"
    ],
    "demo": "",
    "github": "https://github.com/AbhishekKanade06/Projects/tree/main/HotelManagment"
  },
  4: {
    "title": "Carbon Emission Forecasting System",
    "category": "Data Analytics",
    "problem": "Need to predict CO2 emissions for countries based on historical data for environmental planning",
    "challenge": "Develop a reliable prediction model and deploy it with an easy-to-use interface",
    "solution": "Designed and implemented a Random Forest regression model to forecast emissions, deployed through a Streamlit web app for global use",
    "technologies": ["Python", "Random Forest", "Streamlit"],
    "impact": "Predicted emissions for 100 countries, facilitating improved environmental policy decisions",
    "features": [
      "Predictive modeling with Random Forest",
      "Interactive Streamlit app interface",
      "Global dataset handling"
    ],
    demo: "https://projects-fuufzehbnggmzltaari9hj.streamlit.app/",
    github: "https://github.com/AbhishekKanade06/Projects/tree/main/Carbon_Emission_prediction"
  },
  5:{
    "title": "Attendance Scanner",
    "category": "Computer Vision, Automation",
    "problem": "Manual attendance recording from sheets is slow and error-prone",
    "challenge": "Develop high-accuracy OCR system with a graphical user interface for ease of use",
    "solution": "Built an OCR-based sheet scanning system using PaddleOCR and OpenCV, with a Tkinter GUI for exporting attendance to Excel files",
    "technologies": ["Python", "OpenCV", "PaddleOCR", "Tkinter"],
    "impact": "Achieved approximately 90% character recognition accuracy, greatly simplifying attendance recording processes",
    "features": [
      "Sheet image processing",
      "Character recognition using PaddleOCR",
      "Excel export functionality"
    ],
    "demo": "",
    "github": ""
  }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('Initializing portfolio website...');
  
  // Initialize all components
  initNavigation();
  initProjectModal();
  initSkillsAnimation();
  initContactForm();
  initScrollAnimations();
  initHeaderScroll();
  
  console.log('Portfolio website initialized successfully');
});

// Navigation functionality
function initNavigation() {
  const navToggle = document.getElementById('nav-toggle');
  const navMenu = document.getElementById('nav-menu');
  const navLinks = document.querySelectorAll('.nav-link');
  
  if (!navToggle || !navMenu) {
    console.error('Navigation elements not found');
    return;
  }

  // Mobile menu toggle
  navToggle.addEventListener('click', function(e) {
    e.preventDefault();
    navMenu.classList.toggle('active');
    navToggle.classList.toggle('active');
  });

  // Smooth scrolling for navigation links
  navLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      const targetSection = document.querySelector(targetId);
      
      if (targetSection) {
        const headerHeight = 80;
        const targetPosition = targetSection.offsetTop - headerHeight;
        
        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });
        
        // Close mobile menu if open
        navMenu.classList.remove('active');
        navToggle.classList.remove('active');
        
        // Update active state
        navLinks.forEach(function(navLink) {
          navLink.classList.remove('active');
        });
        this.classList.add('active');
      }
    });
  });

  // Update active navigation link on scroll
  window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section[id]');
    const scrollY = window.pageYOffset;
    
    sections.forEach(function(section) {
      const sectionHeight = section.offsetHeight;
      const sectionTop = section.offsetTop - 100;
      const sectionId = section.getAttribute('id');
      const navLink = document.querySelector('.nav-link[href="#' + sectionId + '"]');
      
      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        navLinks.forEach(function(link) {
          link.classList.remove('active');
        });
        if (navLink) {
          navLink.classList.add('active');
        }
      }
    });
  });
}

// Project modal functionality
function initProjectModal() {
  const projectModal = document.getElementById('project-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalBody = document.getElementById('modal-body');
  const modalClose = document.querySelector('.modal-close');
  const modalOverlay = document.querySelector('.modal-overlay');
  
  if (!projectModal || !modalTitle || !modalBody) {
    console.error('Modal elements not found');
    return;
  }

  // Add click listeners to project detail buttons
  const detailButtons = document.querySelectorAll('.project-details-btn');
  detailButtons.forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      const projectCard = this.closest('.project-card');
      if (projectCard) {
        const projectId = projectCard.getAttribute('data-project');
        showProjectModal(projectId);
      }
    });
  });

  // Close modal functionality
  if (modalClose) {
    modalClose.addEventListener('click', function(e) {
      e.preventDefault();
      closeModal();
    });
  }
  
  if (modalOverlay) {
    modalOverlay.addEventListener('click', function(e) {
      closeModal();
    });
  }
  
  // Close modal on Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && !projectModal.classList.contains('hidden')) {
      closeModal();
    }
  });

  function showProjectModal(projectId) {
    const project = projectData[projectId];
    if (!project) {
      console.error('Project not found:', projectId);
      return;
    }

    modalTitle.textContent = project.title;
    
    modalBody.innerHTML = 
      '<div class="project-modal-content">' +
        '<div class="modal-section">' +
          '<div class="project-category">' + project.category + '</div>' +
          '<h4>Problem Statement</h4>' +
          '<p>' + project.problem + '</p>' +
        '</div>' +
        
        '<div class="modal-section">' +
          '<h4>Technical Challenge</h4>' +
          '<p>' + project.challenge + '</p>' +
        '</div>' +
        
        '<div class="modal-section">' +
          '<h4>Solution Approach</h4>' +
          '<p>' + project.solution + '</p>' +
        '</div>' +
        
        '<div class="modal-section">' +
          '<h4>Technologies Used</h4>' +
          '<div class="project-tech">' +
            project.technologies.map(function(tech) {
              return '<span class="tech-tag">' + tech + '</span>';
            }).join('') +
          '</div>' +
        '</div>' +
        
        '<div class="modal-section">' +
          '<h4>Key Features</h4>' +
          '<ul class="feature-list">' +
            project.features.map(function(feature) {
              return '<li>' + feature + '</li>';
            }).join('') +
          '</ul>' +
        '</div>' +
        
        '<div class="modal-section">' +
          '<h4>Measurable Impact</h4>' +
          '<div class="impact-highlight">' +
            '<strong>' + project.impact + '</strong>' +
          '</div>' +
        '</div>' +
        
        '<div class="modal-section">' +
          '<h4>Links</h4>' +
          '<div class="project-links">' +
            '<a href="' + project.demo + '" target="_blank" class="btn btn--primary">Live Demo</a>' +
            '<a href="' + project.github + '" target="_blank" class="btn btn--outline">View Source Code</a>' +
          '</div>' +
        '</div>' +
      '</div>';
    
    projectModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    projectModal.classList.add('hidden');
    document.body.style.overflow = 'auto';
  }
}

// Skills animation
function initSkillsAnimation() {
  const skillsSection = document.getElementById('skills');
  const skillBars = document.querySelectorAll('.skill-progress');
  
  if (!skillsSection || skillBars.length === 0) {
    console.error('Skills elements not found');
    return;
  }
  
  let skillsAnimated = false;

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting && !skillsAnimated) {
        animateSkillBars();
        skillsAnimated = true;
      }
    });
  }, { threshold: 0.5 });

  observer.observe(skillsSection);

  function animateSkillBars() {
    skillBars.forEach(function(bar) {
      const width = bar.getAttribute('data-width');
      setTimeout(function() {
        bar.style.width = width + '%';
      }, 200);
    });
  }
}

// Contact form handling
function initContactForm() {
  const contactForm = document.getElementById('contact-form');
  
  if (!contactForm) {
    console.error('Contact form not found');
    return;
  }
  
  // Ensure form inputs work properly
  const inputs = contactForm.querySelectorAll('input, textarea');
  inputs.forEach(function(input) {
    input.addEventListener('input', function() {
      // Remove any error styling
      this.classList.remove('error');
    });
  });
  
  contactForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form data
    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const subject = document.getElementById('subject').value.trim();
    const message = document.getElementById('message').value.trim();
    
    // Basic validation
    let isValid = true;
    
    if (!name) {
      document.getElementById('name').classList.add('error');
      isValid = false;
    }
    
    if (!email || !isValidEmail(email)) {
      document.getElementById('email').classList.add('error');
      isValid = false;
    }
    
    if (!subject) {
      document.getElementById('subject').classList.add('error');
      isValid = false;
    }
    
    if (!message) {
      document.getElementById('message').classList.add('error');
      isValid = false;
    }
    
    if (!isValid) {
      showNotification('Please fill in all fields correctly.', 'error');
      return;
    }
    
    // Add loading state
    const submitBtn = contactForm.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Sending...';
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    
    // Simulate form submission (replace with actual API call)
    setTimeout(function() {
      // Success simulation
      showNotification('Message sent successfully! I\'ll get back to you soon.', 'success');
      contactForm.reset();
      
      // Reset button
      submitBtn.textContent = originalText;
      submitBtn.classList.remove('loading');
      submitBtn.disabled = false;
    }, 2000);
  });
  
  function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }
}

// Notification system
function showNotification(message, type) {
  type = type || 'info';
  
  const notification = document.createElement('div');
  notification.className = 'notification notification--' + type;
  notification.innerHTML = 
    '<div class="notification-content">' +
      '<span>' + message + '</span>' +
      '<button class="notification-close">&times;</button>' +
    '</div>';
  
  document.body.appendChild(notification);
  
  // Show notification
  setTimeout(function() {
    notification.classList.add('show');
  }, 100);
  
  // Auto remove after 5 seconds
  const autoRemove = setTimeout(function() {
    removeNotification(notification);
  }, 5000);
  
  // Manual close
  const closeBtn = notification.querySelector('.notification-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', function() {
      clearTimeout(autoRemove);
      removeNotification(notification);
    });
  }
  
  function removeNotification(notificationElement) {
    notificationElement.classList.remove('show');
    setTimeout(function() {
      if (notificationElement.parentNode) {
        notificationElement.parentNode.removeChild(notificationElement);
      }
    }, 300);
  }
}

// Intersection Observer for animations
function initScrollAnimations() {
  const animatedElements = document.querySelectorAll('.project-card, .skill-category, .timeline-item, .about-content');
  
  if (animatedElements.length === 0) return;
  
  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  
  animatedElements.forEach(function(el) {
    observer.observe(el);
  });
}

// Header scroll effect
function initHeaderScroll() {
  const header = document.querySelector('.header');
  
  if (!header) return;
  
  window.addEventListener('scroll', function() {
    if (window.scrollY > 100) {
      header.style.background = 'rgba(255, 255, 253, 0.95)';
      header.style.backdropFilter = 'blur(10px)';
    } else {
      header.style.background = 'rgba(255, 255, 253, 0.95)';
      header.style.backdropFilter = 'blur(10px)';
    }
  });
}

// Add notification styles dynamically
const notificationStyles = 
  '.notification {' +
    'position: fixed;' +
    'top: 20px;' +
    'right: 20px;' +
    'z-index: 3000;' +
    'max-width: 400px;' +
    'background: var(--color-surface);' +
    'border: 1px solid var(--color-border);' +
    'border-radius: var(--radius-base);' +
    'box-shadow: var(--shadow-lg);' +
    'transform: translateX(100%);' +
    'transition: transform var(--duration-normal) var(--ease-standard);' +
  '}' +
  '.notification.show {' +
    'transform: translateX(0);' +
  '}' +
  '.notification--success {' +
    'border-color: var(--color-success);' +
    'background: rgba(var(--color-success-rgb), 0.1);' +
  '}' +
  '.notification--error {' +
    'border-color: var(--color-error);' +
    'background: rgba(var(--color-error-rgb), 0.1);' +
  '}' +
  '.notification-content {' +
    'display: flex;' +
    'align-items: center;' +
    'justify-content: space-between;' +
    'padding: var(--space-16);' +
  '}' +
  '.notification-close {' +
    'background: none;' +
    'border: none;' +
    'font-size: var(--font-size-lg);' +
    'cursor: pointer;' +
    'color: var(--color-text-secondary);' +
    'margin-left: var(--space-12);' +
  '}' +
  '.modal-section {' +
    'margin-bottom: var(--space-20);' +
  '}' +
  '.modal-section h4 {' +
    'font-size: var(--font-size-lg);' +
    'margin-bottom: var(--space-8);' +
    'color: var(--color-text);' +
    'border-bottom: 1px solid var(--color-border);' +
    'padding-bottom: var(--space-4);' +
  '}' +
  '.modal-section p {' +
    'color: var(--color-text-secondary);' +
    'line-height: 1.6;' +
    'margin-bottom: var(--space-12);' +
  '}' +
  '.feature-list {' +
    'list-style: none;' +
    'padding: 0;' +
    'margin: 0;' +
  '}' +
  '.feature-list li {' +
    'position: relative;' +
    'padding-left: var(--space-20);' +
    'margin-bottom: var(--space-8);' +
    'color: var(--color-text-secondary);' +
    'line-height: 1.5;' +
  '}' +
  '.feature-list li::before {' +
    'content: "✓";' +
    'position: absolute;' +
    'left: 0;' +
    'color: var(--color-success);' +
    'font-weight: var(--font-weight-bold);' +
  '}' +
  '.impact-highlight {' +
    'background: var(--color-bg-3);' +
    'padding: var(--space-12);' +
    'border-radius: var(--radius-base);' +
    'border-left: 4px solid var(--color-success);' +
  '}' +
  '.project-links {' +
    'display: flex;' +
    'gap: var(--space-12);' +
    'flex-wrap: wrap;' +
  '}' +
  'input.error, textarea.error {' +
    'border-color: var(--color-error) !important;' +
    'background-color: rgba(var(--color-error-rgb), 0.1);' +
  '}';

// Inject notification styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);