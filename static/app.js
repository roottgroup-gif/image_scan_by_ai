// Modern Application Core
let currentAnalysisResult = null;
let currentImageUrl = null;

// Light Mode Only Application

// Enhanced Mobile Menu Management with Responsive Features
function initializeMobileMenu() {
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const mobileMenu = document.getElementById('mobile-menu');
    let isAnimating = false;
    
    if (mobileMenuToggle && mobileMenu) {
        // Add touch-friendly event handling
        const toggleMenu = (e) => {
            e.stopPropagation();
            e.preventDefault();
            
            if (isAnimating) return;
            isAnimating = true;
            
            const wasHidden = mobileMenu.classList.contains('hidden');
            
            if (wasHidden) {
                // Opening menu
                mobileMenu.classList.remove('hidden');
                mobileMenu.style.opacity = '0';
                mobileMenu.style.transform = 'translateY(-10px)';
                
                // Animate in
                requestAnimationFrame(() => {
                    mobileMenu.style.transition = 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)';
                    mobileMenu.style.opacity = '1';
                    mobileMenu.style.transform = 'translateY(0)';
                });
                
                // Prevent body scroll on mobile
                if (window.innerWidth < 768) {
                    document.body.style.overflow = 'hidden';
                }
            } else {
                // Closing menu
                mobileMenu.style.transition = 'all 0.15s ease-out';
                mobileMenu.style.opacity = '0';
                mobileMenu.style.transform = 'translateY(-10px)';
                
                setTimeout(() => {
                    mobileMenu.classList.add('hidden');
                    document.body.style.overflow = 'unset';
                }, 150);
            }
            
            // Update icon with animation
            const icon = mobileMenuToggle.querySelector('i');
            if (icon) {
                const newIcon = wasHidden ? 'x' : 'menu';
                icon.style.transform = 'rotate(180deg)';
                
                setTimeout(() => {
                    icon.setAttribute('data-lucide', newIcon);
                    lucide.createIcons();
                    icon.style.transform = 'rotate(0deg)';
                }, 100);
            }
            
            setTimeout(() => {
                isAnimating = false;
            }, 200);
        };
        
        // Support both click and touch events
        mobileMenuToggle.addEventListener('click', toggleMenu);
        mobileMenuToggle.addEventListener('touchend', (e) => {
            e.preventDefault();
            toggleMenu(e);
        });
        
        // Close mobile menu when clicking/touching outside
        const closeMenu = () => {
            if (!mobileMenu.classList.contains('hidden') && !isAnimating) {
                mobileMenu.style.transition = 'all 0.15s ease-out';
                mobileMenu.style.opacity = '0';
                mobileMenu.style.transform = 'translateY(-10px)';
                
                setTimeout(() => {
                    mobileMenu.classList.add('hidden');
                    document.body.style.overflow = 'unset';
                }, 150);
                
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.setAttribute('data-lucide', 'menu');
                    lucide.createIcons();
                }
            }
        };
        
        document.addEventListener('click', (e) => {
            if (!mobileMenuToggle.contains(e.target) && !mobileMenu.contains(e.target)) {
                closeMenu();
            }
        });
        
        document.addEventListener('touchend', (e) => {
            if (!mobileMenuToggle.contains(e.target) && !mobileMenu.contains(e.target)) {
                closeMenu();
            }
        });
        
        // Close mobile menu when clicking on a link with animation
        const mobileLinks = mobileMenu.querySelectorAll('a');
        mobileLinks.forEach(link => {
            const handleLinkClick = () => {
                if (!isAnimating) {
                    // Add visual feedback
                    link.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
                    
                    setTimeout(() => {
                        link.style.backgroundColor = '';
                        closeMenu();
                    }, 150);
                }
            };
            
            link.addEventListener('click', handleLinkClick);
            link.addEventListener('touchend', (e) => {
                e.preventDefault();
                handleLinkClick();
                // Trigger navigation after animation
                setTimeout(() => {
                    window.location.href = link.href;
                }, 300);
            });
        });
        
        // Handle window resize to close menu on desktop
        window.addEventListener('resize', () => {
            if (window.innerWidth >= 768 && !mobileMenu.classList.contains('hidden')) {
                closeMenu();
            }
        });
        
        // Handle escape key to close menu
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !mobileMenu.classList.contains('hidden')) {
                closeMenu();
            }
        });
        
        // Swipe gesture support for mobile
        let touchStartY = 0;
        let touchStartX = 0;
        
        mobileMenu.addEventListener('touchstart', (e) => {
            touchStartY = e.touches[0].clientY;
            touchStartX = e.touches[0].clientX;
        });
        
        mobileMenu.addEventListener('touchmove', (e) => {
            const touchY = e.touches[0].clientY;
            const touchX = e.touches[0].clientX;
            const deltaY = touchY - touchStartY;
            const deltaX = Math.abs(touchX - touchStartX);
            
            // Close menu on upward swipe (and minimal horizontal movement)
            if (deltaY < -50 && deltaX < 30) {
                e.preventDefault();
                closeMenu();
            }
        });
    }
}

// Enhanced Toast Notification System with Responsive Positioning
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        
        // Responsive positioning
        const isMobile = window.innerWidth < 640;
        if (isMobile) {
            toastContainer.className = 'fixed top-20 left-4 right-4 z-50 space-y-3';
        } else {
            toastContainer.className = 'fixed top-6 right-6 z-50 space-y-3 max-w-sm';
        }
        
        document.body.appendChild(toastContainer);
    }
    
    const toast = document.createElement('div');
    toast.className = 'card p-4 transform transition-all duration-300 ease-in-out translate-x-full opacity-0';
    
    const typeConfig = {
        success: {
            icon: 'check-circle',
            classes: 'border-green-200 bg-green-50',
            iconColor: 'text-green-600'
        },
        error: {
            icon: 'x-circle',
            classes: 'border-red-200 bg-red-50',
            iconColor: 'text-red-600'
        },
        warning: {
            icon: 'alert-triangle',
            classes: 'border-yellow-200 bg-yellow-50',
            iconColor: 'text-yellow-600'
        },
        info: {
            icon: 'info',
            classes: 'border-blue-200 bg-blue-50',
            iconColor: 'text-blue-600'
        }
    };
    
    const config = typeConfig[type] || typeConfig.info;
    toast.className += ` ${config.classes}`;
    
    // Create toast content
    const container = document.createElement('div');
    container.className = 'flex items-start gap-3';
    
    const iconDiv = document.createElement('div');
    iconDiv.className = `flex-shrink-0 ${config.iconColor}`;
    
    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', config.icon);
    icon.className = 'w-5 h-5';
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex-1';
    
    const messageText = document.createElement('p');
    messageText.className = 'text-sm font-medium text-gray-900';
    messageText.textContent = message;
    
    const closeButton = document.createElement('button');
    closeButton.className = 'flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors';
    closeButton.innerHTML = '<i data-lucide="x" class="w-4 h-4"></i>';
    
    // Assemble toast
    iconDiv.appendChild(icon);
    messageDiv.appendChild(messageText);
    container.appendChild(iconDiv);
    container.appendChild(messageDiv);
    container.appendChild(closeButton);
    toast.appendChild(container);
    
    // Add to container
    toastContainer.appendChild(toast);
    
    // Initialize icons
    lucide.createIcons();
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-x-full', 'opacity-0');
        toast.classList.add('translate-x-0', 'opacity-100');
    }, 10);
    
    // Auto remove and close button functionality
    const removeToast = () => {
        toast.classList.add('translate-x-full', 'opacity-0');
        setTimeout(() => {
            if (toastContainer.contains(toast)) {
                toastContainer.removeChild(toast);
            }
        }, 300);
    };
    
    closeButton.addEventListener('click', removeToast);
    
    // Auto remove after 5 seconds
    setTimeout(removeToast, 5000);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getIndicatorIcon(strength) {
    switch (strength.toLowerCase()) {
        case 'strong': return 'alert-triangle';
        case 'moderate': return 'alert-circle';
        case 'weak': return 'info';
        default: return 'help-circle';
    }
}

// Enhanced Local Storage Management
function saveAnalysisToLocalStorage(analysis, imageData) {
    try {
        const analysisHistory = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
        const newAnalysis = {
            ...analysis,
            timestamp: new Date().toISOString(),
            imageData: imageData,
            id: Date.now().toString()
        };
        
        analysisHistory.unshift(newAnalysis);
        
        // Keep only last 20 analyses
        if (analysisHistory.length > 20) {
            analysisHistory.splice(20);
        }
        
        localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));
    } catch (error) {
        console.error('Error saving to local storage:', error);
        showToast('Failed to save analysis history', 'warning');
    }
}

function loadAnalysisHistory() {
    try {
        return JSON.parse(localStorage.getItem('analysisHistory') || '[]');
    } catch (error) {
        console.error('Error loading from local storage:', error);
        return [];
    }
}

function clearAnalysisHistory() {
    try {
        localStorage.removeItem('analysisHistory');
        showToast('Analysis history cleared', 'info');
    } catch (error) {
        console.error('Error clearing local storage:', error);
        showToast('Failed to clear history', 'error');
    }
}

// Enhanced Animation Utilities
function animateElement(element, animationClass, duration = 600) {
    return new Promise((resolve) => {
        element.classList.add(animationClass);
        
        setTimeout(() => {
            element.classList.remove(animationClass);
            resolve();
        }, duration);
    });
}

function fadeInElement(element, delay = 0) {
    element.style.opacity = '0';
    element.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        element.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
    }, delay);
}

function fadeOutElement(element) {
    return new Promise((resolve) => {
        element.style.transition = 'all 0.3s ease-out';
        element.style.opacity = '0';
        element.style.transform = 'translateY(-10px)';
        
        setTimeout(() => {
            resolve();
        }, 300);
    });
}

// Keyboard Navigation Support
function initializeKeyboardNavigation() {
    document.addEventListener('keydown', (e) => {
        // Escape key to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('[id*="modal"]:not(.hidden)');
            modals.forEach(modal => {
                modal.classList.add('hidden');
                document.body.style.overflow = 'unset';
            });
        }
        
        
        // Alt + U for file upload (when on home page)
        if (e.altKey && e.key === 'u' && window.location.pathname === '/') {
            e.preventDefault();
            const fileInput = document.getElementById('file-input');
            if (fileInput) {
                fileInput.click();
            }
        }
    });
}

// Performance Monitoring
function measurePerformance(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    console.log(`${name} took ${(end - start).toFixed(2)} milliseconds`);
    return result;
}

// Enhanced Responsive Utilities
function initializeResponsiveFeatures() {
    // Detect touch device
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (isTouchDevice) {
        document.body.classList.add('touch-device');
    }
    
    // Handle orientation changes
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            // Recalculate viewport height for mobile browsers
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${vh}px`);
            
            // Close mobile menu on orientation change
            const mobileMenu = document.getElementById('mobile-menu');
            if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
                document.body.style.overflow = 'unset';
            }
        }, 100);
    });
    
    // Set initial viewport height
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
    
    // Handle window resize for responsive adjustments
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            // Update viewport height
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${vh}px`);
            
            // Reposition toast container if needed
            const toastContainer = document.getElementById('toast-container');
            if (toastContainer) {
                const isMobile = window.innerWidth < 640;
                if (isMobile) {
                    toastContainer.className = 'fixed top-20 left-4 right-4 z-50 space-y-3';
                } else {
                    toastContainer.className = 'fixed top-6 right-6 z-50 space-y-3 max-w-sm';
                }
            }
        }, 250);
    });
}


// Initialize Application with Enhanced Responsive Features
document.addEventListener('DOMContentLoaded', () => {
    // Core initialization
    initializeMobileMenu();
    initializeKeyboardNavigation();
    initializeResponsiveFeatures();
    
    // Initialize Lucide icons
    lucide.createIcons();
    
    // Add smooth scrolling behavior (disabled on mobile for performance)
    if (window.innerWidth >= 768) {
        document.documentElement.style.scrollBehavior = 'smooth';
    }
    
    // Add loading state management
    window.addEventListener('beforeunload', () => {
        document.body.style.opacity = '0.7';
    });
    
    // Show welcome message on first visit (delayed on mobile)
    if (!localStorage.getItem('hasVisited')) {
        const delay = window.innerWidth < 768 ? 1500 : 1000;
        setTimeout(() => {
            showToast('Welcome to AI Image Detection! Upload an image to get started.', 'info');
            localStorage.setItem('hasVisited', 'true');
        }, delay);
    }
    
    // Add focus management for better accessibility
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });
    
    document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-navigation');
    });
    
    console.log('🚀 Modern AI Detection app initialized successfully');
});