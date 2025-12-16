// ===== UI MANAGEMENT AND INTERACTIONS =====

class TravelAgencyUI {
    constructor() {
        this.currentView = 'workspace';
        this.modals = {};
        this.notifications = [];
        this.animations = {};
        
        this.init();
    }
    
    init() {
        this.setupModals();
        this.setupNotifications();
        this.setupAnimations();
        this.bindGlobalEvents();
    }
    
    // ===== MODAL MANAGEMENT =====
    setupModals() {
        this.modals = {
            addClient: document.getElementById('addClientModal')
        };
        
        // Close modals when clicking outside
        Object.values(this.modals).forEach(modal => {
            if (modal) {
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        this.closeModal(modal);
                    }
                });
            }
        });
        
        // Close modals with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    }
    
    openModal(modalName) {
        const modal = this.modals[modalName];
        if (modal) {
            modal.style.display = 'flex';
            this.animateModalOpen(modal);
        }
    }
    
    closeModal(modal) {
        if (modal) {
            this.animateModalClose(modal).then(() => {
                modal.style.display = 'none';
            });
        }
    }
    
    closeAllModals() {
        Object.values(this.modals).forEach(modal => {
            if (modal) {
                this.closeModal(modal);
            }
        });
    }
    
    // ===== NOTIFICATION SYSTEM =====
    setupNotifications() {
        this.createNotificationContainer();
    }
    
    createNotificationContainer() {
        const container = document.createElement('div');
        container.id = 'notificationContainer';
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        const notification = this.createNotificationElement(message, type);
        const container = document.getElementById('notificationContainer');
        
        if (container) {
            container.appendChild(notification);
            
            // Animate in
            setTimeout(() => {
                notification.classList.add('show');
            }, 100);
            
            // Auto remove
            setTimeout(() => {
                this.removeNotification(notification);
            }, duration);
        }
    }
    
    createNotificationElement(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icon = this.getNotificationIcon(type);
        
        notification.innerHTML = `
            <div class="notification-content">
                <i class="${icon}"></i>
                <span class="notification-message">${message}</span>
            </div>
            <button class="notification-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        return notification;
    }
    
    getNotificationIcon(type) {
        const icons = {
            info: 'fas fa-info-circle',
            success: 'fas fa-check-circle',
            warning: 'fas fa-exclamation-triangle',
            error: 'fas fa-times-circle'
        };
        
        return icons[type] || icons.info;
    }
    
    removeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentElement) {
                notification.parentElement.removeChild(notification);
            }
        }, 300);
    }
    
    // ===== ANIMATION SYSTEM =====
    setupAnimations() {
        this.animations = {
            fadeIn: this.createFadeInAnimation(),
            slideIn: this.createSlideInAnimation(),
            scaleIn: this.createScaleInAnimation()
        };
    }
    
    createFadeInAnimation() {
        return [
            { opacity: 0, transform: 'translateY(20px)' },
            { opacity: 1, transform: 'translateY(0)' }
        ];
    }
    
    createSlideInAnimation() {
        return [
            { transform: 'translateX(-100%)' },
            { transform: 'translateX(0)' }
        ];
    }
    
    createScaleInAnimation() {
        return [
            { transform: 'scale(0.8)', opacity: 0 },
            { transform: 'scale(1)', opacity: 1 }
        ];
    }
    
    animateElement(element, animation, duration = 300) {
        return element.animate(animation, {
            duration,
            easing: 'ease-out',
            fill: 'forwards'
        });
    }
    
    // ===== MODAL ANIMATIONS =====
    animateModalOpen(modal) {
        const content = modal.querySelector('.modal-content');
        if (content) {
            content.style.transform = 'scale(0.8)';
            content.style.opacity = '0';
            
            setTimeout(() => {
                content.style.transition = 'all 0.3s ease-out';
                content.style.transform = 'scale(1)';
                content.style.opacity = '1';
            }, 10);
        }
    }
    
    animateModalClose(modal) {
        return new Promise((resolve) => {
            const content = modal.querySelector('.modal-content');
            if (content) {
                content.style.transition = 'all 0.2s ease-in';
                content.style.transform = 'scale(0.8)';
                content.style.opacity = '0';
                
                setTimeout(resolve, 200);
            } else {
                resolve();
            }
        });
    }
    
    // ===== GLOBAL EVENT BINDING =====
    bindGlobalEvents() {
        // Form validation
        this.setupFormValidation();
        
        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Responsive behavior
        this.setupResponsiveBehavior();
    }
    
    setupFormValidation() {
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!this.validateForm(form)) {
                    e.preventDefault();
                }
            });
        });
        
        // Real-time validation
        this.setupRealTimeValidation();
    }
    
    setupRealTimeValidation() {
        const inputs = document.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
            
            input.addEventListener('input', () => {
                this.clearFieldError(input);
            });
        });
    }
    
    validateForm(form) {
        let isValid = true;
        const fields = form.querySelectorAll('input, select, textarea');
        
        fields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    validateField(field) {
        const value = field.value.trim();
        let isValid = true;
        let errorMessage = '';
        
        // Required field validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            errorMessage = 'This field is required';
        }
        
        // Email validation
        if (field.type === 'email' && value && !this.isValidEmail(value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address';
        }
        
        // Number validation
        if (field.type === 'number' && value) {
            const num = parseFloat(value);
            if (isNaN(num)) {
                isValid = false;
                errorMessage = 'Please enter a valid number';
            }
            
            if (field.hasAttribute('min') && num < parseFloat(field.min)) {
                isValid = false;
                errorMessage = `Value must be at least ${field.min}`;
            }
            
            if (field.hasAttribute('max') && num > parseFloat(field.max)) {
                isValid = false;
                errorMessage = `Value must be at most ${field.max}`;
            }
        }
        
        // Date validation
        if (field.type === 'date' && value) {
            const date = new Date(value);
            if (isNaN(date.getTime())) {
                isValid = false;
                errorMessage = 'Please enter a valid date';
            }
        }
        
        if (!isValid) {
            this.showFieldError(field, errorMessage);
        } else {
            this.clearFieldError(field);
        }
        
        return isValid;
    }
    
    showFieldError(field, message) {
        this.clearFieldError(field);
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        
        field.parentNode.appendChild(errorDiv);
        field.classList.add('error');
    }
    
    clearFieldError(field) {
        const errorDiv = field.parentNode.querySelector('.field-error');
        if (errorDiv) {
            errorDiv.remove();
        }
        field.classList.remove('error');
    }
    
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    // ===== KEYBOARD SHORTCUTS =====
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit form
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const activeForm = document.querySelector('form:focus-within');
                if (activeForm) {
                    activeForm.dispatchEvent(new Event('submit'));
                }
            }
            
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('input[type="search"], input[placeholder*="search"]');
                if (searchInput) {
                    searchInput.focus();
                }
            }
        });
    }
    
    // ===== RESPONSIVE BEHAVIOR =====
    setupResponsiveBehavior() {
        this.handleResize();
        window.addEventListener('resize', () => this.handleResize());
    }
    
    handleResize() {
        const width = window.innerWidth;
        
        if (width <= 768) {
            this.enableMobileMode();
        } else {
            this.disableMobileMode();
        }
    }
    
    enableMobileMode() {
        document.body.classList.add('mobile-mode');
        
        // Collapse panels on mobile
        const panels = document.querySelectorAll('.client-panel, .automation-panel');
        panels.forEach(panel => {
            panel.classList.add('collapsed');
        });
    }
    
    disableMobileMode() {
        document.body.classList.remove('mobile-mode');
        
        // Expand panels on desktop
        const panels = document.querySelectorAll('.client-panel, .automation-panel');
        panels.forEach(panel => {
            panel.classList.remove('collapsed');
        });
    }
    
    // ===== RESULTS DISPLAY =====
    showResults(results) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');
        
        if (!resultsSection || !resultsContent) return;
        
        // Clear previous results
        resultsContent.innerHTML = '';
        
        // Create results content
        const resultsHTML = this.createResultsHTML(results);
        resultsContent.innerHTML = resultsHTML;
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Animate in
        resultsSection.classList.add('fade-in');
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    createResultsHTML(results) {
        if (!results || !results.package) {
            return '<p>No results available</p>';
        }
        
        const package = results.package;
        const quote = results.quote;
        
        return `
            <div class="results-overview">
                <h3>${results.client}'s Travel Package</h3>
                <div class="package-summary">
                    <div class="package-name">${package.name}</div>
                    <div class="package-price">$${package.totalPrice.toLocaleString()}</div>
                </div>
            </div>
            
            <div class="results-grid">
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">Flight Details</div>
                        <div class="result-price">$${package.flight ? package.flight.price : 'N/A'}</div>
                    </div>
                    <div class="result-details">
                        <div class="result-detail">
                            <div class="result-detail-label">Airline</div>
                            <div class="result-detail-value">${package.flight || 'Air France Business'}</div>
                        </div>
                        <div class="result-detail">
                            <div class="result-detail-label">Class</div>
                            <div class="result-detail-value">Business Class</div>
                        </div>
                        <div class="result-detail">
                            <div class="result-detail-label">Route</div>
                            <div class="result-detail-value">${results.formData?.originCity || 'N/A'} → ${results.formData?.destinationCity || 'N/A'}</div>
                        </div>
                    </div>
                </div>
                
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">Hotel Details</div>
                        <div class="result-price">$${package.hotel ? package.hotel.price : 'N/A'}</div>
                    </div>
                    <div class="result-details">
                        <div class="result-detail">
                            <div class="result-detail-label">Hotel</div>
                            <div class="result-detail-value">${package.hotel || 'Le Bristol Paris'}</div>
                        </div>
                        <div class="result-detail">
                            <div class="result-detail-label">Rating</div>
                            <div class="result-detail-value">5 Stars</div>
                        </div>
                        <div class="result-detail">
                            <div class="result-detail-label">Location</div>
                            <div class="result-detail-value">8th Arrondissement, Paris</div>
                        </div>
                    </div>
                </div>
                
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">Financial Summary</div>
                        <div class="result-price">$${package.totalPrice.toLocaleString()}</div>
                    </div>
                    <div class="result-details">
                        <div class="result-detail">
                            <div class="result-detail-label">Package Cost</div>
                            <div class="result-detail-value">$${package.totalPrice.toLocaleString()}</div>
                        </div>
                        <div class="result-detail">
                            <div class="result-detail-label">Commission</div>
                            <div class="result-detail-value">$${package.commission}%</div>
                        </div>
                        <div class="result-detail">
                            <div class="result-detail-label">Profit</div>
                            <div class="result-detail-value">$${package.profit}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="results-actions">
                <button class="btn btn-primary" onclick="this.confirmBooking()">
                    <i class="fas fa-check"></i>
                    Confirm Booking
                </button>
                <button class="btn btn-secondary" onclick="this.generatePDF()">
                    <i class="fas fa-file-pdf"></i>
                    Generate PDF
                </button>
            </div>
        `;
    }
    
    // ===== UTILITY FUNCTIONS =====
    formatCurrency(amount, currency = 'USD') {
        if (currency === 'IDR') {
            return `Rp${amount.toLocaleString('id-ID')}`;
        } else {
            return `$${amount.toLocaleString('en-US')}`;
        }
    }
    
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString();
    }
    
    // ===== PUBLIC METHODS =====
    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }
    
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    updateStatus(message, type = 'info') {
        this.showNotification(message, type);
    }
}

// ===== INITIALIZATION =====
let travelAgencyUI;

document.addEventListener('DOMContentLoaded', function() {
    travelAgencyUI = new TravelAgencyUI();
});

// ===== GLOBAL FUNCTIONS =====
function showNotification(message, type = 'info', duration = 5000) {
    if (travelAgencyUI) {
        travelAgencyUI.showNotification(message, type, duration);
    }
}

function updateStatus(message, type = 'info') {
    if (travelAgencyUI) {
        travelAgencyUI.updateStatus(message, type);
    }
}

// ===== EXPORT FOR OTHER MODULES =====
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TravelAgencyUI;
}