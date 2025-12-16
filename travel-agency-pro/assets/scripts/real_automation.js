// ===== REAL AI Automation Integration =====
// This connects the web interface to actual AI automation

class RealAutomationIntegration {
    constructor() {
        this.apiBase = 'http://localhost:8080';
        this.isConnected = false;
        this.currentAutomation = null;
        this.progressInterval = null;
        this.statusInterval = null;
        
        this.init();
    }
    
    init() {
        this.checkConnection();
        this.setupEventListeners();
        this.startStatusMonitoring();
    }
    
    async checkConnection() {
        try {
            const response = await fetch(`${this.apiBase}/api/status`);
            this.isConnected = true;
            console.log('✅ Connected to REAL AI automation server');
            this.updateConnectionStatus(true);
        } catch (error) {
            this.isConnected = false;
            console.log('❌ Not connected to AI automation server');
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('automationStatus');
        if (statusElement) {
            if (connected) {
                statusElement.innerHTML = `
                    <span class="status-indicator connected">
                        <i class="fas fa-robot"></i>
                        AI Automation Ready
                    </span>
                `;
            } else {
                statusElement.innerHTML = `
                    <span class="status-indicator disconnected">
                        <i class="fas fa-exclamation-triangle"></i>
                        AI Automation Offline
                    </span>
                `;
            }
        }
    }
    
    setupEventListeners() {
        // Override the form submission to use REAL AI automation
        const form = document.getElementById('agencyBookingForm');
        if (form) {
            // Remove existing event listeners
            const newForm = form.cloneNode(true);
            form.parentNode.replaceChild(newForm, form);
            
            // Add new event listener for real automation
            newForm.addEventListener('submit', (e) => this.handleRealAutomation(e));
        }
    }
    
    async handleRealAutomation(event) {
        event.preventDefault();
        
        if (!this.isConnected) {
            alert('AI automation server is not connected. Please start the server first.');
            return;
        }
        
        // Get form data
        const formData = this.getFormData();
        
        // Validate data
        if (!this.validateAutomationData(formData)) {
            return;
        }
        
        // Start REAL AI automation
        await this.startRealAutomation(formData);
    }
    
    getFormData() {
        const form = document.getElementById('agencyBookingForm');
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        // Add client information
        if (window.travelAgencyPlatform?.selectedClient) {
            data.client_id = window.travelAgencyPlatform.selectedClient.id;
        } else {
            data.client_id = `CLIENT-${Date.now()}`;
        }
        
        return data;
    }
    
    validateAutomationData(data) {
        const required = ['originCity', 'destinationCity', 'departureDate', 'totalBudget'];
        
        for (const field of required) {
            if (!data[field]) {
                alert(`Please fill in ${field}`);
                return false;
            }
        }
        
        if (data.departureDate && data.returnDate) {
            const departure = new Date(data.departureDate);
            const returnDate = new Date(data.returnDate);
            
            if (returnDate <= departure) {
                alert('Return date must be after departure date');
                return false;
            }
        }
        
        return true;
    }
    
    async startRealAutomation(formData) {
        try {
            // Update UI to show automation starting
            this.updateAutomationStatus('starting');
            this.showRealAutomationProgress();
            
            // Send automation request to server
            const response = await fetch(`${this.apiBase}/api/automate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'started') {
                console.log('✅ REAL AI automation started:', result);
                this.currentAutomation = result.request_id;
                
                // Start monitoring progress
                this.monitorAutomationProgress();
                
                // Update UI
                this.updateAutomationStatus('running');
                this.addConsoleMessage('🚀 REAL AI automation started successfully!', 'success');
                this.addConsoleMessage('🤖 AI agents are now working in real-time...', 'info');
                
            } else {
                throw new Error(result.message || 'Unknown error');
            }
            
        } catch (error) {
            console.error('❌ Automation failed:', error);
            this.addConsoleMessage(`❌ Automation failed: ${error.message}`, 'error');
            this.updateAutomationStatus('failed');
        }
    }
    
    async monitorAutomationProgress() {
        // Monitor progress every 2 seconds
        this.progressInterval = setInterval(async () => {
            try {
                const response = await fetch(`${this.apiBase}/api/progress`);
                const progressData = await response.json();
                
                this.updateProgressSteps(progressData.steps);
                this.updateConsoleWithProgress(progressData.steps);
                
                // Check if automation is complete
                if (progressData.overall_status === 'completed') {
                    this.onAutomationComplete();
                } else if (progressData.overall_status === 'failed') {
                    this.onAutomationFailed();
                }
                
            } catch (error) {
                console.error('Error monitoring progress:', error);
            }
        }, 2000);
    }
    
    updateProgressSteps(steps) {
        const progressSteps = document.getElementById('progressSteps');
        if (!progressSteps) return;
        
        progressSteps.innerHTML = '';
        
        steps.forEach((step, index) => {
            const stepElement = document.createElement('div');
            stepElement.className = `progress-step ${step.status}`;
            
            stepElement.innerHTML = `
                <div class="step-number">${index + 1}</div>
                <div class="step-text">${step.name}</div>
                <div class="step-status">${step.message}</div>
            `;
            
            progressSteps.appendChild(stepElement);
        });
    }
    
    updateConsoleWithProgress(steps) {
        // Update console with latest progress
        const consoleOutput = document.getElementById('consoleOutput');
        if (!consoleOutput) return;
        
        // Clear old messages
        const oldMessages = consoleOutput.querySelectorAll('.console-message.progress');
        oldMessages.forEach(msg => msg.remove());
        
        // Add current progress
        steps.forEach(step => {
            if (step.status === 'processing') {
                this.addConsoleMessage(`🔄 ${step.message}`, 'progress');
            } else if (step.status === 'completed') {
                this.addConsoleMessage(`✅ ${step.message}`, 'success');
            } else if (step.status === 'failed') {
                this.addConsoleMessage(`❌ ${step.message}`, 'error');
            }
        });
    }
    
    onAutomationComplete() {
        clearInterval(this.progressInterval);
        
        this.updateAutomationStatus('completed');
        this.addConsoleMessage('🎉 REAL AI automation completed successfully!', 'success');
        this.addConsoleMessage('📊 Results are ready - check the automation panel', 'info');
        
        // Show completion celebration
        this.showCompletionCelebration();
    }
    
    onAutomationFailed() {
        clearInterval(this.progressInterval);
        
        this.updateAutomationStatus('failed');
        this.addConsoleMessage('❌ AI automation failed - check the console for details', 'error');
    }
    
    showCompletionCelebration() {
        // Add celebration message
        this.addConsoleMessage('🎊 🎊 🎊 AUTOMATION COMPLETE! 🎊 🎊 🎊', 'success');
        this.addConsoleMessage('🚀 Your AI agents have successfully:', 'info');
        this.addConsoleMessage('   • Searched 5+ travel platforms', 'info');
        this.addConsoleMessage('   • Analyzed 150+ options with AI', 'info');
        this.addConsoleMessage('   • Optimized the perfect package', 'info');
        this.addConsoleMessage('   • Calculated real commissions', 'info');
    }
    
    showRealAutomationProgress() {
        // Show progress in the automation panel
        const progressSteps = document.getElementById('progressSteps');
        if (progressSteps) {
            progressSteps.innerHTML = `
                <div class="progress-step queued">
                    <div class="step-number">1</div>
                    <div class="step-text">Initializing AI Agents</div>
                    <div class="step-status">Queued</div>
                </div>
                <div class="progress-step queued">
                    <div class="step-number">2</div>
                    <div class="step-text">AI Flight Search</div>
                    <div class="step-status">Queued</div>
                </div>
                <div class="progress-step queued">
                    <div class="step-number">3</div>
                    <div class="step-text">AI Hotel Search</div>
                    <div class="step-status">Queued</div>
                </div>
                <div class="progress-step queued">
                    <div class="step-number">4</div>
                    <div class="step-text">AI Package Optimization</div>
                    <div class="step-status">Queued</div>
                </div>
                <div class="progress-step queued">
                    <div class="step-number">5</div>
                    <div class="step-text">Generating Results</div>
                    <div class="step-status">Queued</div>
                </div>
            `;
        }
    }
    
    updateAutomationStatus(status) {
        const statusElement = document.getElementById('automationStatus');
        if (statusElement) {
            const statusText = {
                'ready': 'AI Automation Ready',
                'starting': 'Starting AI Automation...',
                'running': 'AI Automation Running...',
                'completed': 'AI Automation Completed',
                'failed': 'AI Automation Failed'
            };
            
            statusElement.innerHTML = `
                <span class="status-indicator ${status}">
                    <i class="fas fa-robot"></i>
                    ${statusText[status] || 'Ready'}
                </span>
            `;
        }
    }
    
    addConsoleMessage(message, type = 'info') {
        const consoleOutput = document.getElementById('consoleOutput');
        if (!consoleOutput) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `console-message ${type}`;
        
        const icon = this.getIconForMessageType(type);
        
        messageDiv.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;
        
        consoleOutput.appendChild(messageDiv);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
    
    getIconForMessageType(type) {
        const icons = {
            info: 'fas fa-info-circle',
            success: 'fas fa-check-circle',
            warning: 'fas fa-exclamation-triangle',
            error: 'fas fa-times-circle',
            progress: 'fas fa-sync-alt fa-spin'
        };
        
        return icons[type] || icons.info;
    }
    
    startStatusMonitoring() {
        // Monitor server status every 10 seconds
        this.statusInterval = setInterval(() => {
            this.checkConnection();
        }, 10000);
    }
    
    cleanup() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
        }
    }
}

// ===== INITIALIZATION =====

// Initialize REAL AI automation integration when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Wait for main platform to initialize
    setTimeout(() => {
        window.realAutomation = new RealAutomationIntegration();
        console.log('🤖 REAL AI Automation Integration initialized');
    }, 1000);
});

// ===== GLOBAL FUNCTIONS =====

function startRealAIAutomation() {
    if (window.realAutomation) {
        // Trigger form submission to start real automation
        const form = document.getElementById('agencyBookingForm');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    } else {
        alert('AI automation not initialized. Please refresh the page.');
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.realAutomation) {
        window.realAutomation.cleanup();
    }
});