// ===== MAIN APPLICATION CLASS =====

class TravelAgencyPlatform {
    constructor() {
        this.currentLanguage = 'en';
        this.selectedClient = null;
        this.clients = [];
        this.automationQueue = [];
        this.isRunning = false;
        this.currentStep = 0;
        
        this.init();
    }
    
    init() {
        this.loadSampleClients();
        this.setupEventListeners();
        this.updateQueueCount();
        this.setDefaultDates();
    }
    
    // ===== CLIENT MANAGEMENT =====
    loadSampleClients() {
        const sampleClients = [
            {
                id: 1,
                name: this.getTranslatedText('sampleClient1'),
                budget: 4500,
                currency: 'USD',
                priority: 'urgent',
                tripType: this.getTranslatedText('sampleTrip1'),
                specialRequests: 'Luxury honeymoon package with premium service'
            },
            {
                id: 2,
                name: this.getTranslatedText('sampleClient2'),
                budget: 8000,
                currency: 'USD',
                priority: 'medium',
                tripType: this.getTranslatedText('sampleTrip2'),
                specialRequests: 'Corporate retreat with meeting facilities'
            },
            {
                id: 3,
                name: this.getTranslatedText('sampleClient3'),
                budget: 3500,
                currency: 'USD',
                priority: 'standard',
                tripType: this.getTranslatedText('sampleTrip3'),
                specialRequests: 'Family-friendly accommodations'
            }
        ];
        
        this.clients = sampleClients;
        this.renderClientList();
    }
    
    renderClientList() {
        const clientList = document.querySelector('.client-list');
        if (!clientList) return;
        
        clientList.innerHTML = '';
        
        this.clients.forEach(client => {
            const clientElement = this.createClientElement(client);
            clientList.appendChild(clientElement);
        });
    }
    
    createClientElement(client) {
        const clientDiv = document.createElement('div');
        clientDiv.className = 'client-item';
        clientDiv.dataset.clientId = client.id;
        
        const priorityLabel = this.getTranslatedText(`${client.priority}Label`);
        const budgetDisplay = this.formatCurrency(client.budget, client.currency);
        
        clientDiv.innerHTML = `
            <div class="client-header">
                <div class="client-name">${client.name}</div>
                <div class="client-priority ${client.priority}">${priorityLabel}</div>
            </div>
            <div class="client-details">
                <div class="client-budget">${budgetDisplay}</div>
                <div class="client-type">${client.tripType}</div>
            </div>
        `;
        
        clientDiv.addEventListener('click', () => this.selectClient(client));
        
        return clientDiv;
    }
    
    selectClient(client) {
        // Remove active class from all clients
        document.querySelectorAll('.client-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to selected client
        const clientElement = document.querySelector(`[data-client-id="${client.id}"]`);
        if (clientElement) {
            clientElement.classList.add('active');
        }
        
        this.selectedClient = client;
        this.loadClientDataIntoForm(client);
        this.updateWorkspaceStatus('ready');
        
        // Add to automation queue
        this.addToAutomationQueue(client);
    }
    
    loadClientDataIntoForm(client) {
        // Update form with client data
        if (document.getElementById('totalBudget')) {
            document.getElementById('totalBudget').value = client.budget;
        }
        
        // Update special requests
        if (document.getElementById('specialRequests')) {
            document.getElementById('specialRequests').value = client.specialRequests;
        }
        
        // Update origin/destination based on language
        if (this.currentLanguage === 'id') {
            if (document.getElementById('originCity')) {
                document.getElementById('originCity').value = 'Jakarta';
            }
            if (document.getElementById('destinationCity')) {
                document.getElementById('destinationCity').value = 'Paris';
            }
        } else {
            if (document.getElementById('originCity')) {
                document.getElementById('originCity').value = 'New York';
            }
            if (document.getElementById('destinationCity')) {
                document.getElementById('destinationCity').value = 'Paris';
            }
        }
    }
    
    // ===== AUTOMATION QUEUE MANAGEMENT =====
    addToAutomationQueue(client) {
        if (!this.automationQueue.find(c => c.id === client.id)) {
            this.automationQueue.push(client);
            this.updateQueueCount();
            this.renderAutomationQueue();
        }
    }
    
    removeFromAutomationQueue(clientId) {
        this.automationQueue = this.automationQueue.filter(c => c.id !== clientId);
        this.updateQueueCount();
        this.renderAutomationQueue();
    }
    
    updateQueueCount() {
        const queueCount = document.getElementById('queueCount');
        if (queueCount) {
            queueCount.textContent = this.automationQueue.length;
        }
    }
    
    renderAutomationQueue() {
        const progressSteps = document.getElementById('progressSteps');
        if (!progressSteps) return;
        
        progressSteps.innerHTML = '';
        
        this.automationQueue.forEach((client, index) => {
            const stepElement = this.createProgressStep(client, index);
            progressSteps.appendChild(stepElement);
        });
    }
    
    createProgressStep(client, index) {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'progress-step queued';
        stepDiv.dataset.clientId = client.id;
        
        stepDiv.innerHTML = `
            <div class="step-number">${index + 1}</div>
            <div class="step-text">${client.name}</div>
            <div class="step-status">${client.tripType}</div>
        `;
        
        return stepDiv;
    }
    
    // ===== FORM HANDLING =====
    setupEventListeners() {
        // Form submission
        const form = document.getElementById('agencyBookingForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleFormSubmission(e));
        }
        
        // Language change events
        document.addEventListener('languageChanged', (e) => {
            this.currentLanguage = e.detail.language;
            this.updateClientDataForLanguage();
        });
        
        // Add client form
        const addClientForm = document.getElementById('addClientForm');
        if (addClientForm) {
            addClientForm.addEventListener('submit', (e) => this.handleAddClient(e));
        }
    }
    
    handleFormSubmission(e) {
        e.preventDefault();
        
        if (!this.selectedClient) {
            this.showNotification('Please select a client first', 'warning');
            return;
        }
        
        const formData = this.getFormData();
        if (this.validateFormData(formData)) {
            this.startAutomation(formData);
        }
    }
    
    getFormData() {
        const form = document.getElementById('agencyBookingForm');
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        // Add client information
        data.clientId = this.selectedClient.id;
        data.clientName = this.selectedClient.name;
        data.clientBudget = this.selectedClient.budget;
        data.clientCurrency = this.selectedClient.currency;
        
        return data;
    }
    
    validateFormData(data) {
        const requiredFields = ['totalBudget', 'originCity', 'destinationCity', 'departureDate'];
        
        for (let field of requiredFields) {
            if (!data[field]) {
                this.showNotification(`Please fill in ${field}`, 'error');
                return false;
            }
        }
        
        if (data.departureDate && data.returnDate) {
            const departure = new Date(data.departureDate);
            const returnDate = new Date(data.returnDate);
            
            if (returnDate <= departure) {
                this.showNotification('Return date must be after departure date', 'error');
                return false;
            }
        }
        
        return true;
    }
    
    // ===== AUTOMATION START =====
    startAutomation(formData) {
        if (this.isRunning) {
            this.showNotification('Automation is already running', 'warning');
            return;
        }
        
        this.isRunning = true;
        this.updateWorkspaceStatus('processing');
        this.showLoadingOverlay();
        
        // Start the automation process
        this.runAutomationSteps(formData);
    }
    
    runAutomationSteps(formData) {
        // This will be implemented in the automation.js file
        // For now, we'll simulate the process
        this.simulateAutomation(formData);
    }
    
    simulateAutomation(formData) {
        const steps = [
            'Initializing AI agent for luxury travel booking...',
            'Loading client preferences and budget constraints...',
            'Scanning 15+ premium travel sites simultaneously...',
            'Analyzing flight options and pricing...',
            'Evaluating hotel availability and amenities...',
            'Comparing package deals and special offers...',
            'Calculating optimal commission rates...',
            'Generating professional client quote...',
            'Preparing detailed booking summary...',
            'Finalizing automation results...'
        ];
        
        let currentStep = 0;
        
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                this.addConsoleMessage(steps[currentStep], 'info');
                this.updateProgressStep(currentStep, 'processing');
                currentStep++;
            } else {
                clearInterval(stepInterval);
                this.completeAutomation(formData);
            }
        }, 800);
    }
    
    completeAutomation(formData) {
        this.isRunning = false;
        this.updateWorkspaceStatus('completed');
        this.hideLoadingOverlay();
        this.showResults(formData);
        this.updateProgressStep(9, 'completed');
        
        this.addConsoleMessage('Automation completed successfully!', 'success');
    }
    
    // ===== UI UPDATES =====
    updateWorkspaceStatus(status) {
        const statusIndicator = document.getElementById('workspaceStatus');
        if (!statusIndicator) return;
        
        statusIndicator.className = `status-indicator ${status}`;
        
        const statusText = statusIndicator.querySelector('span');
        if (statusText) {
            switch (status) {
                case 'ready':
                    statusText.textContent = this.getTranslatedText('ready');
                    break;
                case 'processing':
                    statusText.textContent = this.getTranslatedText('processing');
                    break;
                case 'completed':
                    statusText.textContent = 'Completed';
                    break;
            }
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
            error: 'fas fa-times-circle'
        };
        
        return icons[type] || icons.info;
    }
    
    updateProgressStep(stepIndex, status) {
        const progressSteps = document.querySelectorAll('.progress-step');
        if (progressSteps[stepIndex]) {
            progressSteps[stepIndex].className = `progress-step ${status}`;
        }
    }
    
    // ===== UTILITY FUNCTIONS =====
    getTranslatedText(key) {
        return translations[this.currentLanguage]?.[key] || translations.en[key] || key;
    }
    
    formatCurrency(amount, currency) {
        if (currency === 'IDR') {
            return `Rp${amount.toLocaleString('id-ID')}`;
        } else {
            return `$${amount.toLocaleString('en-US')}`;
        }
    }
    
    showNotification(message, type = 'info') {
        // Simple notification system
        console.log(`${type.toUpperCase()}: ${message}`);
        
        // You can implement a more sophisticated notification system here
        // For now, we'll use console.log
    }
    
    showLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }
    
    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    // ===== CLIENT ADDITION =====
    handleAddClient(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const clientData = {
            id: Date.now(),
            name: formData.get('clientName'),
            budget: parseFloat(formData.get('clientBudget')),
            currency: this.currentLanguage === 'id' ? 'IDR' : 'USD',
            priority: formData.get('clientPriority'),
            tripType: formData.get('tripType'),
            specialRequests: ''
        };
        
        this.clients.push(clientData);
        this.renderClientList();
        this.closeModal();
        
        // Clear form
        e.target.reset();
    }
    
    // ===== LANGUAGE UPDATES =====
    updateClientDataForLanguage() {
        // Update client names and trip types for current language
        this.clients.forEach(client => {
            if (client.id <= 3) { // Sample clients
                client.name = this.getTranslatedText(`sampleClient${client.id}`);
                client.tripType = this.getTranslatedText(`sampleTrip${client.id}`);
            }
        });
        
        this.renderClientList();
        this.renderAutomationQueue();
    }
    
    // ===== DATE SETTING =====
    setDefaultDates() {
        const today = new Date();
        const tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);
        
        const nextWeek = new Date(today);
        nextWeek.setDate(nextWeek.getDate() + 7);
        
        const formatDate = (date) => {
            return date.toISOString().split('T')[0];
        };
        
        // Set departure date to tomorrow
        if (document.getElementById('departureDate')) {
            document.getElementById('departureDate').value = formatDate(tomorrow);
        }
        
        // Set return date to next week
        if (document.getElementById('returnDate')) {
            document.getElementById('returnDate').value = formatDate(nextWeek);
        }
    }
}

// ===== INITIALIZATION =====
let travelAgencyPlatform;

document.addEventListener('DOMContentLoaded', function() {
    travelAgencyPlatform = new TravelAgencyPlatform();
});

// ===== GLOBAL FUNCTIONS =====
function addNewClient() {
    const modal = document.getElementById('addClientModal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

function closeModal() {
    const modal = document.getElementById('addClientModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function resetForm() {
    const form = document.getElementById('agencyBookingForm');
    if (form) {
        form.reset();
        travelAgencyPlatform.setDefaultDates();
    }
}

function clearConsole() {
    const consoleOutput = document.getElementById('consoleOutput');
    if (consoleOutput) {
        consoleOutput.innerHTML = `
            <div class="console-message info">
                <i class="fas fa-info-circle"></i>
                <span>${travelAgencyPlatform.getTranslatedText('consoleReady')}</span>
            </div>
        `;
    }
}

function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
}

// ===== EXPORT FOR OTHER MODULES =====
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TravelAgencyPlatform;
}