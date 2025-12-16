// ===== TRANSLATIONS SYSTEM =====

const translations = {
    en: {
        // Header
        title: "Travel Agency Pro",
        
        // Client Panel
        clientQueue: "👥 Client Queue",
        addClient: "Add Client",
        
        // Booking Workspace
        bookingWorkspace: "✈️ Booking Workspace",
        ready: "Ready",
        tripDetails: "Trip Details",
        totalBudget: "Total Budget",
        commissionRate: "Commission Rate (%)",
        originCity: "Origin City",
        destinationCity: "Destination City",
        departureDate: "Departure Date",
        returnDate: "Return Date",
        flightPreferences: "Flight Preferences",
        flightClass: "Flight Class",
        economy: "Economy",
        premiumEconomy: "Premium Economy",
        business: "Business",
        first: "First Class",
        preferredAirline: "Preferred Airline",
        maxFlightPrice: "Max Flight Price",
        preferDirect: "Prefer Direct Flights",
        hotelPreferences: "Hotel Preferences",
        preferredArea: "Preferred Area",
        minRating: "Minimum Rating",
        threeStars: "3 Stars",
        fourStars: "4 Stars",
        fiveStars: "5 Stars",
        maxHotelPrice: "Max Hotel Price/Night",
        roomType: "Room Type",
        standard: "Standard",
        deluxe: "Deluxe",
        suite: "Suite",
        specialRequests: "Special Requests",
        specialRequestsDesc: "Additional Requirements",
        startAutomation: "Start Automated Search",
        resetForm: "Reset Form",
        searchResults: "Search Results",
        closeResults: "Close",
        
        // Automation Panel
        automationQueue: "🤖 Automation Queue",
        queueStatus: "Queue:",
        automationConsole: "Automation Console",
        consoleReady: "Console ready. Select a client and start automation.",
        currentProgress: "Current Progress",
        
        // Modal
        addNewClient: "Add New Client",
        clientName: "Client Name",
        clientBudget: "Budget",
        clientPriority: "Priority",
        urgent: "Urgent",
        medium: "Medium",
        standard: "Standard",
        tripType: "Trip Type",
        cancel: "Cancel",
        
        // Status Messages
        processing: "Processing...",
        pleaseWait: "Please wait while we search for the best travel options.",
        
        // Client Priority Labels
        urgentLabel: "URGENT",
        mediumLabel: "MEDIUM",
        standardLabel: "STANDARD",
        
        // Sample Client Data
        sampleClient1: "John & Sarah Doe",
        sampleClient2: "TechCorp Team",
        sampleClient3: "Santoso Family",
        sampleTrip1: "Honeymoon Package",
        sampleTrip2: "Corporate Retreat",
        sampleTrip3: "Family Vacation"
    },
    
    id: {
        // Header
        title: "Travel Agency Pro",
        
        // Client Panel
        clientQueue: "👥 Antrian Klien",
        addClient: "Tambah Klien",
        
        // Booking Workspace
        bookingWorkspace: "✈️ Ruang Kerja Booking",
        ready: "Siap",
        tripDetails: "Detail Perjalanan",
        totalBudget: "Total Budget",
        commissionRate: "Tingkat Komisi (%)",
        originCity: "Kota Asal",
        destinationCity: "Kota Tujuan",
        departureDate: "Tanggal Berangkat",
        returnDate: "Tanggal Kembali",
        flightPreferences: "Preferensi Penerbangan",
        flightClass: "Kelas Penerbangan",
        economy: "Ekonomi",
        premiumEconomy: "Ekonomi Premium",
        business: "Bisnis",
        first: "Kelas Utama",
        preferredAirline: "Maskapai Pilihan",
        maxFlightPrice: "Harga Maksimal Penerbangan",
        preferDirect: "Lebih Suka Penerbangan Langsung",
        hotelPreferences: "Preferensi Hotel",
        preferredArea: "Area Pilihan",
        minRating: "Rating Minimal",
        threeStars: "3 Bintang",
        fourStars: "4 Bintang",
        fiveStars: "5 Bintang",
        maxHotelPrice: "Harga Maksimal Hotel/Malam",
        roomType: "Tipe Kamar",
        standard: "Standar",
        deluxe: "Deluxe",
        suite: "Suite",
        specialRequests: "Permintaan Khusus",
        specialRequestsDesc: "Persyaratan Tambahan",
        startAutomation: "Mulai Pencarian Otomatis",
        resetForm: "Reset Form",
        searchResults: "Hasil Pencarian",
        closeResults: "Tutup",
        
        // Automation Panel
        automationQueue: "🤖 Antrian Otomasi",
        queueStatus: "Antrian:",
        automationConsole: "Konsol Otomasi",
        consoleReady: "Konsol siap. Pilih klien dan mulai otomasi.",
        currentProgress: "Progress Saat Ini",
        
        // Modal
        addNewClient: "Tambah Klien Baru",
        clientName: "Nama Klien",
        clientBudget: "Budget",
        clientPriority: "Prioritas",
        urgent: "Mendesak",
        medium: "Sedang",
        standard: "Standar",
        tripType: "Jenis Perjalanan",
        cancel: "Batal",
        
        // Status Messages
        processing: "Memproses...",
        pleaseWait: "Mohon tunggu sementara kami mencari opsi perjalanan terbaik.",
        
        // Client Priority Labels
        urgentLabel: "MENDESAK",
        mediumLabel: "SEDANG",
        standardLabel: "STANDAR",
        
        // Sample Client Data
        sampleClient1: "Budi & Sari Wijaya",
        sampleClient2: "Tim TechCorp",
        sampleClient3: "Keluarga Santoso",
        sampleTrip1: "Paket Bulan Madu",
        sampleTrip2: "Retreat Korporat",
        sampleTrip3: "Liburan Keluarga"
    }
};

// ===== LANGUAGE SWITCHING FUNCTION =====
function switchLanguage(lang) {
    // Update active button state
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-lang="${lang}"]`).classList.add('active');
    
    // Update all translatable elements
    document.querySelectorAll('[data-lang]').forEach(element => {
        const key = element.getAttribute('data-lang');
        if (translations[lang] && translations[lang][key]) {
            element.textContent = translations[lang][key];
        }
    });
    
    // Update currency symbols
    updateCurrencySymbols(lang);
    
    // Update form placeholders
    updateFormPlaceholders(lang);
    
    // Update sample client data
    updateSampleClientData(lang);
    
    // Store language preference
    localStorage.setItem('preferredLanguage', lang);
    
    // Trigger custom event for other scripts
    document.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang } }));
}

// ===== CURRENCY SYMBOL UPDATES =====
function updateCurrencySymbols(lang) {
    const symbols = {
        en: '$',
        id: 'Rp'
    };
    
    const symbolElements = document.querySelectorAll('.currency-symbol');
    symbolElements.forEach(element => {
        element.textContent = symbols[lang] || '$';
    });
}

// ===== FORM PLACEHOLDER UPDATES =====
function updateFormPlaceholders(lang) {
    const placeholders = {
        en: {
            originCity: "New York",
            destinationCity: "Paris",
            preferredAirline: "Any",
            preferredArea: "City Center"
        },
        id: {
            originCity: "Jakarta",
            destinationCity: "Paris",
            preferredAirline: "Sembarang",
            preferredArea: "Pusat Kota"
        }
    };
    
    const currentPlaceholders = placeholders[lang] || placeholders.en;
    
    // Update input placeholders
    if (document.getElementById('originCity')) {
        document.getElementById('originCity').placeholder = currentPlaceholders.originCity;
    }
    if (document.getElementById('destinationCity')) {
        document.getElementById('destinationCity').placeholder = currentPlaceholders.destinationCity;
    }
    if (document.getElementById('preferredAirline')) {
        document.getElementById('preferredAirline').placeholder = currentPlaceholders.preferredAirline;
    }
    if (document.getElementById('preferredArea')) {
        document.getElementById('preferredArea').placeholder = currentPlaceholders.preferredArea;
    }
}

// ===== SAMPLE CLIENT DATA UPDATES =====
function updateSampleClientData(lang) {
    // This will be called when the client list is populated
    // The actual data update happens in the main.js file
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    // Load preferred language from localStorage
    const savedLanguage = localStorage.getItem('preferredLanguage') || 'en';
    
    // Set initial language
    switchLanguage(savedLanguage);
    
    // Set default dates
    setDefaultDates();
});

// ===== DEFAULT DATE SETTING =====
function setDefaultDates() {
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

// ===== EXPORT FOR OTHER MODULES =====
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { translations, switchLanguage };
}