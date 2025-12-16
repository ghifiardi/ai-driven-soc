// Enhanced Feedback CSV Extraction Script
// =====================================
// Run this in your browser's developer console when viewing the feedback dashboard
// URL: http://10.45.254.19:99/feedback

(function() {
    console.log("🚀 Starting Feedback CSV Extraction...");
    
    // Wait for page to fully load
    setTimeout(function() {
        console.log("🔍 Looking for data tables...");
        
        // Try multiple selectors for different table structures
        const selectors = [
            "table",
            "[data-testid='stTable']",
            ".stTable",
            ".dataframe",
            "div[data-testid='stTable'] table",
            ".streamlit-table table",
            "div.streamlit-table table"
        ];
        
        let table = null;
        let usedSelector = null;
        
        // Find the first available table
        for (const selector of selectors) {
            table = document.querySelector(selector);
            if (table) {
                usedSelector = selector;
                console.log(`✅ Found table using selector: ${selector}`);
                break;
            }
        }
        
        if (!table) {
            console.log("❌ No table found with standard selectors");
            console.log("🔍 Looking for alternative data structures...");
            
            // Try to find data in divs or other containers
            const dataContainers = [
                "div[data-testid='stTable']",
                ".stTable",
                ".dataframe",
                "div.streamlit-table"
            ];
            
            for (const containerSelector of dataContainers) {
                const container = document.querySelector(containerSelector);
                if (container) {
                    console.log(`📦 Found data container: ${containerSelector}`);
                    
                    // Try to extract data from the container
                    const rows = container.querySelectorAll("tr, div[role='row']");
                    if (rows.length > 0) {
                        console.log(`📊 Found ${rows.length} rows in container`);
                        extractDataFromContainer(container);
                        return;
                    }
                }
            }
            
            // If still no data found, show page analysis
            analyzePageStructure();
            return;
        }
        
        console.log(`📊 Extracting data from table (${usedSelector})...`);
        
        // Extract table data
        const rows = Array.from(table.querySelectorAll("tr"));
        console.log(`📋 Found ${rows.length} rows`);
        
        if (rows.length === 0) {
            console.log("❌ No rows found in table");
            return;
        }
        
        // Convert rows to CSV format
        const csvRows = rows.map((row, index) => {
            const cells = Array.from(row.querySelectorAll("th, td"));
            const rowData = cells.map(cell => {
                // Clean up cell content
                let content = cell.innerText || cell.textContent || "";
                content = content.trim();
                
                // Handle empty cells
                if (!content) content = "";
                
                // Escape quotes and wrap in quotes
                content = content.replace(/"/g, '""');
                return `"${content}"`;
            }).join(",");
            
            console.log(`📝 Row ${index + 1}: ${cells.length} cells`);
            return rowData;
        });
        
        // Create CSV content
        const csvContent = csvRows.join("\n");
        
        // Create and download file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `feedback_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
        
        // Add to page and click
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up
        URL.revokeObjectURL(url);
        
        console.log("✅ CSV file downloaded successfully!");
        console.log(`📊 Extracted ${rows.length} rows of data`);
        
        // Show preview of data
        console.log("📋 Data Preview:");
        console.log(csvRows.slice(0, 3).join("\n"));
        if (csvRows.length > 3) {
            console.log("... (and more rows)");
        }
        
    }, 2000); // Wait 2 seconds for page to load
    
    function extractDataFromContainer(container) {
        console.log("📦 Extracting data from container...");
        
        // Try different row selectors
        const rowSelectors = ["tr", "div[role='row']", "div.row", ".row"];
        let rows = [];
        
        for (const selector of rowSelectors) {
            rows = Array.from(container.querySelectorAll(selector));
            if (rows.length > 0) {
                console.log(`✅ Found ${rows.length} rows using selector: ${selector}`);
                break;
            }
        }
        
        if (rows.length === 0) {
            console.log("❌ No rows found in container");
            return;
        }
        
        // Extract data from rows
        const csvRows = rows.map((row, index) => {
            // Try different cell selectors
            const cellSelectors = ["th, td", "div[role='cell']", "div.cell", ".cell", "span", "div"];
            let cells = [];
            
            for (const selector of cellSelectors) {
                cells = Array.from(row.querySelectorAll(selector));
                if (cells.length > 0) {
                    break;
                }
            }
            
            const rowData = cells.map(cell => {
                let content = cell.innerText || cell.textContent || "";
                content = content.trim();
                if (!content) content = "";
                content = content.replace(/"/g, '""');
                return `"${content}"`;
            }).join(",");
            
            console.log(`📝 Row ${index + 1}: ${cells.length} cells`);
            return rowData;
        });
        
        // Create and download CSV
        const csvContent = csvRows.join("\n");
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `feedback_data_container_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        console.log("✅ CSV file downloaded from container!");
        console.log(`📊 Extracted ${rows.length} rows of data`);
    }
    
    function analyzePageStructure() {
        console.log("🔍 Analyzing page structure...");
        
        // Get page information
        console.log("📄 Page Info:");
        console.log(`  - Title: ${document.title}`);
        console.log(`  - URL: ${window.location.href}`);
        console.log(`  - Content Length: ${document.body.innerText.length} characters`);
        
        // Count different elements
        const elementCounts = {
            "tables": document.querySelectorAll("table").length,
            "divs": document.querySelectorAll("div").length,
            "spans": document.querySelectorAll("span").length,
            "paragraphs": document.querySelectorAll("p").length,
            "lists": document.querySelectorAll("ul, ol").length
        };
        
        console.log("📊 Element Counts:");
        Object.entries(elementCounts).forEach(([element, count]) => {
            console.log(`  - ${element}: ${count}`);
        });
        
        // Look for Streamlit-specific elements
        const streamlitElements = document.querySelectorAll("[data-testid]");
        console.log(`🎯 Streamlit Elements: ${streamlitElements.length}`);
        
        if (streamlitElements.length > 0) {
            console.log("📋 Streamlit Element Types:");
            const elementTypes = {};
            streamlitElements.forEach(el => {
                const testid = el.getAttribute("data-testid");
                elementTypes[testid] = (elementTypes[testid] || 0) + 1;
            });
            Object.entries(elementTypes).forEach(([type, count]) => {
                console.log(`  - ${type}: ${count}`);
            });
        }
        
        // Look for data in text content
        const bodyText = document.body.innerText;
        if (bodyText.includes("feedback") || bodyText.includes("rating") || bodyText.includes("comment")) {
            console.log("💡 Found potential feedback data in page text");
            console.log("📝 Consider manual extraction or contact admin for data access");
        }
        
        console.log("💡 Recommendations:");
        console.log("  1. Check if the page is still loading");
        console.log("  2. Try refreshing the page and running the script again");
        console.log("  3. Look for any error messages on the page");
        console.log("  4. Contact the dashboard administrator for data access");
    }
    
})();

console.log("🎯 Feedback CSV Extraction Script Loaded");
console.log("📋 Instructions:");
console.log("  1. Navigate to: http://10.45.254.19:99/feedback");
console.log("  2. Wait for the page to fully load");
console.log("  3. Open Developer Console (F12)");
console.log("  4. Paste and run this script");
console.log("  5. The CSV file will be automatically downloaded");


