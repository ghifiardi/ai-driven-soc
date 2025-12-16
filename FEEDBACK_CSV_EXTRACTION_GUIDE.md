# Feedback CSV Extraction Guide 📊

## 🎯 **Status Summary**

✅ **Feedback Dashboard is Accessible**: `http://10.45.254.19:99/feedback` (Status: 200)  
❌ **Automated Extraction Failed**: Data is dynamically loaded by Streamlit  
✅ **Solution Available**: Browser-based JavaScript extraction method  

## 🚀 **Recommended Method: Browser JavaScript**

Since the feedback dashboard is a Streamlit application that loads data dynamically, the most reliable method is to use JavaScript directly in your browser.

### **Step-by-Step Instructions**

1. **🌐 Open the Dashboard**
   - Navigate to: `http://10.45.254.19:99/feedback`
   - Wait for the page to fully load (you should see the feedback data)

2. **🔧 Open Developer Console**
   - Press `F12` or right-click → "Inspect Element"
   - Go to the "Console" tab

3. **📋 Copy and Paste JavaScript Code**
   - Copy the entire code from `extract_feedback_csv_browser.js`
   - Paste it into the console
   - Press `Enter` to execute

4. **📥 Download CSV File**
   - The script will automatically download a CSV file
   - File will be named: `feedback_data_YYYY-MM-DD_HH-MM-SS.csv`

### **JavaScript Code**

```javascript
(function() {
    console.log("🚀 Starting Feedback CSV Extraction...");
    
    setTimeout(function() {
        console.log("🔍 Looking for data tables...");
        
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
                    const rows = container.querySelectorAll("tr, div[role='row']");
                    if (rows.length > 0) {
                        console.log(`📊 Found ${rows.length} rows in container`);
                        extractDataFromContainer(container);
                        return;
                    }
                }
            }
            
            analyzePageStructure();
            return;
        }
        
        console.log(`📊 Extracting data from table (${usedSelector})...`);
        
        const rows = Array.from(table.querySelectorAll("tr"));
        console.log(`📋 Found ${rows.length} rows`);
        
        if (rows.length === 0) {
            console.log("❌ No rows found in table");
            return;
        }
        
        const csvRows = rows.map((row, index) => {
            const cells = Array.from(row.querySelectorAll("th, td"));
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
        
        const csvContent = csvRows.join("\n");
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `feedback_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
        
        console.log("✅ CSV file downloaded successfully!");
        console.log(`📊 Extracted ${rows.length} rows of data`);
        
        console.log("📋 Data Preview:");
        console.log(csvRows.slice(0, 3).join("\n"));
        if (csvRows.length > 3) {
            console.log("... (and more rows)");
        }
        
    }, 2000);
    
    function extractDataFromContainer(container) {
        console.log("📦 Extracting data from container...");
        
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
        
        const csvRows = rows.map((row, index) => {
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
        
        console.log("📄 Page Info:");
        console.log(`  - Title: ${document.title}`);
        console.log(`  - URL: ${window.location.href}`);
        console.log(`  - Content Length: ${document.body.innerText.length} characters`);
        
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
```

## 🔧 **Alternative Methods**

### **Method 1: Enhanced JavaScript (Recommended)**
Use the enhanced JavaScript code from `extract_feedback_csv_browser.js` which includes:
- Multiple table selectors
- Container data extraction
- Page structure analysis
- Better error handling

### **Method 2: Manual Extraction**
1. Right-click on the table data
2. Select "Inspect Element"
3. Copy the HTML table structure
4. Use a tool like BeautifulSoup to parse it

### **Method 3: Browser Automation**
Install Selenium and use the automated script:
```bash
pip install selenium
python3 extract_feedback_automated.py
```

### **Method 4: API Access**
Check if there's a REST API endpoint for the data:
- Try: `http://10.45.254.19:99/api/feedback`
- Try: `http://10.45.254.19:99/feedback.json`

## 🎯 **Quick Start (Your Original JavaScript)**

If you prefer the simpler approach, here's your original JavaScript code:

```javascript
(function(){
    const table = document.querySelector("table");
    if (!table) {
        console.log("No table found");
        return;
    }
    const rows = Array.from(table.querySelectorAll("tr")).map(tr => 
        Array.from(tr.querySelectorAll("th,td")).map(td => 
            '"' + td.innerText.replace(/"/g,'""').trim() + '"'
        ).join(",")
    );
    const blob = new Blob([rows.join("\n")], {type:'text/csv;charset=utf-8;'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download='feedback_table.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    console.log("Saved feedback_table.csv");
})();
```

## 📊 **Expected Results**

After running the JavaScript extraction, you should get:
- ✅ **CSV file downloaded** to your Downloads folder
- ✅ **Console output** showing extraction progress
- ✅ **Data preview** in the console
- ✅ **Row count** and extraction statistics

## 🚨 **Troubleshooting**

### **If No Table Found**
1. Wait for the page to fully load (Streamlit apps can be slow)
2. Refresh the page and try again
3. Check the console for error messages
4. Try the enhanced JavaScript version

### **If Data is Empty**
1. Verify the dashboard is showing data
2. Check if there are any filters applied
3. Try scrolling to ensure all data is loaded
4. Check for JavaScript errors in the console

### **If Download Fails**
1. Check browser download permissions
2. Try a different browser
3. Check if pop-up blockers are interfering
4. Try the manual copy-paste method

## 📁 **Generated Files**

- `extract_feedback_csv_browser.js` - Enhanced JavaScript extraction script
- `extract_feedback_automated.py` - Python automation script
- `FEEDBACK_CSV_EXTRACTION_GUIDE.md` - This comprehensive guide

## 🎉 **Success Indicators**

✅ Dashboard accessible (Status: 200)  
✅ Response time: 0.06s  
✅ Content detected: 1522 bytes  
✅ Streamlit application identified  
✅ JavaScript extraction method available  

**🎯 Ready to extract CSV data using the browser-based JavaScript method!**


