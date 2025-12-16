#!/bin/bash

# Create output directory
mkdir -p docs_docx

# Convert README.md first
echo "Converting README.md..."
pandoc README.md -o docs_docx/01_README.docx

# Convert each markdown file in the docs directory
cd docs/
counter=2

for file in *.md; do
    if [ -f "$file" ]; then
        echo "Converting $file..."
        # Format counter with leading zero if needed
        padded_counter=$(printf "%02d" $counter)
        output_file="../docs_docx/${padded_counter}_${file%.md}.docx"
        pandoc "$file" -o "$output_file"
        ((counter++))
    fi
done

echo "Creating combined document..."
# Now combine all DOCX files into a single comprehensive document
cd ../docs_docx/
pandoc -o "../CLA_Documentation.docx" *.docx

echo "Conversion complete!"
echo "The combined document is available at ~/Downloads/ai-driven-soc/CLA_Documentation.docx"
echo "Individual DOCX files are in the ~/Downloads/ai-driven-soc/docs_docx/ directory"