# Troubleshooting File Upload

If the file isn't found, try these commands in your SSH session:

## Check if file exists anywhere
```bash
# Check current directory
ls -la

# Search for the file
find ~ -name "COPY_PASTE_TO_SSH.sh" 2>/dev/null

# Check if it was uploaded with a different name
ls -la | grep -i copy
ls -la | grep -i ssh
```

## If file not found, re-upload:
1. Click "UPLOAD FILE" button again
2. Make sure you're in the home directory (`~`) when uploading
3. Verify the upload completed

## Alternative: Create the file directly
If upload keeps failing, you can create the file directly by copy-pasting the content.

