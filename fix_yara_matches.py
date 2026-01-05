#!/usr/bin/env python3
"""
Fix YARA match string handling for Python 3.6 compatibility
"""

import sys

def fix_yara_matches():
    file_path = "thor_endpoint_agent.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace the problematic line
    old_code = """                        "strings": [(s[0], s[1], s[2].decode('utf-8', errors='ignore'))
                                   for s in match.strings[:5]]  # Limit string output"""
    
    new_code = """                        "strings": [(
                                    getattr(s, 'identifier', str(s)),
                                    getattr(s, 'offset', 0),
                                    getattr(s, 'string', b'').decode('utf-8', errors='ignore') if hasattr(s, 'string') else str(s)
                                ) for s in match.strings[:5]]  # Limit string output"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w') as f:
            f.write(content)
        print("✓ Fixed YARA string match handling")
        return True
    else:
        # Try alternative fix
        old_code2 = 'for s in match.strings[:5]]'
        new_code2 = 'for s in list(match.strings)[:5]]'
        
        if old_code2 in content:
            # More comprehensive fix needed
            import re
            pattern = r'"strings": \[\(s\[0\], s\[1\], s\[2\]\.decode\([^)]+\)\)\s+for s in match\.strings\[:5\]\]'
            replacement = '"strings": [(getattr(s, "identifier", str(s)), getattr(s, "offset", 0), getattr(s, "string", b"").decode("utf-8", errors="ignore") if hasattr(s, "string") else str(s)) for s in list(match.strings)[:5]]'
            
            content = re.sub(pattern, replacement, content)
            with open(file_path, 'w') as f:
                f.write(content)
            print("✓ Fixed YARA string match handling (regex)")
            return True
        else:
            print("⚠ Could not find exact pattern to fix")
            print("  Manual fix may be needed")
            return False

if __name__ == "__main__":
    fix_yara_matches()

