#!/usr/bin/env python3
"""
Compare Local vs VM Directory Contents
=====================================

This script helps compare what's on your local machine vs your GCP VM
to identify missing enhanced classification files.
"""

import os
import subprocess
import json
from datetime import datetime

def get_local_files():
    """Get list of files from local directory"""
    local_path = "/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc"
    
    files = []
    for item in os.listdir(local_path):
        item_path = os.path.join(local_path, item)
        if os.path.isfile(item_path):
            stat = os.stat(item_path)
            files.append({
                'name': item,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'path': item_path
            })
    
    return files

def get_vm_files():
    """Get list of files from VM directory (requires SSH access)"""
    try:
        # Try to get VM files via SSH
        result = subprocess.run([
            'ssh', 'app@xdgaisocapp01', 
            'cd /home/app/ai-driven-soc && find . -maxdepth 1 -type f -exec ls -la {} \;'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"❌ SSH connection failed: {result.stderr}")
            return []
        
        files = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 9:
                    filename = parts[-1]
                    size = int(parts[4])
                    files.append({
                        'name': filename,
                        'size': size,
                        'modified': 'Unknown',
                        'path': f'/home/app/ai-driven-soc/{filename}'
                    })
        
        return files
    except Exception as e:
        print(f"❌ Error connecting to VM: {e}")
        return []

def compare_files():
    """Compare local vs VM files"""
    print("🔍 Comparing Local vs VM Directory Contents")
    print("=" * 60)
    
    # Get file lists
    local_files = get_local_files()
    vm_files = get_vm_files()
    
    # Create dictionaries for easy lookup
    local_dict = {f['name']: f for f in local_files}
    vm_dict = {f['name']: f for f in vm_files}
    
    # Find missing files on VM
    missing_on_vm = []
    for filename, file_info in local_dict.items():
        if filename not in vm_dict:
            missing_on_vm.append(file_info)
    
    # Find extra files on VM
    extra_on_vm = []
    for filename, file_info in vm_dict.items():
        if filename not in local_dict:
            extra_on_vm.append(file_info)
    
    # Print comparison results
    print(f"📊 File Count Comparison:")
    print(f"   Local files: {len(local_files)}")
    print(f"   VM files: {len(vm_files)}")
    print(f"   Missing on VM: {len(missing_on_vm)}")
    print(f"   Extra on VM: {len(extra_on_vm)}")
    
    # Show missing enhanced classification files
    enhanced_files = [f for f in missing_on_vm if 'enhanced' in f['name'].lower() or 'classification' in f['name'].lower()]
    
    print(f"\n🚨 CRITICAL: Missing Enhanced Classification Files on VM:")
    if enhanced_files:
        for file_info in enhanced_files:
            print(f"   ❌ {file_info['name']} ({file_info['size']:,} bytes)")
    else:
        print("   ✅ All enhanced classification files are present on VM")
    
    # Show other missing files
    other_missing = [f for f in missing_on_vm if f not in enhanced_files]
    if other_missing:
        print(f"\n📋 Other Missing Files on VM:")
        for file_info in other_missing[:10]:  # Show first 10
            print(f"   ❌ {file_info['name']} ({file_info['size']:,} bytes)")
        if len(other_missing) > 10:
            print(f"   ... and {len(other_missing) - 10} more files")
    
    # Show extra files on VM
    if extra_on_vm:
        print(f"\n📁 Extra Files on VM (not on local):")
        for file_info in extra_on_vm[:10]:  # Show first 10
            print(f"   ➕ {file_info['name']} ({file_info['size']:,} bytes)")
        if len(extra_on_vm) > 10:
            print(f"   ... and {len(extra_on_vm) - 10} more files")
    
    # Check for key TAA files
    print(f"\n🎯 TAA Agent Status:")
    taa_files = ['taa_a2a_mcp_agent.py', 'enhanced_taa_agent.py', 'enhanced_classification_engine.py']
    for taa_file in taa_files:
        local_exists = taa_file in local_dict
        vm_exists = taa_file in vm_dict
        
        status = "✅" if local_exists and vm_exists else "❌"
        print(f"   {status} {taa_file}")
        print(f"      Local: {'Present' if local_exists else 'Missing'}")
        print(f"      VM: {'Present' if vm_exists else 'Missing'}")
    
    # Deployment recommendations
    print(f"\n💡 Deployment Recommendations:")
    if enhanced_files:
        print("   🚨 URGENT: Deploy enhanced classification files immediately!")
        print("   📤 Run: ./deploy_enhanced_classification.sh")
        print("   📋 Or manually upload enhanced files to VM")
    else:
        print("   ✅ Enhanced classification system appears to be deployed")
        print("   🧪 Test the system: python3 enhanced_classification_engine.py")
    
    return {
        'local_files': len(local_files),
        'vm_files': len(vm_files),
        'missing_on_vm': len(missing_on_vm),
        'enhanced_missing': len(enhanced_files),
        'recommendations': 'deploy' if enhanced_files else 'test'
    }

def generate_vm_check_script():
    """Generate a script to run on VM for detailed comparison"""
    script_content = '''#!/bin/bash

echo "🔍 VM Directory Analysis"
echo "======================="

# Current directory
echo "📁 Current directory: $(pwd)"
echo "📁 Directory contents:"
ls -la | head -20

echo ""
echo "🎯 TAA Agent Files:"
ls -la *taa* 2>/dev/null || echo "No TAA files found"

echo ""
echo "🚀 Enhanced Classification Files:"
ls -la *enhanced* 2>/dev/null || echo "No enhanced files found"

echo ""
echo "📊 Python Files:"
ls -la *.py | wc -l
echo "Python files found"

echo ""
echo "🔧 Virtual Environments:"
ls -la venv* 2>/dev/null || echo "No virtual environments found"

echo ""
echo "📋 Requirements Files:"
ls -la requirements*.txt 2>/dev/null || echo "No requirements files found"

echo ""
echo "📈 Recent Files (last 24 hours):"
find . -maxdepth 1 -type f -mtime -1 -exec ls -la {} \\; 2>/dev/null || echo "No recent files"

echo ""
echo "🎯 Services Status:"
ps aux | grep -E "(python|taa|ada)" | grep -v grep || echo "No relevant services running"

echo ""
echo "✅ VM Analysis Complete"
'''
    
    with open('vm_directory_check.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('vm_directory_check.sh', 0o755)
    print(f"📝 Generated vm_directory_check.sh - upload and run this on your VM for detailed analysis")

if __name__ == "__main__":
    try:
        results = compare_files()
        generate_vm_check_script()
        
        print(f"\n📊 Summary:")
        print(f"   Local files: {results['local_files']}")
        print(f"   VM files: {results['vm_files']}")
        print(f"   Missing on VM: {results['missing_on_vm']}")
        print(f"   Enhanced files missing: {results['enhanced_missing']}")
        print(f"   Recommendation: {results['recommendations']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running this script from your local machine with SSH access to the VM")


