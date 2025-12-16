# GCP VM Connectivity Check Guide

## Methods to Check VM Internet & Destination Connectivity

### 1. SSH into VM and Test Connectivity

```bash
# SSH to your VM
gcloud compute ssh YOUR_VM_NAME --zone=YOUR_ZONE

# Once connected to VM, test internet connectivity
ping -c 4 8.8.8.8
ping -c 4 google.com

# Test specific destination
ping -c 4 YOUR_DESTINATION_IP_OR_DOMAIN

# Check DNS resolution
nslookup google.com
dig google.com

# Test HTTP connectivity
curl -I https://google.com
wget --spider https://google.com
```

### 2. Check VM Network Configuration

```bash
# Check network interfaces
ip addr show
ifconfig

# Check routing table
ip route show
route -n

# Check if VM has external IP
curl ifconfig.me
curl ipinfo.io
```

### 3. GCP Console Checks

#### VM Instance Details:
1. Go to Compute Engine > VM instances
2. Click on your VM name
3. Check "Network interfaces" section:
   - **Internal IP**: Always present
   - **External IP**: Check if "Ephemeral" or static IP is assigned
   - **No external IP**: VM can only access internet through NAT gateway

#### Network & Firewall:
1. Go to VPC network > VPC networks
2. Click on your VPC
3. Check subnets and routing
4. Go to VPC network > Firewall
5. Verify firewall rules allow outbound traffic

### 4. Using gcloud CLI (from local machine)

```bash
# Get VM details
gcloud compute instances describe YOUR_VM_NAME --zone=YOUR_ZONE

# Check firewall rules
gcloud compute firewall-rules list

# Check routes
gcloud compute routes list

# Check if VM has external IP
gcloud compute instances list --filter="name:YOUR_VM_NAME"
```

### 5. Test Specific Connectivity

```bash
# Test port connectivity to destination
nc -zv DESTINATION_IP PORT
telnet DESTINATION_IP PORT

# Traceroute to destination
traceroute DESTINATION_IP
mtr DESTINATION_IP

# Test with timeout
timeout 10 bash -c "cat < /dev/null > /dev/tcp/DESTINATION_IP/PORT"
```

## Common Connectivity Issues

### No Internet Access:
- **No External IP**: VM needs external IP or NAT gateway
- **Firewall Rules**: Check egress rules allow outbound traffic
- **Routes**: Verify default route exists (0.0.0.0/0)

### Can't Reach Specific Destination:
- **Firewall Rules**: Check both GCP firewall and destination firewall
- **Network Routes**: Verify routing to destination network
- **DNS Issues**: Test with IP address instead of domain name

## Quick Diagnostic Commands

```bash
# All-in-one connectivity test script
#!/bin/bash
echo "=== VM Network Configuration ==="
hostname -I
ip route show | grep default

echo "=== Internet Connectivity ==="
ping -c 2 8.8.8.8 && echo "✓ Internet OK" || echo "✗ Internet FAIL"

echo "=== DNS Resolution ==="
nslookup google.com && echo "✓ DNS OK" || echo "✗ DNS FAIL"

echo "=== External IP ==="
curl -s ifconfig.me && echo " (External IP detected)" || echo "No external IP"

echo "=== Destination Test ==="
# Replace with your destination
DEST="YOUR_DESTINATION"
ping -c 2 $DEST && echo "✓ $DEST reachable" || echo "✗ $DEST unreachable"
```

## Need Help?

Replace the following placeholders:
- `YOUR_VM_NAME`: Your actual VM instance name
- `YOUR_ZONE`: Your VM's zone (e.g., us-central1-a)
- `YOUR_DESTINATION_IP_OR_DOMAIN`: Target you want to reach
- `PORT`: Specific port if testing service connectivity




