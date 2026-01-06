"""
Configuration file for SOC Dashboard
Store your API keys and database credentials here
"""

# Database configurations
DATABASE_CONFIG = {
    'sqlite': {
        'path': 'soc_data.db'
    },
    'postgresql': {
        'host': 'localhost',
        'port': 5432,
        'database': 'soc_db',
        'user': 'postgres',
        'password': 'your_password'
    },
    'mysql': {
        'host': 'localhost',
        'port': 3306,
        'database': 'soc_db',
        'user': 'root',
        'password': 'your_password'
    }
}

# API Keys (Get these from respective services)
API_KEYS = {
    'abuseipdb': 'your_abuseipdb_api_key_here',
    'virustotal': 'your_virustotal_api_key_here',
    'otx': 'your_alienvault_otx_api_key_here'
}

# Sample log file paths
LOG_PATHS = {
    'security_logs': '/var/log/security.log',
    'apache_logs': '/var/log/apache2/access.log',
    'nginx_logs': '/var/log/nginx/access.log',
    'firewall_logs': '/var/log/ufw.log'
}

# Real-time data refresh intervals (seconds)
REFRESH_INTERVALS = {
    'fast': 5,
    'medium': 10,
    'slow': 30
}
