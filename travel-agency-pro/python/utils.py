#!/usr/bin/env python3
"""
Travel Agency Pro - Utility Functions

This module provides utility functions for the travel agency automation platform.
"""

import os
import json
import logging
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import re
import requests
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


@dataclass
class CurrencyInfo:
    """Currency information for international travel"""
    code: str
    symbol: str
    name: str
    exchange_rate_usd: float
    last_updated: datetime


@dataclass
class LocationInfo:
    """Location information for cities and airports"""
    name: str
    country: str
    timezone: str
    coordinates: tuple
    airport_codes: List[str]
    popular_airlines: List[str]


class CurrencyConverter:
    """Currency conversion utility"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize currency converter
        
        Args:
            api_key: API key for currency conversion service
        """
        self.api_key = api_key or os.getenv('CURRENCY_API_KEY')
        self.base_currency = 'USD'
        self.rates_cache = {}
        self.cache_duration = timedelta(hours=1)
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get exchange rate between currencies
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Exchange rate
        """
        if from_currency == to_currency:
            return 1.0
        
        cache_key = f"{from_currency}_{to_currency}"
        
        # Check cache
        if cache_key in self.rates_cache:
            cached_rate = self.rates_cache[cache_key]
            if datetime.now() - cached_rate['timestamp'] < self.cache_duration:
                return cached_rate['rate']
        
        try:
            # Use free currency API (fallback)
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            rate = data['rates'].get(to_currency, 1.0)
            
            # Cache the result
            self.rates_cache[cache_key] = {
                'rate': rate,
                'timestamp': datetime.now()
            }
            
            return rate
            
        except Exception as e:
            logger.warning(f"Failed to get exchange rate: {e}")
            # Return cached rate if available, otherwise 1.0
            if cache_key in self.rates_cache:
                return self.rates_cache[cache_key]['rate']
            return 1.0
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert amount between currencies
        
        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            
        Returns:
            Converted amount
        """
        if from_currency == to_currency:
            return amount
        
        # For demo purposes, use approximate rates
        rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.73,
            'IDR': 15000.0,
            'JPY': 110.0,
            'AUD': 1.35,
            'CAD': 1.25,
            'CHF': 0.92,
            'CNY': 6.45,
            'INR': 75.0
        }
        
        if from_currency in rates and to_currency in rates:
            usd_amount = amount / rates[from_currency]
            return usd_amount * rates[to_currency]
        
        return amount
    
    def format_currency(self, amount: float, currency: str, locale: str = 'en_US') -> str:
        """
        Format currency amount for display
        
        Args:
            amount: Amount to format
            currency: Currency code
            locale: Locale for formatting
            
        Returns:
            Formatted currency string
        """
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'IDR': 'Rp',
            'JPY': '¥',
            'AUD': 'A$',
            'CAD': 'C$',
            'CHF': 'CHF',
            'CNY': '¥',
            'INR': '₹'
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        if currency == 'IDR':
            # Indonesian Rupiah formatting
            return f"{symbol}{amount:,.0f}"
        elif currency in ['JPY', 'CNY']:
            # Japanese Yen and Chinese Yuan
            return f"{symbol}{amount:,.0f}"
        else:
            # Standard decimal formatting
            return f"{symbol}{amount:,.2f}"


class LocationManager:
    """Location and destination management utility"""
    
    def __init__(self):
        """Initialize location manager"""
        self.locations_cache = {}
        self.popular_destinations = self._load_popular_destinations()
    
    def _load_popular_destinations(self) -> Dict[str, LocationInfo]:
        """Load popular travel destinations"""
        return {
            'paris': LocationInfo(
                name='Paris',
                country='France',
                timezone='Europe/Paris',
                coordinates=(48.8566, 2.3522),
                airport_codes=['CDG', 'ORY', 'BVA'],
                popular_airlines=['Air France', 'Lufthansa', 'British Airways']
            ),
            'tokyo': LocationInfo(
                name='Tokyo',
                country='Japan',
                timezone='Asia/Tokyo',
                coordinates=(35.6762, 139.6503),
                airport_codes=['NRT', 'HND', 'HND'],
                popular_airlines=['Japan Airlines', 'ANA', 'Korean Air']
            ),
            'new_york': LocationInfo(
                name='New York',
                country='USA',
                timezone='America/New_York',
                coordinates=(40.7128, -74.0060),
                airport_codes=['JFK', 'LGA', 'EWR'],
                popular_airlines=['Delta', 'American Airlines', 'United']
            ),
            'jakarta': LocationInfo(
                name='Jakarta',
                country='Indonesia',
                timezone='Asia/Jakarta',
                coordinates=(-6.2088, 106.8456),
                airport_codes=['CGK', 'HLP', 'WII'],
                popular_airlines=['Garuda Indonesia', 'Lion Air', 'AirAsia']
            ),
            'singapore': LocationInfo(
                name='Singapore',
                country='Singapore',
                timezone='Asia/Singapore',
                coordinates=(1.3521, 103.8198),
                airport_codes=['SIN', 'SIN'],
                popular_airlines=['Singapore Airlines', 'Scoot', 'Jetstar Asia']
            )
        }
    
    def get_location_info(self, city_name: str) -> Optional[LocationInfo]:
        """
        Get location information for a city
        
        Args:
            city_name: Name of the city
            
        Returns:
            Location information or None if not found
        """
        city_key = city_name.lower().replace(' ', '_')
        return self.popular_destinations.get(city_key)
    
    def search_locations(self, query: str) -> List[LocationInfo]:
        """
        Search for locations by query
        
        Args:
            query: Search query
            
        Returns:
            List of matching locations
        """
        query = query.lower()
        matches = []
        
        for location in self.popular_destinations.values():
            if (query in location.name.lower() or 
                query in location.country.lower() or
                any(query in code.lower() for code in location.airport_codes)):
                matches.append(location)
        
        return matches
    
    def get_nearby_airports(self, city_name: str, max_distance_km: float = 100) -> List[str]:
        """
        Get nearby airports for a city
        
        Args:
            city_name: Name of the city
            max_distance_km: Maximum distance in kilometers
            
        Returns:
            List of nearby airport codes
        """
        location = self.get_location_info(city_name)
        if not location:
            return []
        
        # For demo purposes, return all airports
        # In production, you'd calculate actual distances
        return location.airport_codes


class DateUtils:
    """Date and time utility functions"""
    
    @staticmethod
    def parse_date(date_string: str) -> Optional[datetime]:
        """
        Parse date string in various formats
        
        Args:
            date_string: Date string to parse
            
        Returns:
            Parsed datetime object or None
        """
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def format_date(date: datetime, format_string: str = '%Y-%m-%d') -> str:
        """
        Format datetime object to string
        
        Args:
            date: Datetime object to format
            format_string: Format string
            
        Returns:
            Formatted date string
        """
        return date.strftime(format_string)
    
    @staticmethod
    def get_date_range(start_date: str, end_date: str) -> List[str]:
        """
        Get list of dates between start and end date
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            List of date strings
        """
        start = DateUtils.parse_date(start_date)
        end = DateUtils.parse_date(end_date)
        
        if not start or not end:
            return []
        
        dates = []
        current = start
        
        while current <= end:
            dates.append(DateUtils.format_date(current))
            current += timedelta(days=1)
        
        return dates
    
    @staticmethod
    def is_valid_date_range(start_date: str, end_date: str) -> bool:
        """
        Check if date range is valid
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            True if valid, False otherwise
        """
        start = DateUtils.parse_date(start_date)
        end = DateUtils.parse_date(end_date)
        
        if not start or not end:
            return False
        
        return start < end
    
    @staticmethod
    def get_season(date_string: str) -> str:
        """
        Get season for a given date
        
        Args:
            date_string: Date string
            
        Returns:
            Season name
        """
        date = DateUtils.parse_date(date_string)
        if not date:
            return 'Unknown'
        
        month = date.month
        
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'


class ValidationUtils:
    """Input validation utility functions"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        return len(digits) >= 10
    
    @staticmethod
    def validate_currency_amount(amount: str) -> bool:
        """
        Validate currency amount
        
        Args:
            amount: Amount string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            amount_float = float(amount)
            return amount_float > 0
        except ValueError:
            return False
    
    @staticmethod
    def validate_date_format(date_string: str) -> bool:
        """
        Validate date format
        
        Args:
            date_string: Date string to validate
            
        Returns:
            True if valid, False otherwise
        """
        return DateUtils.parse_date(date_string) is not None
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """
        Validate required fields in data dictionary
        
        Args:
            data: Data dictionary to validate
            required_fields: List of required field names
            
        Returns:
            List of missing field names
        """
        missing_fields = []
        
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        return missing_fields


class CacheManager:
    """Simple caching utility"""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache = {}
        self.default_ttl = default_ttl
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expiry': expiry
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if expired/not found
        """
        if key not in self.cache:
            return None
        
        cache_entry = self.cache[key]
        
        if datetime.now() > cache_entry['expiry']:
            del self.cache[key]
            return None
        
        return cache_entry['value']
    
    def delete(self, key: str) -> None:
        """
        Delete cache entry
        
        Args:
            key: Cache key to delete
        """
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expiry']
        ]
        
        for key in expired_keys:
            del self.cache[key]


class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_file: str = 'config.json'):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'api_keys': {
                'openai': '',
                'currency': '',
                'weather': ''
            },
            'settings': {
                'headless_mode': True,
                'timeout': 30000,
                'max_retries': 3,
                'cache_ttl': 3600
            },
            'platforms': {
                'flights': ['google_flights', 'kayak', 'expedia'],
                'hotels': ['booking', 'hotels', 'expedia']
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    return self._merge_configs(default_config, file_config)
            else:
                # Create default config file
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return default_config
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Merge custom configuration with defaults"""
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save to file
        self._save_config(self.config)


# ===== UTILITY FUNCTIONS =====

def generate_booking_id() -> str:
    """Generate unique booking ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"TA-{timestamp}-{random_suffix}"


def calculate_commission(total_amount: float, commission_rate: float) -> Dict[str, float]:
    """
    Calculate commission and pricing
    
    Args:
        total_amount: Total package cost
        commission_rate: Commission rate percentage
        
    Returns:
        Dictionary with commission calculations
    """
    commission_amount = (total_amount * commission_rate) / 100
    client_price = total_amount + commission_amount
    profit_margin = (commission_amount / client_price) * 100
    
    return {
        'total_cost': total_amount,
        'commission_rate': commission_rate,
        'commission_amount': commission_amount,
        'client_price': client_price,
        'profit_margin': profit_margin
    }


def format_duration(minutes: int) -> str:
    """
    Format duration in minutes to human-readable string
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted duration string
    """
    if minutes < 60:
        return f"{minutes}m"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if remaining_minutes == 0:
        return f"{hours}h"
    else:
        return f"{hours}h {remaining_minutes}m"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


# ===== MAIN UTILITY CLASS =====

class TravelAgencyUtils:
    """Main utility class combining all utilities"""
    
    def __init__(self):
        """Initialize all utility components"""
        self.currency_converter = CurrencyConverter()
        self.location_manager = LocationManager()
        self.date_utils = DateUtils()
        self.validation_utils = ValidationUtils()
        self.cache_manager = CacheManager()
        self.config_manager = ConfigManager()
    
    def get_currency_info(self, currency_code: str) -> Optional[CurrencyInfo]:
        """Get currency information"""
        # This would typically fetch from an API
        currencies = {
            'USD': CurrencyInfo('USD', '$', 'US Dollar', 1.0, datetime.now()),
            'EUR': CurrencyInfo('EUR', '€', 'Euro', 0.85, datetime.now()),
            'IDR': CurrencyInfo('IDR', 'Rp', 'Indonesian Rupiah', 15000.0, datetime.now())
        }
        return currencies.get(currency_code.upper())
    
    def validate_trip_request(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trip request data
        
        Args:
            trip_data: Trip request data
            
        Returns:
            Validation results
        """
        required_fields = [
            'origin_city', 'destination_city', 'departure_date', 'budget'
        ]
        
        missing_fields = self.validation_utils.validate_required_fields(
            trip_data, required_fields
        )
        
        validation_errors = []
        
        if missing_fields:
            validation_errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        if 'departure_date' in trip_data and 'return_date' in trip_data:
            if not self.date_utils.is_valid_date_range(
                trip_data['departure_date'], trip_data['return_date']
            ):
                validation_errors.append("Invalid date range")
        
        if 'budget' in trip_data:
            if not self.validation_utils.validate_currency_amount(str(trip_data['budget'])):
                validation_errors.append("Invalid budget amount")
        
        return {
            'valid': len(validation_errors) == 0,
            'errors': validation_errors
        }


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example usage of utilities
    utils = TravelAgencyUtils()
    
    # Test currency conversion
    usd_amount = 1000
    idr_amount = utils.currency_converter.convert_amount(usd_amount, 'USD', 'IDR')
    print(f"${usd_amount} = {utils.currency_converter.format_currency(idr_amount, 'IDR')}")
    
    # Test location search
    locations = utils.location_manager.search_locations('paris')
    for location in locations:
        print(f"Found: {location.name}, {location.country}")
    
    # Test date utilities
    date_range = utils.date_utils.get_date_range('2024-12-15', '2024-12-22')
    print(f"Date range: {date_range}")
    
    # Test validation
    trip_data = {
        'origin_city': 'Jakarta',
        'destination_city': 'Paris',
        'departure_date': '2024-12-15',
        'return_date': '2024-12-22',
        'budget': 5000
    }
    
    validation = utils.validate_trip_request(trip_data)
    print(f"Validation result: {validation}")
    
    # Test commission calculation
    commission = calculate_commission(5000, 12.0)
    print(f"Commission calculation: {commission}")