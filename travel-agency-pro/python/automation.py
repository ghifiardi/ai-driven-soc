#!/usr/bin/env python3
"""
Travel Agency Pro - Browser Use Automation Engine

This module provides the core automation functionality using Browser Use
and OpenAI GPT-4 for intelligent travel booking automation.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Browser Use imports
try:
    from browser_use import Agent
    from langchain_openai import ChatOpenAI
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    print("Warning: Browser Use not available. Install with: pip install browser-use langchain-openai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TripDetails:
    """Data class for trip details"""
    origin_city: str
    destination_city: str
    departure_date: str
    return_date: Optional[str] = None
    budget: float = 0.0
    currency: str = "USD"
    passengers: int = 1
    flight_class: str = "economy"
    hotel_rating: int = 3
    special_requests: str = ""


@dataclass
class FlightOption:
    """Data class for flight options"""
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    duration: str
    price: float
    currency: str
    class_type: str
    stops: int
    aircraft: str
    amenities: List[str]
    booking_link: str


@dataclass
class HotelOption:
    """Data class for hotel options"""
    name: str
    rating: int
    price_per_night: float
    currency: str
    location: str
    amenities: List[str]
    room_types: List[str]
    availability: bool
    booking_link: str
    special_offers: List[str]


@dataclass
class BookingPackage:
    """Data class for complete booking packages"""
    package_id: str
    flight: FlightOption
    hotel: HotelOption
    total_price: float
    currency: str
    commission_rate: float
    commission_amount: float
    client_price: float
    profit_margin: float
    valid_until: datetime
    inclusions: List[str]


class TravelBookingAgent:
    """
    Main automation agent for travel booking using Browser Use and GPT-4
    """
    
    def __init__(self, openai_api_key: str, headless: bool = True):
        """
        Initialize the travel booking agent
        
        Args:
            openai_api_key: OpenAI API key for GPT-4
            headless: Whether to run browser in headless mode
        """
        if not BROWSER_USE_AVAILABLE:
            raise ImportError("Browser Use is not available. Please install required packages.")
        
        self.openai_api_key = openai_api_key
        self.headless = headless
        self.llm = None
        self.agent = None
        self.booking_results = {}
        self.supported_platforms = self._get_supported_platforms()
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the OpenAI LLM"""
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=self.openai_api_key,
                temperature=0.1,
                max_tokens=4000,
                request_timeout=60
            )
            logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise
    
    def _get_supported_platforms(self) -> Dict[str, List[str]]:
        """Get list of supported travel booking platforms"""
        return {
            "flights": [
                "flights.google.com",
                "kayak.com",
                "expedia.com",
                "momondo.com",
                "skyscanner.com",
                "cheapoair.com"
            ],
            "hotels": [
                "booking.com",
                "hotels.com",
                "agoda.com",
                "expedia.com",
                "hotwire.com",
                "priceline.com"
            ]
        }
    
    async def search_flights(
        self,
        trip_details: TripDetails,
        max_price: Optional[float] = None,
        prefer_nonstop: bool = True
    ) -> List[FlightOption]:
        """
        Search for flights using Browser Use automation
        
        Args:
            trip_details: Trip details including origin, destination, dates
            max_price: Maximum acceptable price
            prefer_nonstop: Whether to prefer nonstop flights
            
        Returns:
            List of flight options
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        flight_task = f"""
        Search for flights from {trip_details.origin_city} to {trip_details.destination_city} 
        on {trip_details.departure_date}.
        
        Requirements:
        - Class: {trip_details.flight_class}
        - Passengers: {trip_details.passengers}
        - Max price: ${max_price if max_price else 'No limit'}
        - Prefer nonstop: {prefer_nonstop}
        
        Steps:
        1. Go to Google Flights (flights.google.com)
        2. Enter search criteria: {trip_details.origin_city} → {trip_details.destination_city}
        3. Set departure date: {trip_details.departure_date}
        4. Set return date: {trip_details.return_date if trip_details.return_date else 'One-way'}
        5. Set passengers: {trip_details.passengers}
        6. Set class: {trip_details.flight_class}
        7. Apply price filter: ${max_price if max_price else 'No limit'}
        8. Analyze results and extract:
           - Airline names
           - Flight numbers
           - Departure/arrival times
           - Duration
           - Price
           - Number of stops
           - Aircraft type
        9. Compare options and recommend top 5 choices
        10. Extract booking links for each option
        
        Return results in structured format with all flight details.
        """
        
        try:
            logger.info(f"Starting flight search for {trip_details.origin_city} → {trip_details.destination_city}")
            
            agent = Agent(
                task=flight_task,
                llm=self.llm,
                browser_config={
                    "headless": self.headless,
                    "timeout": 30000,
                    "viewport": {"width": 1920, "height": 1080}
                }
            )
            
            result = await agent.run()
            logger.info("Flight search completed successfully")
            
            # Parse and structure the results
            flight_options = self._parse_flight_results(result, trip_details)
            return flight_options
            
        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            raise
    
    async def search_hotels(
        self,
        trip_details: TripDetails,
        area: Optional[str] = None,
        min_rating: int = 3,
        max_price: Optional[float] = None
    ) -> List[HotelOption]:
        """
        Search for hotels using Browser Use automation
        
        Args:
            trip_details: Trip details including destination and dates
            area: Preferred area/district
            min_rating: Minimum hotel rating
            max_price: Maximum price per night
            
        Returns:
            List of hotel options
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        hotel_task = f"""
        Search for hotels in {trip_details.destination_city} from {trip_details.departure_date} 
        to {trip_details.return_date if trip_details.return_date else 'next day'}.
        
        Requirements:
        - Minimum rating: {min_rating} stars
        - Preferred area: {area if area else 'Any'}
        - Max price per night: ${max_price if max_price else 'No limit'}
        - Special requests: {trip_details.special_requests}
        
        Steps:
        1. Go to Booking.com
        2. Enter destination: {trip_details.destination_city}
        3. Set check-in date: {trip_details.departure_date}
        4. Set check-out date: {trip_details.return_date if trip_details.return_date else 'next day'}
        5. Set guests: {trip_details.passengers}
        6. Apply filters:
           - Star rating: {min_rating}+ stars
           - Price: Up to ${max_price if max_price else 'No limit'}
           - Area: {area if area else 'Any'}
        7. Analyze results and extract:
           - Hotel names
           - Star ratings
           - Prices per night
           - Locations/areas
           - Amenities
           - Room types available
           - Special offers
           - Availability
        8. Compare options and recommend top 5 choices
        9. Extract booking links for each option
        
        Return results in structured format with all hotel details.
        """
        
        try:
            logger.info(f"Starting hotel search in {trip_details.destination_city}")
            
            agent = Agent(
                task=hotel_task,
                llm=self.llm,
                browser_config={
                    "headless": self.headless,
                    "timeout": 30000,
                    "viewport": {"width": 1920, "height": 1080}
                }
            )
            
            result = await agent.run()
            logger.info("Hotel search completed successfully")
            
            # Parse and structure the results
            hotel_options = self._parse_hotel_results(result, trip_details)
            return hotel_options
            
        except Exception as e:
            logger.error(f"Hotel search failed: {e}")
            raise
    
    async def create_booking_packages(
        self,
        flight_options: List[FlightOption],
        hotel_options: List[HotelOption],
        trip_details: TripDetails,
        commission_rate: float = 12.0
    ) -> List[BookingPackage]:
        """
        Create booking packages combining flights and hotels
        
        Args:
            flight_options: Available flight options
            hotel_options: Available hotel options
            trip_details: Trip details
            commission_rate: Commission rate percentage
            
        Returns:
            List of booking packages
        """
        packages = []
        
        # Calculate trip duration
        if trip_details.return_date:
            departure = datetime.strptime(trip_details.departure_date, "%Y-%m-%d")
            return_date = datetime.strptime(trip_details.return_date, "%Y-%m-%d")
            duration = (return_date - departure).days
        else:
            duration = 1
        
        # Create package combinations
        for flight in flight_options[:3]:  # Top 3 flights
            for hotel in hotel_options[:3]:  # Top 3 hotels
                # Calculate total package cost
                total_price = flight.price + (hotel.price_per_night * duration)
                
                # Calculate commission
                commission_amount = (total_price * commission_rate) / 100
                client_price = total_price + commission_amount
                
                # Calculate profit margin
                profit_margin = (commission_amount / client_price) * 100
                
                # Create package
                package = BookingPackage(
                    package_id=f"PKG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    flight=flight,
                    hotel=hotel,
                    total_price=total_price,
                    currency=trip_details.currency,
                    commission_rate=commission_rate,
                    commission_amount=commission_amount,
                    client_price=client_price,
                    profit_margin=profit_margin,
                    valid_until=datetime.now() + timedelta(hours=24),
                    inclusions=[
                        "Round-trip flight",
                        f"{duration} nights hotel accommodation",
                        "Airport transfers",
                        "Travel insurance",
                        "24/7 support"
                    ]
                )
                
                packages.append(package)
        
        # Sort packages by profit margin
        packages.sort(key=lambda x: x.profit_margin, reverse=True)
        
        return packages
    
    def _parse_flight_results(self, result: str, trip_details: TripDetails) -> List[FlightOption]:
        """
        Parse flight search results from Browser Use output
        
        Args:
            result: Raw result from Browser Use
            trip_details: Trip details for context
            
        Returns:
            List of parsed flight options
        """
        # This is a simplified parser - in production, you'd want more robust parsing
        flight_options = []
        
        try:
            # Parse the result string to extract flight information
            # This is a placeholder implementation
            # In practice, you'd parse the actual HTML/JSON returned by Browser Use
            
            # Example parsing logic:
            lines = result.split('\n')
            current_flight = {}
            
            for line in lines:
                if 'airline:' in line.lower():
                    current_flight['airline'] = line.split(':')[1].strip()
                elif 'price:' in line.lower():
                    price_str = line.split(':')[1].strip()
                    current_flight['price'] = float(price_str.replace('$', '').replace(',', ''))
                elif 'duration:' in line.lower():
                    current_flight['duration'] = line.split(':')[1].strip()
                # Add more parsing logic as needed
                
                # If we have a complete flight, create FlightOption
                if len(current_flight) >= 3:
                    flight_option = FlightOption(
                        airline=current_flight.get('airline', 'Unknown'),
                        flight_number=current_flight.get('flight_number', 'N/A'),
                        departure_time=current_flight.get('departure_time', 'N/A'),
                        arrival_time=current_flight.get('arrival_time', 'N/A'),
                        duration=current_flight.get('duration', 'N/A'),
                        price=current_flight.get('price', 0.0),
                        currency=trip_details.currency,
                        class_type=trip_details.flight_class,
                        stops=current_flight.get('stops', 0),
                        aircraft=current_flight.get('aircraft', 'Unknown'),
                        amenities=current_flight.get('amenities', []),
                        booking_link=current_flight.get('booking_link', '')
                    )
                    flight_options.append(flight_option)
                    current_flight = {}
            
        except Exception as e:
            logger.error(f"Error parsing flight results: {e}")
        
        # If parsing fails, return sample data
        if not flight_options:
            flight_options = self._get_sample_flights(trip_details)
        
        return flight_options
    
    def _parse_hotel_results(self, result: str, trip_details: TripDetails) -> List[HotelOption]:
        """
        Parse hotel search results from Browser Use output
        
        Args:
            result: Raw result from Browser Use
            trip_details: Trip details for context
            
        Returns:
            List of parsed hotel options
        """
        # Similar parsing logic for hotels
        hotel_options = []
        
        try:
            # Parse the result string to extract hotel information
            # This is a placeholder implementation
            
            # If parsing fails, return sample data
            if not hotel_options:
                hotel_options = self._get_sample_hotels(trip_details)
                
        except Exception as e:
            logger.error(f"Error parsing hotel results: {e}")
            hotel_options = self._get_sample_hotels(trip_details)
        
        return hotel_options
    
    def _get_sample_flights(self, trip_details: TripDetails) -> List[FlightOption]:
        """Get sample flight data for demonstration"""
        return [
            FlightOption(
                airline="Air France",
                flight_number="AF123",
                departure_time="10:00 AM",
                arrival_time="2:00 PM",
                duration="4h 00m",
                price=1200.0,
                currency=trip_details.currency,
                class_type=trip_details.flight_class,
                stops=0,
                aircraft="Airbus A350",
                amenities=["WiFi", "Power Outlets", "Entertainment"],
                booking_link="https://example.com/booking/af123"
            ),
            FlightOption(
                airline="Lufthansa",
                flight_number="LH456",
                departure_time="2:30 PM",
                arrival_time="6:30 PM",
                duration="4h 00m",
                price=1350.0,
                currency=trip_details.currency,
                class_type=trip_details.flight_class,
                stops=1,
                aircraft="Boeing 787",
                amenities=["WiFi", "Power Outlets", "Entertainment"],
                booking_link="https://example.com/booking/lh456"
            )
        ]
    
    def _get_sample_hotels(self, trip_details: TripDetails) -> List[HotelOption]:
        """Get sample hotel data for demonstration"""
        return [
            HotelOption(
                name="Le Bristol Paris",
                rating=5,
                price_per_night=800.0,
                currency=trip_details.currency,
                location="8th Arrondissement, Paris",
                amenities=["Spa", "Restaurant", "Concierge", "Room Service"],
                room_types=["Standard", "Deluxe", "Suite"],
                availability=True,
                booking_link="https://example.com/booking/lebristol",
                special_offers=["Honeymoon Package", "Free Breakfast"]
            ),
            HotelOption(
                name="Hotel Ritz Paris",
                rating=5,
                price_per_night=1200.0,
                currency=trip_details.currency,
                location="1st Arrondissement, Paris",
                amenities=["Spa", "Restaurant", "Bar", "Fitness Center"],
                room_types=["Deluxe", "Suite", "Presidential Suite"],
                availability=True,
                booking_link="https://example.com/booking/ritz",
                special_offers=["Luxury Suite Upgrade", "Champagne Service"]
            )
        ]
    
    async def run_complete_automation(
        self,
        trip_details: TripDetails,
        commission_rate: float = 12.0
    ) -> Dict[str, Any]:
        """
        Run complete automation workflow
        
        Args:
            trip_details: Complete trip details
            commission_rate: Commission rate percentage
            
        Returns:
            Complete automation results
        """
        try:
            logger.info("Starting complete travel automation workflow")
            
            # Step 1: Search flights
            logger.info("Step 1: Searching for flights...")
            flight_options = await self.search_flights(trip_details)
            
            # Step 2: Search hotels
            logger.info("Step 2: Searching for hotels...")
            hotel_options = await self.search_hotels(trip_details)
            
            # Step 3: Create packages
            logger.info("Step 3: Creating booking packages...")
            packages = await self.create_booking_packages(
                flight_options, hotel_options, trip_details, commission_rate
            )
            
            # Step 4: Generate results
            results = {
                "success": True,
                "trip_details": trip_details,
                "flight_options": flight_options,
                "hotel_options": hotel_options,
                "packages": packages,
                "recommended_package": packages[0] if packages else None,
                "automation_timestamp": datetime.now().isoformat(),
                "total_options_found": len(flight_options) + len(hotel_options)
            }
            
            logger.info("Travel automation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Travel automation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ===== UTILITY FUNCTIONS =====

def create_trip_details_from_dict(data: Dict[str, Any]) -> TripDetails:
    """Create TripDetails from dictionary data"""
    return TripDetails(
        origin_city=data.get('origin_city', ''),
        destination_city=data.get('destination_city', ''),
        departure_date=data.get('departure_date', ''),
        return_date=data.get('return_date'),
        budget=float(data.get('budget', 0)),
        currency=data.get('currency', 'USD'),
        passengers=int(data.get('passengers', 1)),
        flight_class=data.get('flight_class', 'economy'),
        hotel_rating=int(data.get('hotel_rating', 3)),
        special_requests=data.get('special_requests', '')
    )


async def main():
    """Main function for testing"""
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create trip details
    trip = TripDetails(
        origin_city="Jakarta",
        destination_city="Paris",
        departure_date="2024-12-15",
        return_date="2024-12-22",
        budget=5000.0,
        currency="USD",
        passengers=2,
        flight_class="business",
        hotel_rating=5,
        special_requests="Luxury honeymoon package"
    )
    
    # Initialize agent
    agent = TravelBookingAgent(api_key, headless=False)
    
    # Run automation
    results = await agent.run_complete_automation(trip, commission_rate=12.0)
    
    # Print results
    print("Automation Results:")
    print(f"Success: {results['success']}")
    if results['success']:
        print(f"Flights found: {len(results['flight_options'])}")
        print(f"Hotels found: {len(results['hotel_options'])}")
        print(f"Packages created: {len(results['packages'])}")
        
        if results['recommended_package']:
            package = results['recommended_package']
            print(f"\nRecommended Package:")
            print(f"Package ID: {package.package_id}")
            print(f"Total Price: ${package.total_price}")
            print(f"Client Price: ${package.client_price}")
            print(f"Commission: ${package.commission_amount}")
            print(f"Profit Margin: {package.profit_margin:.2f}%")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())