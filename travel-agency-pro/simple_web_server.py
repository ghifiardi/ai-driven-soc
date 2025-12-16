#!/usr/bin/env python3
"""
Simple Web Server with AI Automation Integration
This is a working version that serves the web interface and handles automation
"""

import http.server
import socketserver
import json
import os
import asyncio
import threading
from urllib.parse import urlparse, parse_qs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAutomationHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with AI automation capabilities"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.path = '/index.html'
        
        # Serve static files
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        """Handle POST requests for automation"""
        if self.path == '/api/automate':
            self.handle_automation_request()
        else:
            self.send_error(404, "Not Found")
    
    def handle_automation_request(self):
        """Handle automation requests"""
        try:
            # Get request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            logger.info(f"Received automation request: {request_data}")
            
            # Start automation in background thread
            threading.Thread(target=self.run_automation, args=(request_data,), daemon=True).start()
            
            # Send immediate response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            response = {
                'status': 'started',
                'message': 'AI automation started successfully',
                'request_id': request_data.get('client_id', 'unknown')
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Error handling automation request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def run_automation(self, request_data):
        """Run the AI automation"""
        try:
            logger.info(f"Starting automation for {request_data.get('client_id')}")
            
            # Simulate AI automation process
            import time
            time.sleep(2)  # Simulate processing time
            
            # Generate simulated results
            results = self.generate_simulated_results(request_data)
            
            logger.info(f"Automation completed for {request_data.get('client_id')}")
            
            # Store results (in a real app, you'd save to database)
            self.automation_results = results
            
        except Exception as e:
            logger.error(f"Automation failed: {e}")
    
    def generate_simulated_results(self, request_data):
        """Generate simulated automation results"""
        import random
        from datetime import datetime
        
        # Calculate realistic pricing
        base_flight_cost = request_data.get('totalBudget', 5000) * 0.4
        base_hotel_cost = request_data.get('totalBudget', 5000) * 0.3
        
        flight_cost = base_flight_cost * random.uniform(0.8, 1.2)
        hotel_cost_per_night = base_hotel_cost * random.uniform(0.8, 1.2)
        
        # Calculate trip duration
        if request_data.get('returnDate'):
            from datetime import datetime
            departure = datetime.strptime(request_data['departureDate'], '%Y-%m-%d')
            return_date = datetime.strptime(request_data['returnDate'], '%Y-%m-%d')
            duration = (return_date - departure).days
        else:
            duration = 1
        
        total_hotel_cost = hotel_cost_per_night * duration
        total_cost = flight_cost + total_hotel_cost
        
        # Calculate commission
        commission_rate = 0.12
        commission = total_cost * commission_rate
        client_price = total_cost + commission
        profit_margin = (commission / client_price) * 100
        
        return {
            'automation_id': f"AUTO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'client_id': request_data.get('client_id', 'unknown'),
            'status': 'completed',
            'package': {
                'selected_flight': {
                    'airline': random.choice(['Air France', 'Lufthansa', 'Singapore Airlines', 'Qatar Airways']),
                    'flight_number': f"FL{random.randint(100, 999)}",
                    'departure_time': f"{random.randint(6, 22):02d}:{random.randint(0, 59):02d}",
                    'arrival_time': f"{random.randint(6, 22):02d}:{random.randint(0, 59):02d}",
                    'price': round(flight_cost, 2),
                    'class': request_data.get('flightClass', 'Economy'),
                    'duration': f"{random.randint(4, 8)}h {random.randint(0, 59)}m",
                    'stops': random.randint(0, 1)
                },
                'selected_hotel': {
                    'name': random.choice([
                        'Le Bristol Paris', 'Hotel Ritz Paris', 'Shangri-La Hotel Paris',
                        'Four Seasons Hotel George V', 'Le Meurice', 'Hotel Plaza Athénée'
                    ]),
                    'rating': round(random.uniform(4.0, 5.0), 1),
                    'price_per_night': round(hotel_cost_per_night, 2),
                    'location': random.choice(['8th Arrondissement', '1st Arrondissement', '16th Arrondissement']),
                    'amenities': random.sample(['WiFi', 'Pool', 'Spa', 'Restaurant', 'Gym', 'Concierge'], random.randint(3, 5))
                },
                'total_cost': round(total_cost, 2),
                'trip_duration': duration,
                'commission': round(commission, 2),
                'client_price': round(client_price, 2),
                'profit_margin': round(profit_margin, 2)
            },
            'automation_timestamp': datetime.now().isoformat()
        }
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()

def run_server(port=8080):
    """Run the web server"""
    print(f"🚀 Starting Travel Agency Pro Web Server on port {port}")
    print(f"🌐 Open your browser and visit: http://localhost:{port}")
    print("🤖 AI automation integration is ready!")
    print("=" * 60)
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create server
    with socketserver.TCPServer(("", port), AIAutomationHandler) as httpd:
        print(f"✅ Server started successfully on port {port}")
        print("🔄 Press Ctrl+C to stop the server")
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            httpd.shutdown()
            print("✅ Server stopped")

if __name__ == "__main__":
    # Check if OpenAI API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   Some features may not work properly")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print()
    else:
        print(f"✅ OpenAI API Key configured: {api_key[:20]}...")
        print()
    
    # Run the server
    run_server(8080)