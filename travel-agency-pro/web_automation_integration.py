#!/usr/bin/env python3
"""
Web Interface + AI Automation Integration
This connects the web UI to the real AI automation system
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import os

# Import our AI automation system
from fixed_ai_agents import AIOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebAutomationHandler(BaseHTTPRequestHandler):
    """HTTP handler for web automation integration"""
    
    def __init__(self, *args, orchestrator=None, **kwargs):
        self.orchestrator = orchestrator
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Serve the main HTML file
            try:
                with open('index.html', 'r') as f:
                    html_content = f.read()
                self.wfile.write(html_content.encode())
            except FileNotFoundError:
                self.wfile.write(b"<h1>Travel Agency Pro</h1><p>index.html not found</p>")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def do_POST(self):
        """Handle POST requests for automation"""
        if self.path == '/api/automate':
            self.handle_automation_request()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def handle_automation_request(self):
        """Handle automation requests from web interface"""
        try:
            # Get request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            logger.info(f"Received automation request: {request_data}")
            
            # Start automation in background
            asyncio.create_task(self.run_automation(request_data))
            
            # Send immediate response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'started',
                'message': 'Automation started successfully',
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
    
    async def run_automation(self, request_data: Dict[str, Any]):
        """Run the AI automation"""
        try:
            logger.info(f"Starting automation for {request_data.get('client_id')}")
            
            # Convert web form data to automation format
            automation_request = self.convert_web_to_automation(request_data)
            
            # Run automation
            result = await self.orchestrator.automate_travel_booking(automation_request)
            
            # Store result for web interface to retrieve
            self.orchestrator.results[automation_request['client_id']] = result
            
            logger.info(f"Automation completed for {request_data.get('client_id')}")
            
        except Exception as e:
            logger.error(f"Automation failed: {e}")
            # Store error result
            error_result = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.orchestrator.results[request_data.get('client_id', 'unknown')] = error_result
    
    def convert_web_to_automation(self, web_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert web form data to automation format"""
        return {
            'client_id': web_data.get('client_id', f"CLIENT-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            'origin': web_data.get('originCity', 'Jakarta'),
            'destination': web_data.get('destinationCity', 'Paris'),
            'departure_date': web_data.get('departureDate', '2024-12-15'),
            'return_date': web_data.get('returnDate'),
            'budget': float(web_data.get('totalBudget', 5000)),
            'passengers': int(web_data.get('passengers', 1)),
            'preferences': {
                'flight_class': web_data.get('flightClass', 'Economy'),
                'preferred_airlines': web_data.get('preferredAirline', '').split(',') if web_data.get('preferredAirline') else [],
                'hotel_rating': int(web_data.get('minRating', 3)),
                'preferred_amenities': web_data.get('preferredAmenities', '').split(',') if web_data.get('preferredAmenities') else [],
                'preferred_areas': web_data.get('preferredArea', '').split(',') if web_data.get('preferredArea') else []
            },
            'special_requests': web_data.get('specialRequests', '')
        }
    
    def log_message(self, format, *args):
        """Custom logging for web requests"""
        logger.info(f"Web Request: {format % args}")

class WebAutomationServer:
    """Web server with AI automation integration"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.orchestrator = AIOrchestrator()
        self.server = None
    
    async def start(self):
        """Start the web server with AI automation"""
        logger.info(f"🚀 Starting Web Automation Server on port {self.port}")
        
        # Start AI agents
        logger.info("🤖 Initializing AI agents...")
        await self.orchestrator.start_agents()
        
        # Create custom handler with orchestrator
        def handler_factory(*args, **kwargs):
            return WebAutomationHandler(*args, orchestrator=self.orchestrator, **kwargs)
        
        # Start HTTP server
        self.server = HTTPServer(('localhost', self.port), handler_factory)
        
        logger.info(f"✅ Web server started at http://localhost:{self.port}")
        logger.info("🌐 Open your browser and use the web interface!")
        logger.info("🤖 AI agents are ready for automation")
        
        # Start server in background
        server_task = asyncio.create_task(self.run_server())
        
        return server_task
    
    async def run_server(self):
        """Run the HTTP server"""
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down server...")
            self.server.shutdown()
    
    async def stop(self):
        """Stop the server and agents"""
        logger.info("🛑 Stopping web automation server...")
        
        if self.server:
            self.server.shutdown()
        
        # Stop AI agents
        for agent in self.orchestrator.agents.values():
            await agent.stop()
        
        logger.info("✅ Server stopped")

async def main():
    """Main function"""
    print("🚀 Travel Agency Pro - Web + AI Automation Integration")
    print("=" * 60)
    print("This connects your web interface to REAL AI automation")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("Please set: export OPENAI_API_KEY='your-key'")
        return
    
    print(f"✅ OpenAI API Key configured: {api_key[:20]}...")
    
    # Create and start server
    server = WebAutomationServer(port=8080)
    
    try:
        await server.start()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await server.stop()

if __name__ == "__main__":
    # Run the integrated web + AI automation server
    asyncio.run(main())