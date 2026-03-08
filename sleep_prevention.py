"""
============================================================================
K1RL QUASAR — Sleep Prevention Service
============================================================================
Prevents HuggingFace Spaces from sleeping after 48 hours of inactivity
by self-pinging every 30 minutes and simulating visitor activity
============================================================================
"""

import time
import logging
import requests
import threading
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SLEEP-PREVENTION] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/user/app/logs/sleep_prevention.log', mode='a')
    ]
)
log = logging.getLogger('sleep_prevention')

# Configuration
PING_INTERVAL = 1800  # 30 minutes in seconds
HEALTH_ENDPOINT = "http://localhost:7860/health"
DASHBOARD_ENDPOINT = "http://localhost:7860/"
SPACE_URL = os.environ.get('SPACE_URL', 'https://KarlQuant-k1rl-quasar.hf.space')

def self_ping():
    """Ping our own endpoints to maintain activity"""
    try:
        # Ping health endpoint
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            log.info(f"✅ Health ping successful: {response.status_code}")
        else:
            log.warning(f"⚠️ Health ping returned: {response.status_code}")
        
        # Ping dashboard endpoint  
        response = requests.get(DASHBOARD_ENDPOINT, timeout=10)
        if response.status_code == 200:
            log.info(f"✅ Dashboard ping successful: {response.status_code}")
        else:
            log.warning(f"⚠️ Dashboard ping returned: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        log.error(f"❌ Self-ping failed: {e}")
    except Exception as e:
        log.error(f"❌ Unexpected ping error: {e}")

def external_ping():
    """Ping external Space URL if available"""
    if SPACE_URL and SPACE_URL.startswith('https://'):
        try:
            response = requests.get(f"{SPACE_URL}/health", timeout=15)
            if response.status_code == 200:
                log.info(f"🌐 External ping successful: {SPACE_URL}")
            else:
                log.warning(f"⚠️ External ping returned: {response.status_code}")
        except Exception as e:
            log.debug(f"External ping failed (normal if Space not public): {e}")

def run_prevention_loop():
    """Main sleep prevention loop"""
    log.info("🛡️ Sleep Prevention Service started")
    log.info(f"   Ping interval: {PING_INTERVAL}s ({PING_INTERVAL//60} minutes)")
    log.info(f"   Local health: {HEALTH_ENDPOINT}")
    log.info(f"   Space URL: {SPACE_URL}")
    
    ping_count = 0
    
    while True:
        try:
            ping_count += 1
            log.info(f"🏓 Sleep prevention ping #{ping_count}")
            
            # Ping local endpoints
            self_ping()
            
            # Ping external Space URL
            external_ping()
            
            # Log uptime info
            uptime_hours = (ping_count * PING_INTERVAL) / 3600
            log.info(f"⏱️ Service uptime: {uptime_hours:.1f} hours")
            
        except Exception as e:
            log.error(f"❌ Prevention loop error: {e}")
        
        # Wait until next ping
        time.sleep(PING_INTERVAL)

if __name__ == '__main__':
    run_prevention_loop()
