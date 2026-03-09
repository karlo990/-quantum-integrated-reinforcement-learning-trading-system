#!/usr/bin/env python3
"""
K1RL QUASAR — Visitor Tracking Module (HuggingFace Spaces)
Tracks IPs, locations, devices with container-safe persistence
"""

import os
import json
import time
import threading
import requests
from datetime import datetime
from pathlib import Path

try:
    from user_agents import parse as parse_ua
except ImportError:
    # Fallback if user-agents not available
    def parse_ua(ua_string):
        return type('UA', (), {
            'browser': type('Browser', (), {'family': 'Unknown', 'version_string': ''}),
            'os': type('OS', (), {'family': 'Unknown', 'version_string': ''}),
            'device': type('Device', (), {'family': 'Unknown'}),
            'is_mobile': False,
            'is_tablet': False,
            'is_pc': True,
            'is_bot': False
        })

# ============================================================================
# CONFIGURATION (HuggingFace Spaces paths)
# ============================================================================

BASE_DIR = Path('/home/user/app')
VISITOR_FILE = BASE_DIR / 'data' / 'visitors.json'
VISITOR_TIMEOUT = 120  # seconds before considered inactive

# Ensure data directory exists
VISITOR_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STORAGE
# ============================================================================

# Active visitors: {ip: {'last_seen': timestamp, 'user_agent': str}}
active_visitors = {}

# All-time visitors: {ip: {'first_seen': str, 'last_seen': str, 'visits': int, 'location': dict, 'device': dict}}
all_time_visitors = {}

visitor_lock = threading.Lock()

# ============================================================================
# GEOLOCATION (Free API with fallback)
# ============================================================================

def get_location(ip):
    """Get location from IP using free API with fallbacks"""
    try:
        # Skip local/private IPs
        if ip in ['127.0.0.1', 'localhost'] or ip.startswith(('192.168.', '10.', '172.')):
            return {'country': 'Local', 'city': 'Local', 'region': '', 'country_code': 'LO', 'isp': 'Local'}
        
        # Try ip-api.com (free, 1000 requests per month)
        response = requests.get(
            f'http://ip-api.com/json/{ip}?fields=status,country,countryCode,region,city,isp',
            timeout=3
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return {
                    'country': data.get('country', 'Unknown'),
                    'country_code': data.get('countryCode', '??'),
                    'city': data.get('city', 'Unknown'),
                    'region': data.get('region', ''),
                    'isp': data.get('isp', 'Unknown ISP')
                }
    
    except Exception as e:
        print(f"⚠️ Geolocation API failed for {ip}: {e}")
    
    # Fallback location data
    return {
        'country': 'Unknown',
        'city': 'Unknown',
        'country_code': '??',
        'region': '',
        'isp': 'Unknown ISP'
    }

def parse_device(user_agent_string):
    """Parse user agent to get device info with fallback"""
    try:
        ua = parse_ua(user_agent_string)
        return {
            'browser': f"{ua.browser.family} {ua.browser.version_string}".strip(),
            'os': f"{ua.os.family} {ua.os.version_string}".strip(),
            'device': ua.device.family,
            'is_mobile': ua.is_mobile,
            'is_tablet': ua.is_tablet,
            'is_pc': ua.is_pc,
            'is_bot': ua.is_bot
        }
    except Exception as e:
        print(f"⚠️ User agent parsing failed: {e}")
        # Fallback device info
        return {
            'browser': 'Unknown',
            'os': 'Unknown', 
            'device': 'Unknown',
            'is_mobile': False,
            'is_tablet': False,
            'is_pc': True,
            'is_bot': False
        }

# ============================================================================
# PERSISTENCE (Container-safe)
# ============================================================================

def load_visitors():
    """Load all-time visitors from file"""
    global all_time_visitors
    try:
        if VISITOR_FILE.exists():
            with open(VISITOR_FILE, 'r') as f:
                data = json.load(f)
                all_time_visitors = data.get('visitors', {})
                count = len(all_time_visitors)
                print(f"📊 Loaded {count} historical visitors from {VISITOR_FILE}")
        else:
            print(f"📊 No existing visitor data found at {VISITOR_FILE}")
            all_time_visitors = {}
    except Exception as e:
        print(f"⚠️ Could not load visitor data: {e}")
        all_time_visitors = {}

def save_visitors():
    """Save all-time visitors to file"""
    try:
        visitor_data = {
            'visitors': all_time_visitors,
            'total_count': len(all_time_visitors),
            'last_updated': datetime.now().isoformat(),
            'platform': 'huggingface-spaces'
        }
        
        # Atomic write (write to temp file then rename)
        temp_file = VISITOR_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(visitor_data, f, indent=2)
        
        temp_file.rename(VISITOR_FILE)
        print(f"💾 Saved {len(all_time_visitors)} visitors to {VISITOR_FILE}")
        
    except Exception as e:
        print(f"⚠️ Could not save visitor data: {e}")

# ============================================================================
# TRACKING FUNCTIONS
# ============================================================================

def track_visitor(ip, user_agent=''):
    """Track a visitor by IP with location and device"""
    with visitor_lock:
        now = time.time()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update active visitors
        active_visitors[ip] = {
            'last_seen': now,
            'user_agent': user_agent
        }
        
        # Check if new visitor
        is_new = ip not in all_time_visitors
        
        if is_new:
            print(f"👤 New visitor detected: {ip}")
            
            # Get location and device info (these can be slow, so do it in background)
            def get_visitor_info():
                try:
                    location = get_location(ip)
                    device = parse_device(user_agent)
                    
                    with visitor_lock:
                        all_time_visitors[ip] = {
                            'first_seen': now_str,
                            'last_seen': now_str,
                            'location': location,
                            'device': device,
                            'visits': 1,
                            'user_agent_history': [user_agent] if user_agent else []
                        }
                    
                    save_visitors()
                    count = len(all_time_visitors)
                    city = location.get('city', '?')
                    country = location.get('country', '?')
                    print(f"👤 Visitor #{count}: {ip} from {city}, {country}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing new visitor {ip}: {e}")
            
            # Run in background thread to not slow down the request
            info_thread = threading.Thread(target=get_visitor_info, daemon=True)
            info_thread.start()
        
        else:
            # Update existing visitor
            visitor = all_time_visitors[ip]
            visitor['last_seen'] = now_str
            visitor['visits'] = visitor.get('visits', 0) + 1
            
            # Update user agent history (keep last 5)
            if user_agent:
                ua_history = visitor.setdefault('user_agent_history', [])
                if user_agent not in ua_history:
                    ua_history.append(user_agent)
                    visitor['user_agent_history'] = ua_history[-5:]  # Keep last 5
            
            # Update device info if we have better data now
            if user_agent and visitor.get('device', {}).get('browser') == 'Unknown':
                visitor['device'] = parse_device(user_agent)
            
            # Save periodically (every 10 visits)
            if visitor['visits'] % 10 == 0:
                save_visitors()

def cleanup_inactive():
    """Remove inactive visitors from active list"""
    with visitor_lock:
        now = time.time()
        stale = [ip for ip, data in active_visitors.items() 
                if now - data['last_seen'] > VISITOR_TIMEOUT]
        for ip in stale:
            del active_visitors[ip]

def get_stats():
    """Get basic visitor statistics — bots are excluded from headline counts"""
    cleanup_inactive()
    with visitor_lock:
        active_ips = list(active_visitors.keys())

        # Separate humans from bots using stored device info
        human_active = [ip for ip in active_ips
                        if not all_time_visitors.get(ip, {}).get('device', {}).get('is_bot', False)]
        bot_active   = [ip for ip in active_ips
                        if all_time_visitors.get(ip, {}).get('device', {}).get('is_bot', False)]

        human_total = sum(1 for data in all_time_visitors.values()
                         if not data.get('device', {}).get('is_bot', False))
        bot_total   = len(all_time_visitors) - human_total

        return {
            'active_now':      len(human_active),
            'active_ips':      human_active,
            'all_time_unique': human_total,
            'bots_active':     len(bot_active),
            'bots_total':      bot_total,
        }

def get_detailed_stats():
    """Get detailed visitor statistics — bots excluded from human headline counts"""
    cleanup_inactive()
    with visitor_lock:
        active_ips = list(active_visitors.keys())

        # Countries/cities tracked for humans only; bots counted separately
        countries = {}
        cities    = {}
        devices   = {'mobile': 0, 'tablet': 0, 'pc': 0, 'bot': 0}
        browsers  = {}

        human_active = [ip for ip in active_ips
                        if not all_time_visitors.get(ip, {}).get('device', {}).get('is_bot', False)]
        human_total  = 0

        for ip, data in all_time_visitors.items():
            device_info = data.get('device', {})
            is_bot = device_info.get('is_bot', False)

            # Device bucket
            if is_bot:
                devices['bot'] += 1
            elif device_info.get('is_mobile'):
                devices['mobile'] += 1
            elif device_info.get('is_tablet'):
                devices['tablet'] += 1
            else:
                devices['pc'] += 1

            # Country / city / browser stats — humans only
            if not is_bot:
                human_total += 1
                country = data.get('location', {}).get('country', 'Unknown')
                countries[country] = countries.get(country, 0) + 1

                city = data.get('location', {}).get('city', 'Unknown')
                cities[city] = cities.get(city, 0) + 1

                browser = device_info.get('browser', 'Unknown').split()[0]
                browsers[browser] = browsers.get(browser, 0) + 1

        bot_total = len(all_time_visitors) - human_total

        return {
            'total_unique':  human_total,
            'bots_total':    bot_total,
            'active_now':    len(human_active),
            'bots_active':   len(active_ips) - len(human_active),
            'countries':     dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)[:15]),
            'cities':        dict(sorted(cities.items(),    key=lambda x: x[1], reverse=True)[:15]),
            'devices':       devices,
            'browsers':      dict(sorted(browsers.items(),  key=lambda x: x[1], reverse=True)[:10]),
        }

def _mask_ip(ip):
    """Mask last octet of IPv4 for privacy, or last segment of IPv6"""
    try:
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.•••"
        parts6 = ip.split(':')
        if len(parts6) >= 4:
            return ':'.join(parts6[:4]) + ':••••'
    except Exception:
        pass
    return ip[:8] + '•••'


def get_all_visitors():
    """Get all visitor details with location, device, and masked IP"""
    cleanup_inactive()
    with visitor_lock:
        active_ips  = set(active_visitors.keys())
        active_seen = {ip: active_visitors[ip]['last_seen'] for ip in active_ips}

        visitors_list = []
        for ip, data in all_time_visitors.items():
            device_info = data.get('device', {})
            is_bot      = device_info.get('is_bot', False)
            is_active   = ip in active_ips

            # Calculate seconds since last activity for active visitors
            seconds_ago = None
            if is_active:
                seconds_ago = int(time.time() - active_seen[ip])

            visitors_list.append({
                'ip_masked':        _mask_ip(ip),
                'is_active':        is_active,
                'is_bot':           is_bot,
                'seconds_ago':      seconds_ago,
                'first_seen':       data.get('first_seen', ''),
                'last_seen':        data.get('last_seen', ''),
                'visits':           data.get('visits', 1),
                'location':         data.get('location', {}),
                'device':           device_info,
                'user_agent_count': len(data.get('user_agent_history', []))
            })

        # Sort: active first, then bots last, then by visits desc
        visitors_list.sort(
            key=lambda x: (not x['is_active'], x['is_bot'], -x['visits'])
        )

        human_active = [v for v in visitors_list if v['is_active'] and not v['is_bot']]
        human_total  = sum(1 for v in visitors_list if not v['is_bot'])
        bot_total    = len(visitors_list) - human_total

        return {
            'total_unique':    human_total,
            'bots_total':      bot_total,
            'active_now':      len(human_active),
            'visitors':        visitors_list[:100],
            'showing_count':   min(100, len(visitors_list))
        }

# ============================================================================
# BACKGROUND CLEANUP TASK
# ============================================================================

def background_cleanup():
    """Background task to periodically clean up and save data"""
    while True:
        try:
            time.sleep(300)  # Every 5 minutes
            cleanup_inactive()
            
            # Save data every hour
            current_time = int(time.time())
            if current_time % 3600 < 300:  # Within 5 minutes of the hour
                save_visitors()
                
        except Exception as e:
            print(f"⚠️ Background cleanup error: {e}")

# ============================================================================
# INITIALIZE
# ============================================================================

# Load existing visitor data
load_visitors()

# Start background cleanup task
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

print(f"✅ K1RL QUASAR Visitor Tracker ready")
print(f"   Platform: HuggingFace Spaces")
print(f"   Historical visitors: {len(all_time_visitors)}")
print(f"   Data file: {VISITOR_FILE}")
print("=" * 50)

# Export main functions for use by other modules
__all__ = ['track_visitor', 'get_stats', 'get_detailed_stats', 'get_all_visitors', 'cleanup_inactive']
