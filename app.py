# app.py - AgriMinds Flask API
# Complete backend with Firebase & CatBoost Integration (Direct Passwordless Login)

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import numpy as np
import pandas as pd
import json
import requests
import os
import math
from datetime import datetime
from catboost import CatBoostClassifier, CatBoostRegressor

# ══════════════════════════════════════════════════
# FIREBASE SETUP
# ══════════════════════════════════════════════════
import firebase_admin
from firebase_admin import credentials, db

if not firebase_admin._apps:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agriminds-a7aea-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

print("✅ Firebase connected!")

app = Flask(__name__)
app.secret_key = 'agriminds_secret_key_2026'

WEATHER_API_KEY = '721c5f58efe0fd4a484bbda9c82427d0'
WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather'

# ══════════════════════════════════════════════════
# CATBOOST MODEL LOADING
# ══════════════════════════════════════════════════
print("Loading AgriMinds CatBoost ML models...")
model1 = CatBoostClassifier()
model1.load_model('models/fertilizer_model.cbm')

model2 = CatBoostRegressor()
model2.load_model('models/amount_model.cbm')

model3 = CatBoostClassifier()
model3.load_model('models/timing_model.cbm')
print("✅ CatBoost models loaded successfully!")

# ══════════════════════════════════════════════════
# FIREBASE HELPER FUNCTIONS
# ══════════════════════════════════════════════════

def get_user_from_firebase(phone):
    try:
        ref = db.reference(f'users/{phone}')
        return ref.get()
    except Exception as e:
        print(f"Firebase get user error: {e}")
        return None

def save_user_to_firebase(phone, user_data):
    try:
        ref = db.reference(f'users/{phone}')
        ref.set(user_data)
        return True
    except Exception as e:
        print(f"Firebase save user error: {e}")
        return False

def get_sensor_data_from_firebase():
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: db.reference('readings').get())
            try:
                data = future.result(timeout=6)
            except concurrent.futures.TimeoutError:
                raise Exception('Firebase read timed out after 6s')

        print(f"🔍 DEBUG - Raw sensor data type: {type(data)}")

        if data is None:
            raise Exception("No sensor data in Firebase yet")

        latest = None

        if isinstance(data, list):
            readings = [r for r in data if r is not None]
            if not readings:
                raise Exception("Empty readings list")
            latest = readings[-1]
            print(f"🔍 DEBUG - Found latest reading from list: {latest}")

        elif isinstance(data, dict):
            sensor_keys = {'nitrogen','phosphorus','potassium','ec','humidity',
                           'ph','soil_moisture','n','p','k','N','P','K',
                           'temperature','temp'}
            if sensor_keys & set(data.keys()):
                latest = data
                print(f"🔍 DEBUG - Data is a single reading: {latest}")
            else:
                keys = sorted(data.keys())
                print(f"🔍 DEBUG - Found {len(keys)} timestamped readings, keys: {keys[-3:]}")
                for key in reversed(keys):
                    entry = data[key]
                    if isinstance(entry, dict) and len(entry) > 0:
                        latest = entry
                        print(f"🔍 DEBUG - Latest reading from key '{key}': {latest}")
                        break

        if latest is None:
            raise Exception("Could not extract latest reading")

        temperature_val = 0.0
        if isinstance(data, dict):
            for key in reversed(sorted(data.keys())):
                entry = data[key]
                if isinstance(entry, dict) and 'temperature' in entry:
                    try:
                        temperature_val = float(entry['temperature'])
                        print(f"🌡️ Found temperature {temperature_val} in key {key}")
                        break
                    except:
                        pass

        def get_val(d, *keys, default=0.0):
            for k in keys:
                if k in d:
                    try:
                        val = float(d[k])
                        print(f"🔍 DEBUG - Found {k}={val}")
                        return val
                    except:
                        pass
            print(f"🔍 DEBUG - No value found for {keys}, using default={default}")
            return default

        result = {
            'success':         True,
            'n':               get_val(latest, 'nitrogen', 'Nitrogen', 'N', 'n'),
            'p':               get_val(latest, 'phosphorus', 'Phosphorus', 'P', 'p'),
            'k':               get_val(latest, 'potassium', 'Potassium', 'K', 'k'),
            'moisture':        get_val(latest, 'soil_moisture', 'moisture', 'Moisture', 'soilMoisture'),
            'ph':              get_val(latest, 'ph', 'pH', 'soil_ph', 'soilPH', default=6.5),
            'ec':              get_val(latest, 'ec', 'EC', 'electrical_conductivity', default=0.0),
            'temperature':     temperature_val if temperature_val > 0 else get_val(latest, 'temperature', 'Temperature', 'temp', 'Temp', default=0.0),
            'humidity':        get_val(latest, 'humidity', 'Humidity', 'hum', 'Hum', default=0.0),
            'days_since_fert': int(get_val(latest, 'days_since_fert', 'daysSinceFert', 'days', default=3)),
            'timestamp':       latest.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'source':          'firebase_live'
        }
        
        print(f"✅ DEBUG - Returning sensor data: {result}")
        return result

    except Exception as e:
        print(f"⚠️ Firebase sensor read error: {e}")
        return {'success': False, 'error': str(e), 'source': 'error'}

def send_valve_command_to_firebase(valve, state):
    try:
        ref = db.reference('commands')
        ref.set({
            'valve':     valve,
            'state':     state,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status':    'pending'
        })
        return True
    except Exception as e:
        print(f"Firebase valve command error: {e}")
        return False

# ══════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json()
                phone = data.get('phone', '').strip()
            else:
                phone = request.form.get('phone', '').strip()

            if len(phone) != 10 or not phone.isdigit():
                return jsonify({'success': False, 'message': 'Invalid phone number. Must be 10 digits.'})

            session['phone'] = phone
            user = get_user_from_firebase(phone)
            
            if user:
                session['name'] = user.get('name', 'Farmer')
                return jsonify({'success': True, 'redirect': '/dashboard'})
            else:
                session['name'] = 'Farmer'
                return jsonify({'success': True, 'redirect': '/register'})
                
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
            
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        phone = data.get('phone', '').strip()
        
        if len(phone) != 10 or not phone.isdigit():
            return jsonify({'success': False, 'message': 'Invalid phone number'})
        
        sowing_date = data.get('sowing_date', '').strip()
        
        user_data = {
            'name':        data.get('name'),
            'phone':       phone,
            'state':       data.get('state'),
            'city':        data.get('city', ''),
            'land_size':   data.get('land_size'),
            'crop_type':   data.get('crop_type'),
            'soil_type':   data.get('soil_type'),
            'sowing_date': sowing_date if sowing_date else '',
            'registered':  datetime.now().isoformat()
        }
        
        save_user_to_firebase(phone, user_data)
        session['phone'] = phone
        session['name']  = data.get('name')
        
        print(f"✅ New user saved to Firebase without OTP verification: {data.get('name')} ({phone})")
        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        print(f"❌ Register error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/dashboard')
def dashboard():
    if 'phone' not in session:
        return redirect(url_for('login'))
    phone = session['phone']
    user  = get_user_from_firebase(phone) or {}
    
    sowing_date_value = user.get('sowing_date', '')
    
    return render_template('dashboard.html',
        user_name    = user.get('name', 'Farmer'),
        farm_size    = user.get('land_size', '--'),
        crop_type    = user.get('crop_type', 'Rice'),
        sowing_date  = sowing_date_value,
        soil_type    = user.get('soil_type', 'Red_Loam'),
        city         = user.get('city', 'Tiruchirappalli')
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build raw DataFrame matching new training column names exactly
        row = pd.DataFrame([{
            'Crop_Type'            : data.get('crop_type', 'Rice'),
            'Growth_Stage'         : data.get('growth_stage', 'Vegetative'),
            'Soil_Type'            : data.get('soil_type', 'Red_Loam'),
            'Soil_pH'              : float(data.get('soil_ph', 6.5)),
            'EC_dS_m'              : float(data.get('ec', 0.3)),
            'Soil_Moisture_pct'    : float(data.get('soil_moisture', 50)),
            'Temperature_C'        : float(data.get('temperature', 28)),
            'Humidity_pct'         : float(data.get('humidity', 60)),
            'Rainfall_7d_mm'       : float(data.get('rainfall', 0)),
            'Forecast_Rain_24h_mm' : float(data.get('forecast_rain', 0)),
            'Days_Since_Last_Fert' : int(data.get('days_since_fert', 3)),
            'N_Soil_mg_kg'         : float(data.get('n_soil', 0)),
            'P_Soil_mg_kg'         : float(data.get('p_soil', 0)),
            'K_Soil_mg_kg'         : float(data.get('k_soil', 0))
        }])

        # Model 1 — Fertilizer Recommendation (Y1)
        probs = model1.predict_proba(row)[0]
        classes = model1.classes_
        fert_idx = int(np.argmax(probs))
        fert_name = str(classes[fert_idx])
        confidence = float(probs[fert_idx]) * 100

        if fert_name in ['No_Fertigation_Needed', 'Delay_Fertigation']:
            return jsonify({
                'success': True,
                'fertilizer': fert_name,
                'confidence': round(confidence, 1),
                'amount': 0,
                'days': 0,
                'minutes': 0,
                'sessions': 0
            })

        # Model 2 — Amount in kg/ha (Y2)
        amount = float(model2.predict(row)[0])
        amount = max(0.1, round(amount, 2))

        # Model 3 — Apply Within Days (Y3)
        days = int(model3.predict(row)[0])

        # Hydraulics & Drip system calculations based on user land profiles
        farm_ha = float(data.get('farm_size', 0.5))
        effective_kg = amount / 0.85
        total_grams = effective_kg * 1000 * farm_ha
        sessions = math.ceil(total_grams / 800)
        minutes = math.ceil(((total_grams / sessions) / 4.0 / 400) * 60)

        return jsonify({
            'success': True,
            'fertilizer': fert_name,
            'confidence': round(confidence, 1),
            'amount': amount,
            'days': days,
            'minutes': minutes,
            'sessions': sessions,
            'grams': round(total_grams, 1)
        })
    except Exception as e:
        print(f"❌ Model Inference Calculation Fault: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/weather')
def weather():
    try:
        city = request.args.get('city', 'Tiruchirappalli')
        response = requests.get(WEATHER_API_URL, params={
            'q': city + ',IN', 'appid': WEATHER_API_KEY, 'units': 'metric'
        }, timeout=5)

        weather_data = response.json()

        if response.status_code == 200:
            return jsonify({
                'success':     True,
                'temperature': round(weather_data['main']['temp'], 1),
                'humidity':    weather_data['main']['humidity'],
                'description': weather_data['weather'][0]['description'].title(),
                'rainfall':    weather_data.get('rain', {}).get('1h', 0),
                'city':        weather_data['name']
            })
        else:
            raise Exception(f"API error {response.status_code}")

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug-firebase')
def debug_firebase():
    try:
        ref  = db.reference('readings')
        data = ref.get()
        return jsonify({
            'type':    str(type(data)),
            'data':    data,
            'is_none': data is None
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/sensor-data')
def sensor_data():
    return jsonify(get_sensor_data_from_firebase())

@app.route('/valve', methods=['POST'])
def valve():
    try:
        data      = request.get_json()
        valve_key = data.get('valve')
        state     = data.get('state')
        db.reference(f'commands/{valve_key}').set(state)
        return jsonify({'success': True, 'valve': valve_key, 'state': state})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════
# DATA CONTEXT HANDLERS
# ══════════════════════════════════════════════════

def get_user_context():
    phone = session.get('phone', '')
    user  = get_user_from_firebase(phone) or {}
    return {
        'user_name':   user.get('name', 'Farmer'),
        'farm_size':   user.get('land_size', '--'),
        'crop_type':   user.get('crop_type', 'Rice'),
        'sowing_date': user.get('sowing_date', ''),
        'soil_type':   user.get('soil_type', 'Red_Loam'),
        'city':        user.get('city', 'Tiruchirappalli'),
        'state':       user.get('state', ''),
        'phone':       phone,
    }

@app.route('/history')
def history():
    if 'phone' not in session:
        return redirect(url_for('login'))
    ctx = get_user_context()
    try:
        ref  = db.reference('readings')
        data = ref.order_by_key().limit_to_last(20).get()
        readings = []
        if isinstance(data, dict):
            for key in sorted(data.keys(), reverse=True):
                r = data[key]
                if r: readings.append(r)
        elif isinstance(data, list):
            readings = [r for r in reversed(data) if r][:20]
    except:
        readings = []
    return render_template('history.html', readings=readings, **ctx)

@app.route('/irrigation')
def irrigation():
    if 'phone' not in session:
        return redirect(url_for('login'))
    ctx = get_user_context()
    try:
        ref   = db.reference('commands')
        cmds  = ref.get() or {}
    except:
        cmds  = {}
    return render_template('irrigation.html', commands=cmds, **ctx)

@app.route('/my-farm')
def my_farm():
    if 'phone' not in session:
        return redirect(url_for('login'))
    ctx = get_user_context()
    return render_template('my_farm.html', **ctx)

@app.route('/my-farm/update', methods=['POST'])
def update_farm():
    if 'phone' not in session:
        return jsonify({'success': False})
    try:
        phone = session['phone']
        data  = request.get_json()
        ref   = db.reference(f'users/{phone}')
        ref.update({
            'name':        data.get('name'),
            'city':        data.get('city'),
            'state':       data.get('state'),
            'land_size':   data.get('land_size'),
            'crop_type':   data.get('crop_type'),
            'soil_type':   data.get('soil_type'),
            'sowing_date': data.get('sowing_date'),
        })
        session['name'] = data.get('name')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/alerts')
def alerts():
    if 'phone' not in session:
        return redirect(url_for('login'))
    ctx = get_user_context()
    try:
        ref        = db.reference('alerts')
        data       = ref.order_by_key().limit_to_last(50).get()
        alerts_list = []
        if isinstance(data, dict):
            for key in sorted(data.keys(), reverse=True):
                a = data[key]
                if a: alerts_list.append(a)
    except:
        alerts_list = []
    return render_template('alerts.html', alerts_list=alerts_list, **ctx)

@app.route('/alerts/log', methods=['POST'])
def log_alert():
    try:
        data = request.get_json()
        ref  = db.reference('alerts')
        ref.push({
            'message':   data.get('message'),
            'type':      data.get('type', 'warning'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'phone':     session.get('phone', '')
        })
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False})

@app.route('/settings')
def settings():
    if 'phone' not in session:
        return redirect(url_for('login'))
    ctx = get_user_context()
    return render_template('settings.html', **ctx)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
