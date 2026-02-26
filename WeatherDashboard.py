"""
╔══════════════════════════════════════════════════════════╗
║       REAL-TIME WEATHER ANALYTICS DASHBOARD              ║
║  Python | REST API | Pandas | Matplotlib | SQLite | ML   ║
╚══════════════════════════════════════════════════════════╝

How to get FREE API Key:
1. Go to https://openweathermap.org/
2. Sign up (free)
3. Go to API Keys section
4. Copy your API key
5. Replace 'YOUR_API_KEY' below with your actual key
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sqlite3
import json
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  CONFIG - Replace with your API key
# ============================================================
API_KEY = "YOUR_API_KEY"  # Get free key from openweathermap.org
BASE_URL = "http://api.openweathermap.org/data/2.5/"

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Hyderabad", "Pune", "Jaipur",
    "Ahmedabad", "Surat"
]

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
#  DATABASE SETUP (SQLite)
# ============================================================
def setup_database():
    """Create SQLite database to store weather data"""
    conn = sqlite3.connect('weather_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city        TEXT,
            temperature REAL,
            feels_like  REAL,
            humidity    INTEGER,
            wind_speed  REAL,
            description TEXT,
            timestamp   TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database setup complete!")

def save_to_db(city, temp, feels_like, humidity, wind, description):
    """Save weather data to SQLite"""
    conn = sqlite3.connect('weather_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO weather (city, temperature, feels_like, humidity, wind_speed, description, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (city, temp, feels_like, humidity, wind, description,
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def load_from_db():
    """Load all weather data from SQLite"""
    conn = sqlite3.connect('weather_data.db')
    df = pd.read_sql_query("SELECT * FROM weather", conn)
    conn.close()
    return df

# ============================================================
#  API - FETCH LIVE WEATHER DATA
# ============================================================
def fetch_weather(city):
    """Fetch real-time weather data from OpenWeatherMap API"""
    try:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code == 200:
            weather = {
                'city'        : city,
                'temperature' : data['main']['temp'],
                'feels_like'  : data['main']['feels_like'],
                'humidity'    : data['main']['humidity'],
                'wind_speed'  : data['wind']['speed'],
                'description' : data['weather'][0]['description'],
                'timestamp'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return weather
        else:
            print(f"  ⚠️  API Error for {city}: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"  ⚠️  Connection error for {city}: {e}")
        return None

def fetch_all_cities():
    """Fetch weather for all cities"""
    print("\n🌍 Fetching live weather data...")
    weather_data = []

    for city in CITIES:
        data = fetch_weather(city)
        if data:
            weather_data.append(data)
            save_to_db(city, data['temperature'], data['feels_like'],
                      data['humidity'], data['wind_speed'], data['description'])
            print(f"  ✅ {city}: {data['temperature']}°C, {data['description']}")
        time.sleep(0.5)  # Rate limiting

    if not weather_data:
        print("  ⚠️  No live data fetched. Using demo data...")
        return generate_demo_data()

    return pd.DataFrame(weather_data)

def generate_demo_data():
    """Generate demo data when API key not set"""
    np.random.seed(42)
    data = []
    for city in CITIES:
        temp = np.random.uniform(20, 42)
        data.append({
            'city'        : city,
            'temperature' : round(temp, 1),
            'feels_like'  : round(temp + np.random.uniform(-3, 3), 1),
            'humidity'    : np.random.randint(40, 95),
            'wind_speed'  : round(np.random.uniform(1, 20), 1),
            'description' : np.random.choice(['clear sky', 'few clouds', 'scattered clouds',
                                               'light rain', 'moderate rain', 'haze']),
            'timestamp'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        save_to_db(data[-1]['city'], data[-1]['temperature'], data[-1]['feels_like'],
                   data[-1]['humidity'], data[-1]['wind_speed'], data[-1]['description'])
    return pd.DataFrame(data)

# ============================================================
#  ANALYSIS 1: Temperature Comparison
# ============================================================
def analysis_temperature(df):
    print("\n📊 Analysis 1: Temperature Comparison...")
    df_sorted = df.sort_values('temperature', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df_sorted['city'], df_sorted['temperature'],
                  color=COLORS[:len(df_sorted)], edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, df_sorted['temperature']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}°C', ha='center', fontweight='bold', fontsize=10)

    ax.set_title('🌡️ Real-Time Temperature Comparison (All Cities)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.axhline(y=df['temperature'].mean(), color='red', linestyle='--',
               alpha=0.7, label=f'Average: {df["temperature"].mean():.1f}°C')
    ax.legend()
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('1_temperature.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 1_temperature.png")

# ============================================================
#  ANALYSIS 2: Humidity vs Temperature
# ============================================================
def analysis_humidity_temp(df):
    print("\n📊 Analysis 2: Humidity vs Temperature...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    scatter = axes[0].scatter(df['temperature'], df['humidity'],
                               c=df['temperature'], cmap='RdYlBu_r',
                               s=200, edgecolors='white', linewidth=1.5)
    for _, row in df.iterrows():
        axes[0].annotate(row['city'], (row['temperature'], row['humidity']),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.colorbar(scatter, ax=axes[0], label='Temperature (°C)')
    axes[0].set_title('Humidity vs Temperature', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Humidity (%)')

    # Humidity bar chart
    df_sorted = df.sort_values('humidity', ascending=False)
    bars = axes[1].barh(df_sorted['city'], df_sorted['humidity'],
                         color=COLORS[:len(df_sorted)], edgecolor='white')
    for bar, val in zip(bars, df_sorted['humidity']):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{val}%', va='center', fontweight='bold', fontsize=9)
    axes[1].set_title('Humidity by City', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Humidity (%)')

    plt.suptitle('💧 Humidity & Temperature Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('2_humidity_temp.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 2_humidity_temp.png")

# ============================================================
#  ANALYSIS 3: Wind Speed
# ============================================================
def analysis_wind(df):
    print("\n📊 Analysis 3: Wind Speed Analysis...")
    df_sorted = df.sort_values('wind_speed', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(df_sorted['city'], df_sorted['wind_speed'],
                  color=COLORS[:len(df_sorted)], edgecolor='white')
    for bar, val in zip(bars, df_sorted['wind_speed']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}', ha='center', fontweight='bold', fontsize=10)

    ax.set_title('💨 Wind Speed Comparison (m/s)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('3_wind_speed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 3_wind_speed.png")

# ============================================================
#  ANALYSIS 4: Weather Conditions Distribution
# ============================================================
def analysis_conditions(df):
    print("\n📊 Analysis 4: Weather Conditions...")
    conditions = df['description'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(conditions.values, labels=conditions.index,
           autopct='%1.1f%%', colors=COLORS[:len(conditions)],
           startangle=90, textprops={'fontsize': 11})
    ax.set_title('🌤️ Weather Conditions Distribution',
                 fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('4_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 4_conditions.png")

# ============================================================
#  ANALYSIS 5: Heat Index (Feels Like)
# ============================================================
def analysis_feels_like(df):
    print("\n📊 Analysis 5: Actual vs Feels Like Temperature...")

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, df['temperature'], width,
                   label='Actual Temp', color='#FF6B6B', edgecolor='white')
    bars2 = ax.bar(x + width/2, df['feels_like'], width,
                   label='Feels Like', color='#4ECDC4', edgecolor='white')

    ax.set_title('🌡️ Actual Temperature vs Feels Like',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['city'], rotation=20, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('5_feels_like.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 5_feels_like.png")

# ============================================================
#  ML: TEMPERATURE PREDICTION
# ============================================================
def ml_temperature_prediction(df):
    print("\n🤖 ML: Temperature Prediction Model...")

    # Generate historical data for training
    historical_data = []
    for _ in range(200):
        humidity  = np.random.randint(30, 100)
        wind      = np.random.uniform(0, 25)
        hour      = np.random.randint(0, 24)
        temp      = 35 - (humidity * 0.15) - (wind * 0.3) + (np.sin(hour/24 * 2 * np.pi) * 5) + np.random.normal(0, 2)
        historical_data.append([humidity, wind, hour, temp])

    hist_df = pd.DataFrame(historical_data, columns=['humidity', 'wind_speed', 'hour', 'temperature'])

    X = hist_df[['humidity', 'wind_speed', 'hour']]
    y = hist_df['temperature']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for current cities
    current_hour = datetime.now().hour
    df['predicted_temp'] = model.predict(
        df[['humidity', 'wind_speed']].assign(hour=current_hour)
    ).round(1)

    mae = mean_absolute_error(df['temperature'], df['predicted_temp'])
    print(f"\n✅ Model MAE: {mae:.2f}°C")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['temperature'], width,
           label='Actual', color='#FF6B6B', edgecolor='white')
    ax.bar(x + width/2, df['predicted_temp'], width,
           label='Predicted', color='#45B7D1', edgecolor='white')

    ax.set_title(f'🤖 Actual vs Predicted Temperature (MAE: {mae:.2f}°C)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['city'], rotation=20, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig('6_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 6_prediction.png")
    return mae

# ============================================================
#  WEATHER ALERTS
# ============================================================
def check_alerts(df):
    print("\n🚨 Weather Alerts:")
    print("-" * 40)
    alerts = False
    for _, row in df.iterrows():
        if row['temperature'] > 40:
            print(f"  🔴 HEAT ALERT    : {row['city']} - {row['temperature']}°C (Extreme Heat!)")
            alerts = True
        elif row['temperature'] < 10:
            print(f"  🔵 COLD ALERT    : {row['city']} - {row['temperature']}°C (Very Cold!)")
            alerts = True
        if row['humidity'] > 85:
            print(f"  💧 HUMIDITY ALERT: {row['city']} - {row['humidity']}% (Very Humid!)")
            alerts = True
        if row['wind_speed'] > 15:
            print(f"  💨 WIND ALERT    : {row['city']} - {row['wind_speed']} m/s (Strong Wind!)")
            alerts = True
    if not alerts:
        print("  ✅ No extreme weather alerts!")

# ============================================================
#  SUMMARY
# ============================================================
def print_summary(df, mae):
    print("\n" + "="*55)
    print("     🌍 WEATHER ANALYTICS SUMMARY")
    print("="*55)
    print(f"  Cities Analyzed       : {len(df)}")
    print(f"  Hottest City          : {df.loc[df['temperature'].idxmax(), 'city']} ({df['temperature'].max()}°C)")
    print(f"  Coolest City          : {df.loc[df['temperature'].idxmin(), 'city']} ({df['temperature'].min()}°C)")
    print(f"  Most Humid            : {df.loc[df['humidity'].idxmax(), 'city']} ({df['humidity'].max()}%)")
    print(f"  Avg Temperature       : {df['temperature'].mean():.1f}°C")
    print(f"  ML Model MAE          : {mae:.2f}°C")
    print(f"  Data Stored in        : weather_data.db (SQLite)")
    print(f"  Timestamp             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

# ============================================================
#  MAIN
# ============================================================
def main():
    print("="*55)
    print("   🌍 REAL-TIME WEATHER ANALYTICS DASHBOARD")
    print("   Python | REST API | Pandas | ML | SQLite")
    print("="*55)

    # Setup database
    setup_database()

    # Fetch live data
    if API_KEY == "YOUR_API_KEY":
        print("\n⚠️  API Key not set — running in DEMO mode!")
        print("   Get free key from: https://openweathermap.org/")
        df = generate_demo_data()
    else:
        df = fetch_all_cities()

    # Run analyses
    analysis_temperature(df)
    analysis_humidity_temp(df)
    analysis_wind(df)
    analysis_conditions(df)
    analysis_feels_like(df)

    # ML
    mae = ml_temperature_prediction(df)

    # Alerts
    check_alerts(df)

    # Summary
    print_summary(df, mae)

    print("\n✅ All analyses complete!")
    print("📁 6 charts saved as PNG files!")
    print("🗄️  Data stored in weather_data.db!")

if __name__ == "__main__":
    main()