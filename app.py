import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
from math import radians, cos, sin, asin, sqrt

ACC_URL = "https://raw.githubusercontent.com/Jamo1993/Fysiikka_lopputyo/main/Acceleration2.csv"
LOC_URL = "https://raw.githubusercontent.com/Jamo1993/Fysiikka_lopputyo/main/Location2.csv"

#df_acc = pd.read_csv(ACC_URL)
#df_loc = pd.read_csv(LOC_URL)

st.title("Spurttivartti")

def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance between two points (km). lon/lat in decimal degrees."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371.0
    return c * r


df_acc = pd.read_csv("Acceleration2.csv")
df_loc = pd.read_csv("Location2.csv")

df_loc = df_loc[df_loc["Time (s)"] > 5].reset_index(drop=True)


t_acc = df_acc["Time (s)"].values.astype(float)
z = df_acc["Linear Acceleration z (m/s^2)"].values.astype(float)

t_acc = t_acc - t_acc[0]
dt_acc = float(np.mean(np.diff(t_acc)))
fs_acc = 1.0 / dt_acc
nyq_acc = fs_acc / 2.0

order = 3
cutoff_hz = 1 / 0.3  

z_filt = butter_lowpass_filter(z, cutoff_hz, nyq_acc, order)

crossings = 0
for i in range(len(z_filt) - 1):
    if z_filt[i] * z_filt[i + 1] < 0:
        crossings += 1
steps_filt = crossings / 2.0


N = len(z)
freq = np.fft.fftfreq(N, d=dt_acc)
fft = np.fft.fft(z, N)
psd = (fft * np.conj(fft) / N).real

L = np.arange(1, int(N / 2))  

mask = (freq[L] > 0.8) & (freq[L] < 4.0)
freq_band = freq[L][mask]
psd_band = psd[L][mask]

idx_max = int(np.argmax(psd_band))
f_max = float(freq_band[idx_max])

duration_acc = float(t_acc[-1] - t_acc[0])
steps_fft = f_max * duration_acc
T_step = 1.0 / f_max


df_loc["Distance_calc"] = 0.0

for i in range(len(df_loc) - 1):
    lon1 = float(df_loc.loc[i, "Longitude (°)"])
    lat1 = float(df_loc.loc[i, "Latitude (°)"])
    lon2 = float(df_loc.loc[i + 1, "Longitude (°)"])
    lat2 = float(df_loc.loc[i + 1, "Latitude (°)"])

    df_loc.loc[i + 1, "Distance_calc"] = haversine(lon1, lat1, lon2, lat2)

df_loc["total_distance"] = df_loc["Distance_calc"].cumsum()
total_distance_km = float(df_loc["total_distance"].iloc[-1])

total_time_s = float(df_loc["Time (s)"].iloc[-1] - df_loc["Time (s)"].iloc[0])
avg_speed_ms = (total_distance_km * 1000.0) / total_time_s


steps_final = 410
step_length_cm = (total_distance_km * 1000.0 / steps_final) * 100.0

st.subheader("Tulokset")

st.markdown(f"""
Askelmäärä (Suodatus): {int(steps_filt)} askelta  
Askelmäärä (Fourier): {int(np.round(steps_fft))} askelta  
Keskinopeus: {avg_speed_ms:.2f} m/s  
Kokonaismatka: {total_distance_km:.3f} km  
Askepituus: {step_length_cm:.1f} cm
""")

st.divider()
st.subheader("Suodatettu kiihtyvyysdata (Z)")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(t_acc, z_filt, label="suodatettu Z")
ax1.set_xlabel("Aika (s)")
ax1.set_ylabel("Suodatettu Z (m/s²)")
ax1.set_title("Suodatetun Z-komponentin kiihtyvyysdata")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

st.subheader("Tehospektri (Z)")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(freq[L], psd[L])
ax2.set_xlim(0, 10)
ax2.set_xlabel("Taajuus [Hz]")
ax2.set_ylabel("Teho")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("Reitti kartalla")

lat_center = float(df_loc["Latitude (°)"].mean())
lon_center = float(df_loc["Longitude (°)"].mean())

m = folium.Map(location=[lat_center, lon_center], zoom_start=15)
folium.PolyLine(
    df_loc[["Latitude (°)", "Longitude (°)"]].values,
    color="red",
    weight=3
).add_to(m)

st_folium(m, width=750, height=450)
    