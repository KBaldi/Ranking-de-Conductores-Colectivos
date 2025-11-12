
import streamlit as st
import geopandas as gpd, pandas as pd, numpy as np, folium, matplotlib.pyplot as plt
from streamlit_folium import st_folium
from shapely.geometry import Point
import contextily as cx
from io import BytesIO
import base64

st.set_page_config(page_title="Ranking Conductores", layout="wide")
st.title("🚌 Ranking de Suavidad de Conducción")
st.markdown("Subí tus archivos GPS y recorridos para obtener el ranking **instantáneo**.")

gps_file = st.file_uploader("GPS (.gpkg)", type="gpkg")
rec_file = st.file_uploader("Recorridos (.gpkg)", type="gpkg")

if gps_file and rec_file:
    gps = gpd.read_file(gps_file)
    rec = gpd.read_file(rec_file)
    linea = gps['linea'].value_counts().index[0]
    gps = gps[gps['linea'] == linea].copy()
    rec = rec[rec['linea'] == linea].copy()
    st.success(f"Línea detectada: {linea}")

    if gps.crs.is_geographic:
        gps = gps.to_crs(epsg=9265)
    if rec.crs != gps.crs:
        rec = rec.to_crs(gps.crs)

    gps['x'] = gps.geometry.x; gps['y'] = gps.geometry.y
    gps = gps.sort_values(['vehiculo', 'fecha']).reset_index(drop=True)
    gps['dx'] = gps.groupby('vehiculo')['x'].diff()
    gps['dy'] = gps.groupby('vehiculo')['y'].diff()
    gps['dt_s'] = gps.groupby('vehiculo')['fecha'].diff().dt.total_seconds()
    gps['dist_m'] = (gps['dx']**2 + gps['dy']**2)**0.5
    gps['vel_ms'] = gps['dist_m'] / gps['dt_s'].replace(0, np.nan)
    gps['acc_ms2'] = gps.groupby('vehiculo')['vel_ms'].diff() / gps['dt_s']

    p95_pos = gps['acc_ms2'].quantile(0.95)
    p95_neg = gps['acc_ms2'].quantile(0.05)
    gps['evento'] = (gps['acc_ms2'] > p95_pos).astype(int) * 1 + (gps['acc_ms2'] < p95_neg).astype(int) * -1

    rec_dict = rec.rename(columns={'recorrido': 'ramal'}).set_index(['linea', 'ramal', 'sentido'])['geometry'].to_dict()
    gps = gpd.sjoin_nearest(gps, rec.rename(columns={'recorrido': 'ramal'})[['linea', 'ramal', 'sentido', 'geometry']], how='left')
    def prog(row):
        geom = rec_dict.get((row.linea_left, row.ramal_left, row.sentido))
        if geom is None: return np.nan
        return geom.project(geom.interpolate(geom.project(row.geometry))) / geom.length
    gps['progresiva'] = gps.apply(prog, axis=1)
    long_media = rec.geometry.length.mean()
    mask = gps['progresiva'].notna()
    gps.loc[mask, 'tramo_id'] = (gps.loc[mask, 'progresiva'] * long_media / 100).astype(int)

    tramo_cond = gps[gps['tramo_id'].notna()].groupby(['tramo_id', 'vehiculo'], as_index=False).agg(
        eventos=('evento', lambda x: (x != 0).sum()),
        dist_km=('dist_m', lambda d: d.sum() / 1000)
    ).assign(eventos_km=lambda d: d['eventos'] / d['dist_km'].replace(0, np.nan))
    cond_stats = tramo_cond.groupby('vehiculo').agg(eventos_km=('eventos_km', 'mean')).reset_index()
    media_f, std_f = cond_stats['eventos_km'].mean(), cond_stats['eventos_km'].std()
    cond_stats['z_score'] = (cond_stats['eventos_km'] - media_f) / std_f
    cond_stats['anomalo'] = cond_stats['z_score'] > 2

    st.subheader("🏆 Top-10 conductores-anómalos")
    top10 = cond_stats.nlargest(10, 'z_score')[['vehiculo', 'eventos_km', 'z_score']]
    st.dataframe(top10)

    st.subheader("🗺️ Mapa de eventos")
    m = gps.explore(column='evento', cmap=['yellow', 'gray', 'red'], tiles='CartoDB Positron', zoom_start=12)
    st_folium(m, width=700, height=500)

    csv = top10.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ranking_{linea}.csv">📥 Descargar CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
