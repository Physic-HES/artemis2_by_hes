import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from astropy.time import Time
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec

# Intentar importar tqdm para barra de progreso, si no definir dummy
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs): self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass

# --- CONFIGURACIÓN Y CARGA DE DATOS ---
DATA_PATH = 'OEM_2026.04.02_post_USS_to_EI_v2.asc'
MU_EARTH = 398600.4415
MU_MOON = 4902.8000
MIN_ACC, MAX_ACC = -1.2, 0.5
ACC_WINDOW_HOURS = 1.0
STEP = 4  # Saltear datos para ir el doble de rápido

# Cargar telemetría
data = np.loadtxt(DATA_PATH, skiprows=20, dtype=str)
artemis_t = [datetime.fromisoformat(t) for t in data[:,0]]
artemis_vec = data[:,1:].astype(float)
artemis_t_sec = np.array([(t - artemis_t[0]).total_seconds() for t in artemis_t])

# Cargar aceleración de motores (generada previamente)
aceleracion_motores_raw = np.loadtxt('aceleracion_motores.csv', delimiter='\t')
interp_aceleracion = interp1d(artemis_t_sec, aceleracion_motores_raw, axis=0, kind='nearest', fill_value='extrapolate')

# Posición de la Luna
moon_coord = coord.get_body('moon', time=Time(artemis_t)).represent_as('cartesian')
pos_luna_km = moon_coord.xyz.to_value(u.km).T 
interp_luna = interp1d(artemis_t_sec, pos_luna_km, axis=0, kind='cubic')

# --- DEFINICIÓN DE LA FÍSICA ---
def cowell_eom(t, estado):
    r_nave, v_nave = estado[0:3], estado[3:6]
    t_clamped = np.clip(t, artemis_t_sec[0], artemis_t_sec[-1])
    
    r_luna = interp_luna(t_clamped)
    a_motor = interp_aceleracion(t_clamped)
    
    d_nave = np.linalg.norm(r_nave)
    d_luna = np.linalg.norm(r_luna)
    d_rel = np.linalg.norm(r_nave - r_luna)
    
    a_tierra = -MU_EARTH * r_nave / (d_nave**3)
    a_luna = -MU_MOON * ((r_nave - r_luna) / (d_rel**3) + r_luna / (d_luna**3))
    a_total = a_tierra + a_luna + 1.04361 * a_motor # Factor de corrección que compensa la presion de radiacion solar y otros efectos no modelados
    
    return np.concatenate((v_nave, a_total))

# --- PRECALCULO DE PREDICCIÓN ---
n_total = len(artemis_t_sec)
predicted_states = np.zeros_like(artemis_vec)
predicted_states[0] = artemis_vec[0]

quarter_time = artemis_t_sec[-1] / 4
start_idx = 0
pbar = tqdm(total=n_total-1, desc='Integrando trayectoria')

while start_idx < n_total - 1:
    end_time = min(artemis_t_sec[start_idx] + quarter_time, artemis_t_sec[-1])
    end_idx = np.searchsorted(artemis_t_sec, end_time)
    if end_idx <= start_idx: end_idx = start_idx + 1
    
    sol = solve_ivp(cowell_eom, (artemis_t_sec[start_idx], artemis_t_sec[end_idx]), 
                    predicted_states[start_idx], method='DOP853', rtol=1e-10, atol=1e-12, t_eval=artemis_t_sec[start_idx:end_idx+1])
    
    predicted_states[start_idx:start_idx+len(sol.t)] = sol.y.T
    pbar.update(len(sol.t)-1)
    start_idx += len(sol.t)-1
pbar.close()

# --- PREPARACIÓN DE GRÁFICOS ---
plt.ion()
fig = plt.figure(figsize=(16, 8))
if hasattr(fig.canvas.manager, 'window'):
    try: fig.canvas.manager.window.showMaximized()
    except: pass

gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_dist = fig.add_subplot(gs[0, 1])
ax_vel = fig.add_subplot(gs[0, 2])
ax_acel = fig.add_subplot(gs[1, 1:])

time_hours = artemis_t_sec / 3600
dist_tele = np.linalg.norm(artemis_vec[:, :3], axis=1) / 1000.0
dist_pred = np.linalg.norm(predicted_states[:, :3], axis=1) / 1000.0
vel_tele = np.linalg.norm(artemis_vec[:, 3:6], axis=1)
vel_pred = np.linalg.norm(predicted_states[:, 3:6], axis=1)
moon_mm = pos_luna_km / 1000.0

# Plot 3D estático
ax3d.plot(artemis_vec[:,0]/1000, artemis_vec[:,1]/1000, artemis_vec[:,2]/1000, 'b--', label='Telemetría', linewidth=2)
ax3d.plot(predicted_states[:,0]/1000, predicted_states[:,1]/1000, predicted_states[:,2]/1000, 'orange', label='Integración', linewidth=2)
ax3d.plot(moon_mm[:,0], moon_mm[:,1], moon_mm[:,2], 'gray')
ax3d.plot(0, 0, 0, 'g.', markersize=20, label='Tierra')
ax3d.set_xlabel('x [Mm]'); ax3d.set_ylabel('y [Mm]'); ax3d.set_zlabel('z [Mm]')
ax3d.set_ylim(-475, 125)

# Elementos móviles 3D
point_moon, = ax3d.plot([], [], [], 'k.', label='Luna')
point_capsula, = ax3d.plot([], [], [], 'ro', label='Orion')
ax3d.legend(loc='upper left')

# Plots 2D estáticos
ax_dist.plot(time_hours, dist_tele, 'b--', linewidth=2)
ax_dist.plot(time_hours, dist_pred, 'orange', linewidth=2)
dot_dist, = ax_dist.plot([], [], 'ro')
ax_dist.set_title('Distancia [Mm]')

ax_vel.plot(time_hours, vel_tele, 'b--', linewidth=2)
ax_vel.plot(time_hours, vel_pred, 'orange', linewidth=2)
dot_vel, = ax_vel.plot([], [], 'ro')
ax_vel.set_title('Velocidad [km/s]')

# Aceleración (Ventana móvil)
line_ax, = ax_acel.plot([], [], 'r', label='a_x')
line_ay, = ax_acel.plot([], [], 'g', label='a_y')
line_az, = ax_acel.plot([], [], 'b', label='a_z')
ax_acel.set_ylim(MIN_ACC, MAX_ACC)
ax_acel.legend(loc='upper right')
ax_acel.set_title('Encendidos de motores [m/s²]')

fig.tight_layout()

# Control de animación
stop_animation = False
fig.canvas.mpl_connect('key_press_event', lambda e: globals().update(stop_animation=True) if e.key == 'q' else None)

# --- BUCLE DE ANIMACIÓN ---
for i in range(0, n_total, STEP):
    # Actualizar 3D
    point_moon.set_data([moon_mm[i, 0]], [moon_mm[i, 1]])
    point_moon.set_3d_properties([moon_mm[i, 2]])
    point_capsula.set_data([predicted_states[i, 0]/1000], [predicted_states[i, 1]/1000])
    point_capsula.set_3d_properties([predicted_states[i, 2]/1000])
    
    # Actualizar 2D
    dot_dist.set_data([time_hours[i]], [dist_pred[i]])
    dot_vel.set_data([time_hours[i]], [vel_pred[i]])
    
    # Actualizar Ventana de Aceleración (última hora)
    w_start = max(0, i - int(ACC_WINDOW_HOURS * 3600 / (artemis_t_sec[1]-artemis_t_sec[0]))) # Aproximación
    # Mejor usar búsqueda por tiempo para precisión
    idx_window = np.where((time_hours >= time_hours[i] - ACC_WINDOW_HOURS) & (time_hours <= time_hours[i]))[0]
    
    line_ax.set_data(time_hours[idx_window], aceleracion_motores_raw[idx_window, 0] * 1000)
    line_ay.set_data(time_hours[idx_window], aceleracion_motores_raw[idx_window, 1] * 1000)
    line_az.set_data(time_hours[idx_window], aceleracion_motores_raw[idx_window, 2] * 1000)
    
    ax_acel.set_xlim(time_hours[i] - ACC_WINDOW_HOURS, time_hours[i])
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
    if stop_animation: break

plt.ioff()
plt.show()
