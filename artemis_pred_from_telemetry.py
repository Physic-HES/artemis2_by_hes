import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from astropy.time import Time
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import pandas as pd
# from matplotlib.animation import FFMpegWriter

#download ephemeris data from https://www.nasa.gov/missions/artemis/artemis-2/track-nasas-artemis-ii-mission-in-real-time/

#where the downloaded data is saved
data_path = 'OEM_2026.04.02_post_USS_to_EI_v2.asc'

#load in and format data
data = np.loadtxt(data_path,skiprows=20,dtype=str)
artemis_t = [datetime.fromisoformat(t) for t in data[:,0]]
artemis_vec = data[:,1:].astype(float)
artemis_t_sec = [(t-artemis_t[0]).total_seconds() for t in artemis_t]

#get vector magnitudes (distance and speed)
artemis_d = np.linalg.norm(artemis_vec[:,:3],axis=1)
artemis_v = np.linalg.norm(artemis_vec[:,3:],axis=1)

# Limites para los encendidos de los motores (en m/s²)
min_acc = -1.2
max_acc = 0.5

### add moon ###

#get position of moon
moon_coord = coord.get_body('moon',time=Time(artemis_t))
moon_coord = moon_coord.represent_as('cartesian')

# 1. Extraer los arrays de tu código de astropy
# moon_coord ya lo tenés. Lo pasamos a km y hacemos la transpuesta para que quede (N_puntos, 3)
pos_luna_km = moon_coord.xyz.to_value(u.km).T 

# 2. Crear el interpolador para la posición de la Luna
# Usamos tu vector de tiempo en segundos 'artemis_t_sec'
interp_luna = interp1d(artemis_t_sec, pos_luna_km, axis=0, kind='cubic')

# Parámetros gravitacionales (km^3/s^2)
MU_EARTH = 398600.4415
MU_MOON = 4902.8000

# 3. Definir la EDO (Método de Cowell)
def cowell_eom(t, estado):
    # Clamp t to the interpolation range to avoid extrapolation errors
    t_clamped = min(t, artemis_t_sec[-1])
    
    r_nave = estado[0:3]
    v_nave = estado[3:6]
    
    # Posición de la luna interpolada en el instante t
    r_luna = interp_luna(t_clamped)
    
    # Normas (distancias)
    d_nave = np.linalg.norm(r_nave)
    d_luna = np.linalg.norm(r_luna)
    d_relativa = np.linalg.norm(r_nave - r_luna)
    
    # Aceleración gravitatoria de la Tierra (Término central)
    a_tierra = -MU_EARTH * r_nave / (d_nave**3)
    
    # Aceleración gravitatoria de la Luna (Directa + Indirecta)
    a_luna = -MU_MOON * ((r_nave - r_luna) / (d_relativa**3) + r_luna / (d_luna**3))
    
    a_total = a_tierra + a_luna
    
    return np.concatenate((v_nave, a_total))

# 4. Configurar e iniciar la integración

plt.ion()
fig = plt.figure(figsize=(16, 8))
fig.canvas.manager.window.showMaximized()
gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])
ax3d = fig.add_subplot(gs[:, 0], projection='3d')  # ocupa filas 0-1, columna 0
ax_dist = fig.add_subplot(gs[0, 1])  # fila 0, columna 1
ax_vel = fig.add_subplot(gs[0, 2])  # fila 0, columna 2
ax_acel = fig.add_subplot(gs[1, 1:])  # fila 1, columnas 1-2

fig.subplots_adjust(wspace=0.3)

# Variable para detener la animación
stop_animation = False

def on_key(event):
    global stop_animation
    if event.key == 'q':
        stop_animation = True

# Conectar el evento de teclado
fig.canvas.mpl_connect('key_press_event', on_key)

# Preconvertir posiciones de la Luna a megametros
moon_x = moon_coord.x.to_value(u.km) / 1000.0
moon_y = moon_coord.y.to_value(u.km) / 1000.0
moon_z = moon_coord.z.to_value(u.km) / 1000.0

# Plotear elementos fijos en Mm
line_telemetria = ax3d.plot(artemis_vec[:,0] / 1000.0,
                          artemis_vec[:,1] / 1000.0,
                          artemis_vec[:,2] / 1000.0,
                          color='blue', linestyle='--', linewidth=1.5,
                          label='Telemetría')
point_earth = ax3d.plot(0, 0, 0, 'g.', markersize=20, label='Tierra')
# Trayectoria de la luna
ax3d.plot(moon_x, moon_y, moon_z, color='gray', linestyle='-')
ax3d.set_xlabel('x [Mm]')
ax3d.set_ylabel('y [Mm]')
ax3d.set_zlabel('z [Mm]')
ax3d.set_ylim(-475, 125)
# ax3d.legend()  # Remover esta línea para agregar después de todos los elementos

# Configurar el gráfico de distancia (arriba izquierda derecha)
ax_dist.set_xlabel('Tiempo [horas]')
ax_dist.set_ylabel('Distancia [Mm]')
ax_dist.set_title('Distancia de Orion')
line_dist_tele = None
line_dist_pred = None

# Configurar el gráfico de velocidad (arriba derecha derecha)
ax_vel.set_xlabel('Tiempo [horas]')
ax_vel.set_ylabel('Velocidad [km/s]')
ax_vel.set_title('Módulo de la velocidad')
line_vel_tele = None
line_vel_pred = None

# Configurar el gráfico de aceleración de motores (abajo)
ax_acel.set_xlabel('Tiempo [ventana móvil de 1 hora]')
ax_acel.set_ylabel('Encendidos [m/s²]')
line_acc_x = None
line_acc_y = None
line_acc_z = None

# Calcular telemetría completa para plots 2D
dist_tele_full = np.linalg.norm(artemis_vec[:, :3], axis=1) / 1000.0
vel_tele_full = np.linalg.norm(artemis_vec[:, 3:6], axis=1)
time_hours_full = np.array(artemis_t_sec) / 3600

# Calcular tiempo total de la misión y un cuarto
total_time = artemis_t_sec[-1]
quarter_time = total_time / 4

# Ventana móvil de aceleración en horas
acc_window_hours = 1.0

# Inicializar referencias para elementos variables
line_prediccion = None
point_moon = None
point_capsula = None
point_pred_dist = None
point_pred_vel = None
aceleracion_motores = np.zeros((len(artemis_t_sec),3))

# Configurar el video
# metadata = dict(title='Artemis II Prediction', artist='Matplotlib')
# writer = FFMpegWriter(fps=30, metadata=metadata)
# writer.setup(fig, "artemis_animation.mp4", dpi=100)

# print("Iniciando simulación y grabando video (artemis_animation.mp4)...")

for i in range(len(artemis_t_sec)):
    # Calcular tiempo final para la predicción: un cuarto hacia adelante, o hasta el final
    end_time = min(artemis_t_sec[i] + quarter_time, artemis_t_sec[-1])
    
    # Encontrar el índice final j tal que artemis_t_sec[j] <= end_time
    j = i
    while j < len(artemis_t_sec) - 1 and artemis_t_sec[j + 1] <= end_time:
        j += 1
    
    t_span = (artemis_t_sec[i], artemis_t_sec[j])
    t_eval = artemis_t_sec[i:j+1]
    
    # Tomamos la posición y velocidad inicial de tus datos de telemetría
    estado_inicial = np.concatenate((artemis_vec[i,:3], artemis_vec[i,3:]))

    # Ejecutar el integrador DOP853
    solucion_teorica_1 = solve_ivp(cowell_eom, t_span, estado_inicial, 
                                method='DOP853', rtol=1e-8, atol=1e-10, 
                                t_eval=t_eval)

    # Diagnosticar si la integración llega al final
    expected_len = len(t_eval)
    if len(solucion_teorica_1.t) < expected_len:
        print(f"Advertencia: Integración en i={i} no llegó al final. Longitud: {len(solucion_teorica_1.t)} / {expected_len}")
        # Opcional: usar telemetría si falla
        solucion_teorica_1.y = artemis_vec[i:j+1].T  # Pero esto es aproximado

    # Calcular los módulos de velocidad
    vel_pred = solucion_teorica_1.y[3:6]  # [vx, vy, vz] para cada tiempo
    vel_pred_norm = np.linalg.norm(vel_pred, axis=0)  # módulo velocidad predicción

    # Calcular las distancias y diferencias en Mm
    dist_pred = np.linalg.norm(solucion_teorica_1.y[:3], axis=0) / 1000.0  # distancia predicción en Mm

    # Calcular la diferencia de velocidades
    diff_vel = np.linalg.norm(artemis_vec[i:j+1, 3:6].T - solucion_teorica_1.y[3:6], axis=0)  # ||v_tele - v_pred||

    time_hours_pred = time_hours_full[i:j+1]  # tiempos absolutos para plots

    # Calculo de aceleracion de motores por diferencias entre telemetría e integración
    if i < len(artemis_t_sec) - 1 and j > i and solucion_teorica_1.y.shape[1] > 1:
        dt_hours = (time_hours_full[i+1] - time_hours_full[i])
        aceleracion_motores[i,:] = (artemis_vec[i+1, 3:6].T - solucion_teorica_1.y[3:6,1]) / (dt_hours * 3600)  # Convertir de horas a segundos
    else:
        aceleracion_motores[i,:] = 0.0  # No hay siguiente punto o integración insuficiente

    # Ventana móvil de aceleración de 1 hora
    window_start = time_hours_full[i] - acc_window_hours
    window_indices = np.where(time_hours_full >= window_start)[0]
    window_indices = window_indices[window_indices <= i]
    window_times = time_hours_full[window_indices]

    if line_prediccion is None:
        # Primera iteración: plotear los elementos variables
        line_prediccion = ax3d.plot(solucion_teorica_1.y[0] / 1000.0,
                                    solucion_teorica_1.y[1] / 1000.0,
                                    solucion_teorica_1.y[2] / 1000.0,
                                    color='orange', linestyle='-', linewidth=1.5,
                                    label='Integración')
        
        point_moon = ax3d.plot([moon_x[i]], [moon_y[i]], [moon_z[i]], 'k.', label='Luna')
        point_capsula = ax3d.plot([artemis_vec[i,0] / 1000.0],
                                 [artemis_vec[i,1] / 1000.0],
                                 [artemis_vec[i,2] / 1000.0], 'ro', label='Orion')
        
        # Agregar leyenda ordenada
        handles, labels = ax3d.get_legend_handles_labels()
        order = ['Telemetría', 'Integración', 'Tierra', 'Orion', 'Luna']
        ordered_handles = [handles[labels.index(label)] for label in order if label in labels]
        ordered_labels = [label for label in order if label in labels]
        ax3d.legend(ordered_handles, ordered_labels, loc='upper left')
        
        # Plotear las líneas en los gráficos 2D con telemetría completa
        line_dist_tele = ax_dist.plot(time_hours_full, dist_tele_full, color='blue', linestyle='--', label='Telemetría')
        line_dist_pred = ax_dist.plot(time_hours_pred, dist_pred, color='orange', linestyle='-', label='Predicción')
        point_pred_dist = ax_dist.plot([time_hours_pred[0]], [dist_pred[0]], 'ro', markersize=6)[0]
        ax_dist.legend()

        line_vel_tele = ax_vel.plot(time_hours_full, vel_tele_full, color='blue', linestyle='--', label='Telemetría')
        line_vel_pred = ax_vel.plot(time_hours_pred, vel_pred_norm, color='orange', linestyle='-', label='Predicción')
        point_pred_vel = ax_vel.plot([time_hours_pred[0]], [vel_pred_norm[0]], 'ro', markersize=6)[0]
        ax_vel.legend()

        line_acc_x = ax_acel.plot(window_times, aceleracion_motores[window_indices,0] * 1000, color='red', label='a_x')[0]
        line_acc_y = ax_acel.plot(window_times, aceleracion_motores[window_indices,1] * 1000, color='green', label='a_y')[0]
        line_acc_z = ax_acel.plot(window_times, aceleracion_motores[window_indices,2] * 1000, color='blue', label='a_z')[0]
        ax_acel.legend()
        # Ajustar límites del eje X a los datos, Y fijo
        ax_acel.relim()
        ax_acel.autoscale_view(scalex=True, scaley=False)
        ax_acel.set_ylim(min_acc, max_acc)
    else:
        # Iteraciones siguientes: actualizar solo los datos de los elementos variables
        line_prediccion[0].set_data(solucion_teorica_1.y[0] / 1000.0, solucion_teorica_1.y[1] / 1000.0)
        line_prediccion[0].set_3d_properties(solucion_teorica_1.y[2] / 1000.0)

        point_moon[0].set_data([moon_x[i]], [moon_y[i]])
        point_moon[0].set_3d_properties([moon_z[i]])
        point_capsula[0].set_data([artemis_vec[i,0] / 1000.0], [artemis_vec[i,1] / 1000.0])
        point_capsula[0].set_3d_properties([artemis_vec[i,2] / 1000.0])
        
        # Actualizar solo las líneas de predicción en los gráficos 2D
        line_dist_pred[0].set_data(time_hours_pred, dist_pred)
        point_pred_dist.set_data([time_hours_pred[0]], [dist_pred[0]])

        line_vel_pred[0].set_data(time_hours_pred, vel_pred_norm)
        point_pred_vel.set_data([time_hours_pred[0]], [vel_pred_norm[0]])

        line_acc_x.set_data(window_times, aceleracion_motores[window_indices,0] * 1000)
        line_acc_y.set_data(window_times, aceleracion_motores[window_indices,1] * 1000)
        line_acc_z.set_data(window_times, aceleracion_motores[window_indices,2] * 1000)
        # Ajustar límites del eje X a los datos, Y fijo
        ax_acel.relim()
        ax_acel.autoscale_view(scalex=True, scaley=False)
        ax_acel.set_ylim(min_acc, max_acc)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Grabar el frame actual
    # writer.grab_frame()
    
    # if i % 100 == 0:
    #     print(f"Procesado frame {i}/{len(artemis_t_sec)}")

    if stop_animation:
        print("Animación detenida por el usuario.")
        break

# writer.finish()
# print("Video guardado exitosamente como 'artemis_animation.mp4'")
plt.ioff()
# Guardar matriz de aceleracion de motores en un csv con numpy
# np.savetxt('aceleracion_motores.csv', aceleracion_motores, delimiter='\t', comments='')
