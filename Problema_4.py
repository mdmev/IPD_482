import numpy as np
import pygame
from scipy.integrate import odeint

# Parámetros del robot uniciclo
masa_robot = 1.0  # masa del robot (kg)
momento_inercia = 0.1  # momento de inercia del robot (kg*m^2)
radio_ruedas = 0.1  # radio de las ruedas (m)
distancia_ruedas = 0.2  # distancia entre las ruedas (m)

# Colores para los diferentes perfiles de torque
colores_torque = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

def ecuaciones_movimiento(estado, t, Tl, Tr):
    x, y, theta, vx, vy, omega = estado
    Fl = 0.1 * vx  # Fricción l
    Fr = 0.1 * vx  # Fricción r
    dx = vx * np.cos(theta) - vy * np.sin(theta)
    dy = vx * np.sin(theta) + vy * np.cos(theta)
    dtheta = omega
    dvx = (Tl + Tr) / (masa_robot * radio_ruedas) - (Fl + Fr) / masa_robot
    dvy = 0
    domega = (Tr - Tl) * distancia_ruedas / (2 * momento_inercia * radio_ruedas)
    return [dx, dy, dtheta, dvx, dvy, domega]

def get_torque(t):
    if t < 1:
        return 0.3, 0.3
    elif t < 2:
        return -0.6, -0.6
    elif t < 5:
        return -0.7, 0
    else:
        return -0.5, 0.5

def integrate(t, estado_inicial):
    solucion = odeint(lambda z, t: ecuaciones_movimiento(z, t, *get_torque(t)), estado_inicial, t)
    return solucion

def simular_movimiento(solucion):
    x, y, theta = solucion[:, 0], solucion[:, 1], solucion[:, 2]
    return x, y, theta

def configurar_pantalla():
    pygame.init()
    width, heigth = 2000, 1000
    pantalla = pygame.display.set_mode((width, heigth))
    pygame.display.set_caption("Simulación")
    return pantalla

def dibujar_trayectoria(pantalla, x, y, frame, escala, offset_x, offset_y):
    for i in range(frame):
        pygame.draw.line(pantalla, (0, 0, 255), (x[i] * escala + offset_x, y[i] * escala + offset_y),
                         (x[i+1] * escala + offset_x, y[i+1] * escala + offset_y), 2)

def determinar_color_robot(frame, t):
    if frame < len(t):
        if t[frame] < 1:
            color = colores_torque[0]
        elif t[frame] < 2:
            color = colores_torque[1]
        elif t[frame] < 3:
            color = colores_torque[2]
        else:
            color = colores_torque[3]
    else:
        color = colores_torque[-1]
    return color

def draw_robot(pantalla, x, y, theta, frame, escala, offset_x, offset_y, color):
    robot_x, robot_y = x[frame] * escala + offset_x, y[frame] * escala + offset_y
    pygame.draw.circle(pantalla, color, (int(robot_x), int(robot_y)), 20)
    pygame.draw.line(pantalla, (0, 255, 0), (robot_x, robot_y),
                     (robot_x + 30 * np.cos(theta[frame]), robot_y + 30 * np.sin(theta[frame])), 3)

def main():
    # Condiciones iniciales y tiempo de simulación
    init_states = [0, 0, 0, 0, 0, 0]
    t_sim = 7 
    dt = 0.01  
    t = np.arange(0, t_sim, dt)

    solucion = integrate(t, init_states)
    x, y, theta = simular_movimiento(solucion)
    pantalla = configurar_pantalla()
    clock = pygame.time.Clock()

    # Escala y desplazamiento para la visualización
    escala = 100
    offset_x, offset_y = pantalla.get_width() // 2, pantalla.get_height() // 2

    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pantalla.fill((0, 0, 0))

        dibujar_trayectoria(pantalla, x, y, frame, escala, offset_x, offset_y)
        print(f"Posición actual: ({x[frame]}, {y[frame]})")
        print(f"Ángulo actual: {theta[frame]}")
        color_robot = determinar_color_robot(frame, t)

        draw_robot(pantalla, x, y, theta, frame, escala, offset_x, offset_y, color_robot)

        pygame.display.flip()
        clock.tick(60)
        frame = (frame + 1) % len(x)

    pygame.quit()

if __name__ == "__main__":
    main()