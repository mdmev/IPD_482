import numpy as np
import pygame
import sys
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time

# Parámetros del robot
r = 0.05  # radio de las ruedas en metros
R = 0.1  # distancia del centro a las ruedas en metros
robot_radius = 70  # radio del cuerpo del robot en pixeles
theta_1 = 0
theta_2 = 2 * np.pi / 3  # 120 grados en radianes
theta_3 = 4 * np.pi / 3  # 240 grados en radianes
WHEEL_WIDTH = 40
WHEEL_HEIGHT = 20
# Añadimos una lista para guardar la trayectoria del robot
trajectory = []


def get_velocities(t):
    if t < 5 :
        # Configuración para avanzar hacia adelante
        v1 = -50
        v2 = 0
        v3 = 50
    else:
        # Configuración para girar
        v1, v2, v3 = 0.2, 0.2, 0.2
    return v1, v2, v3


def draw_wheel(surface, color, center, robot_angle, wheel_angle, width, height):
    # Creamos una superficie para dibujar la rueda
    wheel_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    # Dibujamos el rectángulo en la superficie
    pygame.draw.rect(wheel_surface, color, wheel_surface.get_rect())
    # Rotamos la superficie. Ajustamos el ángulo en 90 grados para compensar la orientación inicial del rectángulo.
    rotated_wheel = pygame.transform.rotate(wheel_surface, -np.degrees(robot_angle + wheel_angle) + 90)
    # Obtenemos el nuevo rectángulo después de la rotación para colocar correctamente el centro
    rotated_rect = rotated_wheel.get_rect(center=center)
    # Bliteamos la rueda rotada en la superficie principal
    surface.blit(rotated_wheel, rotated_rect.topleft)

def draw_robot(screen, x, y, angle, v1, v2, v3):
    robot_color = (255, 0, 0)  # color rojo
    wheel_color_moving = (0, 255, 0)  # color amarillo para ruedas en movimiento
    wheel_color_stopped = (255, 0, 0)  # color verde para ruedas detenidas
    
    # Dibujamos el cuerpo del robot
    pygame.draw.circle(screen, robot_color, (int(x), int(y)), robot_radius)
    
    # Ángulos de las ruedas (orientación relativa al robot)
    wheel_angles = [theta_1, theta_2, theta_3]
    
    # Posiciones de las ruedas
    wheel_positions = [(x + robot_radius * np.cos(theta + angle), y + robot_radius * np.sin(theta + angle)) for theta in wheel_angles]
    
    # Velocidades para el color de las ruedas
    velocities = [v1, v2, v3]
    
    # Dibujamos las ruedas
    for wheel_pos, wheel_angle, v in zip(wheel_positions, wheel_angles, velocities):
        color = wheel_color_moving if abs(v) > 0.01 else wheel_color_stopped
        draw_wheel(screen, color, wheel_pos, angle, wheel_angle, WHEEL_WIDTH, WHEEL_HEIGHT)

def cinemática_directa(v1, v2, v3):
    # Matriz de transformación según la cinemática externa dada
    M = np.array([
        [-np.sin(theta_1), -np.sin(theta_1 + theta_2), -np.sin(theta_1 + theta_3)],
        [np.cos(theta_1), np.cos(theta_1 + theta_2), np.cos(theta_1 + theta_3)],
        [1/(3*R), 1/(3*R), 1/(3*R)]
    ])
    vels = np.array([v1, v2, v3])
    x_dot, y_dot, theta_dot = np.dot(M, vels)
    return x_dot, y_dot, theta_dot

def robot_dynamics(state, t):
    x, y, theta = state
    v1, v2, v3 = get_velocities(t)
    x_dot, y_dot, theta_dot = cinemática_directa(v1, v2, v3)
    return [x_dot, y_dot, theta_dot]


# Modificamos la función run_simulation para que guarde la posición del robot
def run_simulation():
    pygame.init()
    size = 1200, 1200
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Robot Omnidireccional")
    black = 0, 0, 0
    initial_conditions = [size[0] // 2, size[1] // 2, 0]
    t = np.linspace(0, 10, 1000)
    states = odeint(robot_dynamics, initial_conditions, t)
    for i, state in enumerate(states):
        x, y, theta = state
        print(f"Posición actual: ({x}, {y})")
        print(f"Ángulo actual: {theta}")
        # Guardamos la posición actual en la lista trajectory
        trajectory.append((x, y))
        t_current = t[i]
        v1, v2, v3 = get_velocities(t_current)
        x = np.clip(x, robot_radius, size[0] - robot_radius)
        y = np.clip(y, robot_radius, size[1] - robot_radius)
        screen.fill(black)
        draw_robot(screen, x, y, theta, v1, v2, v3)
        pygame.display.flip()
        pygame.time.delay(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

# Función para actualizar el plot en tiempo real
def update_plot(i):
    x_vals, y_vals = zip(*trajectory)
    plt.cla()
    plt.plot(x_vals, y_vals, label='Trajectory')
    plt.scatter(x_vals[-1], y_vals[-1], color='red') 
    plt.xlim(0, 1200)
    plt.ylim(0, 1200)
    plt.gca().invert_yaxis()  # Invertimos el eje y para que coincida con pygame
    plt.legend()

# Función para ejecutar la simulación y el plot en threads separados
def run_simulation_with_plot():
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()
    ani = FuncAnimation(plt.gcf(), update_plot, interval=100)
    plt.show()
    simulation_thread.join()


def main():
    run_simulation_with_plot()


if __name__ == "__main__":
    main()