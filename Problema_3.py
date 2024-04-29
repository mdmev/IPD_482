import numpy as np
import pygame
from scipy.integrate import odeint

# Parámetros del sistema
L0 = 2.5  # Longitud del tractor
L1 = 0.8  # Longitud del trailer
l1 = 3  # Distancia del enganche
v0 = 1.2  # Velocidad del tractor
omega0 = -0.4  # Velocidad angular de las ruedas para el giro

# Tiempo de simulación
t_max = 40.0
dt = 0.1
t = np.arange(0, t_max, dt)

def modelo_cinemáticas(z, t):
    x0, y0, theta0, x1, y1, theta1 = z
    if t < 5:
        v = 0
        omega = 0
    elif t < 12:
        v = v0
        omega = 0
    else:
        v = v0
        omega = omega0
    hitch_x = x0 - (L0/2 + l1) * np.cos(theta0)
    hitch_y = y0 - (L0/2 + l1) * np.sin(theta0)
    omega1 = (v / L1) * np.sin(theta0 - theta1)
    dx0 = v * np.cos(theta0)
    dy0 = v * np.sin(theta0)
    dtheta0 = omega
    dx1 = v * np.cos(theta1) + (hitch_x - x1) / dt
    dy1 = v * np.sin(theta1) + (hitch_y - y1) / dt
    dtheta1 = omega1
    return [dx0, dy0, dtheta0, dx1, dy1, dtheta1]

def draw_system(screen, tractor_pos, trailer_pos, color):
    x1, y1, theta1 = trailer_pos
    x0, y0, theta0 = tractor_pos

    # Dibujamos los cuerpos
    trailer_draw = np.array([[L1/2, L1/2, -L1/2, -L1/2, L1/2], [L1, -L1, -L1, L1, L1]])
    tractor_draw = np.array([[L0/2, L0/2, -L0/2, -L0/2, L0/2], [L0/4, -L0/4, -L0/4, L0/4, L0/4]])
    
    rotation_tractor = np.array([[np.cos(theta0), -np.sin(theta0)], [np.sin(theta0), np.cos(theta0)]])
    rotation_trailer = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
    
    tractor_draw = rotation_tractor @ tractor_draw + np.array([[x0], [y0]])
    trailer_draw = rotation_trailer @ trailer_draw + np.array([[x1], [y1]])
    
    trac_pix = (tractor_draw * 50 + np.array([[1000], [500]])).astype(int)
    trai_pix = (trailer_draw * 50 + np.array([[1000], [500]])).astype(int)

    pygame.draw.polygon(screen, color, trac_pix.T)
    pygame.draw.polygon(screen, color, trai_pix.T)
    
    # Dibujamos el anclaje entre los cuerpos
    hitch_tractor_x = x0 - (L0/2) * np.cos(theta0)
    hitch_tractor_y = y0 - (L0/2) * np.sin(theta0)
    hitch_trailer_x = x1 + (L1/2) * np.cos(theta1)
    hitch_trailer_y = y1 + (L1/2) * np.sin(theta1)
    h_trac_pix = (np.array([hitch_tractor_x, hitch_tractor_y]) * 50 + np.array([1000, 500])).astype(int)
    h_trai_pix = (np.array([hitch_trailer_x, hitch_trailer_y]) * 50 + np.array([1000, 500])).astype(int)
    
    # Se dibuja el anclaje
    pygame.draw.line(screen, (255, 255, 255), h_trac_pix, h_trai_pix, 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((2000, 1000))
    pygame.display.set_caption("Simulación")
    colors = [(255, 255, 255), (0, 255, 0), (135, 206, 235)]
    clock = pygame.time.Clock()
    running = True
    frame = 0
    tractor_trajectory = []
    trailer_trajectory = []
    init_states = [0, 0, 0, -3.15, 0, 0]
    solution = odeint(modelo_cinemáticas, init_states, t)
    tractor_poses = solution[:, :3]
    trailer_poses = solution[:, 3:]
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        if frame < len(tractor_poses):
            tractor_pose = tractor_poses[frame]
            trailer_pose = trailer_poses[frame]
            current_time = frame * dt
            if current_time < 5:
                color = colors[0]
            elif current_time < 12:
                color = colors[1]
            else:
                color = colors[2]
            draw_system(screen, tractor_pose, trailer_pose, color)
            tractor_trajectory.append(tractor_pose[:2])
            trailer_trajectory.append(trailer_pose[:2])
            if len(tractor_trajectory) > 1:
                tractor_trajectory_pixels = (np.array(tractor_trajectory) * 50 + np.array([1000, 500])).astype(int)
                pygame.draw.lines(screen, color, False, tractor_trajectory_pixels, 2)
            if len(trailer_trajectory) > 1:
                trailer_trajectory_pixels = (np.array(trailer_trajectory) * 50 + np.array([1000, 500])).astype(int)
                pygame.draw.lines(screen, color, False, trailer_trajectory_pixels, 2)
            frame += 1
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()
