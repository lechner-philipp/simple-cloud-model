import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import os
from scipy.sparse.linalg import *
import json


resolution = (40, 62)
dx = 10
dy = 10
g = 9.81
dt = 1

num_cells = resolution[0] * resolution[1]
solver_resolution = 50


def main():
    for filename in os.listdir("output"):
        file_path = os.path.join("output", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    pt = np.ones(resolution) * 300
    Psi = np.zeros(resolution)
    pt += get_initial_temperature_anomaly()

    idx = 0
    t = 0
    nu = 0
    theta_0 = 300

    fig, (ax1, ax2) = plt.subplots(1, 2)
    cfig, (cax1, cax2) = plt.subplots(1, 2)
    current_cbar = None

    energy_conservation = []
    adjusted_energy_conservation = []
    phi_conservation = []
    adjusted_phi = []

    Q0 = get_q()
    E0 = get_potential_energy(Q0 / theta_0)
    phi0 = get_phi_squared(Q0)

    data = {}

    while t < 601:
        xi = get_vorticity(Psi)
        Q = get_q()

        phi = pt_to_phi(pt, theta_0)

        phi[0, :] = phi[2, :]
        phi[-1, :] = phi[-3, :]
        phi[:, 0] = phi[:, 2]
        phi[:, -1] = phi[:, -3]

        xi[0, :] = -xi[2, :]
        xi[-1, :] = -xi[-3, :]
        xi[:, 0] = -xi[:, 2]
        xi[:, -1] = -xi[:, -3]

        xi = update_xi(xi, g, phi, dx, dt)

        xi[0, :] = -xi[2, :]
        xi[-1, :] = -xi[-3, :]
        xi[:, 0] = -xi[:, 2]
        xi[:, -1] = -xi[:, -3]

        A, b = prepare_conditioned_inverse_laplace(xi)

        A_t = np.transpose(A)

        A = np.dot(A_t, A)
        b = np.dot(A_t, b)

        sol = gmres(A, b, custom_flatten(Psi), maxiter=solver_resolution)[0]
        Psi = custom_reshape(sol, resolution)

        phi = update_phi(phi, Psi, Q, theta_0, nu, dx, dt)

        phi[0, :] = phi[2, :]
        phi[-1, :] = phi[-3, :]
        phi[:, 0] = phi[:, 2]
        phi[:, -1] = phi[:, -3]

        pt = phi_to_pt(phi, theta_0)

        print("---")
        print(t)
        print(np.min(pt))
        print(np.median(pt))
        print(np.max(pt))

        kinetic_energy = get_kinetic_energy(Psi)
        potential_energy = get_potential_energy(phi)
        E = kinetic_energy - potential_energy

        energy_conservation.append(E)
        adjusted_energy_conservation.append(E - E0 * t / dt)
        phi_conservation.append(get_phi_squared(phi))
        adjusted_phi.append(get_phi_squared(phi) - phi0 * t / dt)

        data[t] = {
            "min_pt": np.min(pt),
            "median_pt": np.median(pt),
            "max_pt": np.max(pt),
            "phi": phi_conservation[-1],
            "energy": E,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "adjusted_phi": adjusted_phi[-1],
            "adjusted_energy": adjusted_energy_conservation[-1]
        }
        with open("output/data.json", "w+") as f:
            json.dump(data, f)

        ax1.clear()
        display1 = np.array(list(pt))
        display1 = display1.transpose((1, 0))
        heatmap1 = ax1.imshow(display1, vmin=300, vmax=300.7, cmap='seismic', origin='lower')

        ax2.clear()
        display2 = np.array(list(Psi))
        display2 = display2.transpose((1, 0))
        heatmap2 = ax2.imshow(display2, cmap='bone', origin='lower')
        ax2.set_title(f"t = {t}")
        if current_cbar:
            current_cbar.remove()

        cbar = ax2.figure.colorbar(heatmap2, ax=ax2)
        cbar.set_label('Stream function (Psi)')

        cax1.clear()
        cax1.plot(phi_conservation, label="Phi conservation")
        cax1.legend()

        cax2.clear()
        cax2.plot(energy_conservation, label="Energy conservation")
        cax2.legend()

        current_cbar = cbar

        fig.savefig(f"output/{idx}.png")

        cfig.savefig(f"output/c{idx}.png")

        # Pausing to display
        plt.pause(0.1)

        t += dt
        idx += 1


def get_phi_squared(phi):
    total = 0
    for idx in range(1, len(phi) - 1):
        for jdx in range(1, len(phi[idx]) - 1):
            total += phi[idx][jdx] ** 2

    return total


def custom_flatten(arr):
    arr_flat = []
    for y in range(resolution[1]):
        for x in range(resolution[0]):
            arr_flat.append(arr[x][y])
    return arr_flat


def custom_reshape(arr, _shape):
    arr_reshaped = np.zeros(_shape)

    idx = 0
    for y in range(_shape[1]):
        for x in range(_shape[0]):
            arr_reshaped[x][y] = arr[idx]
            idx += 1

    return arr_reshaped


def get_kinetic_energy(_psi):
    v = get_velocity(_psi)
    total = 0
    for d in range(2):
        for idx in range(1, len(v[d]) - 1):
            for jdx in range(1, len(v[d][idx]) - 1):
                total += 0.5 * v[d][idx][jdx] ** 2

    return total


def get_potential_energy(_phi):
    total = 0
    for i in range(len(_phi)):
        for j in range(len(_phi[i])):
            total += _phi[i][j] * g * j * dy
    return total


def ajacobian(psi, xi):
    sol = np.zeros_like(psi)

    for y in range(1, psi.shape[1] - 1):
        for x in range(1, psi.shape[0] - 1):
            sol[x][y] = 1 / (12 * dx * dy) * (
                    (psi[x, y - 1] + psi[x + 1, y - 1] - psi[x, y + 1] - psi[x + 1, y + 1]) * (xi[x + 1, y] + xi[x, y])
                    - (psi[x - 1, y - 1] + psi[x, y - 1] - psi[x - 1, y + 1] - psi[x, y + 1]) * (
                                xi[x, y] + xi[x - 1, y])
                    + (psi[x + 1, y] + psi[x + 1, y + 1] - psi[x - 1, y] - psi[x - 1, y + 1]) * (
                                xi[x, y + 1] + xi[x, y])
                    - (psi[x + 1, y - 1] + psi[x + 1, y] - psi[x - 1, y - 1] - psi[x - 1, y]) * (
                                xi[x, y] + xi[x, y - 1])

                    + (psi[x + 1, y] - psi[x, y + 1]) * (xi[x + 1, y + 1] + xi[x, y])
                    - (psi[x, y - 1] - psi[x - 1, y]) * (xi[x, y] + xi[x - 1, y - 1])
                    + (psi[x, y + 1] - psi[x - 1, y]) * (xi[x - 1, y + 1] + xi[x, y])
                    - (psi[x + 1, y] - psi[x, y - 1]) * (xi[x, y] + xi[x + 1, y - 1])
            )
    return sol


def pt_to_phi(pt, theta_0):
    pt = pt.copy()
    pt -= theta_0
    pt /= theta_0
    return pt


def phi_to_pt(phi, theta_0):
    phi = phi.copy()
    phi *= theta_0
    phi += theta_0
    return phi


def get_initial_temperature_anomaly():
    Q = np.zeros(resolution)
    pi = np.pi

    for x in range(17):
        for y in range(10, 31):
            term1 = 0.5 * np.cos(pi * x / 32)
            term2 = np.cos(pi * (y - 10) / 40) ** 2
            Q[x + 1, y + 1] = term1 * term2
    Q[:, 10] = Q[:, 11] / 2
    return Q


def get_velocity(Psi):
    vx = np.gradient(Psi, axis=1) / dy
    vy = -np.gradient(Psi, axis=0) / dx
    return vx, vy


def get_vorticity(Psi):
    return laplace(Psi) / (dx * dy)


def get_q():
    Q = np.zeros(resolution)
    Q0 = 0.00  # 0.004

    pi = np.pi

    for x in range(17):
        for y in range(8, 13):
            term1 = np.cos(pi * x / 32)
            term2 = np.cos(pi * (y - 10) / 4) ** 2
            Q[x + 1, y + 1] = Q0 * term1 * term2

    return Q


def update_xi(xi, g, phi, dx, dt):
    xi_new = xi + dt * (-g * 2 * np.gradient(phi, axis=-2) / dx)

    return xi_new


def update_phi(phi, Psi, Q, theta_0, nu, dx, dt):
    term1 = ajacobian(Psi, phi)
    term2 = Q / theta_0
    term3 = nu * laplace(phi) / (dx ** 2)

    phi_new = phi + dt * (term1 + term2 + term3)

    return phi_new


def prepare_conditioned_inverse_laplace(_laplace):
    indices = np.zeros(resolution, dtype=int)
    idx = 0
    for y in range(resolution[1]):
        for x in range(resolution[0]):
            indices[x][y] = idx
            idx += 1

    A = np.zeros((num_cells * 4, num_cells))
    b = np.zeros(num_cells * 4)

    for y in range(1, resolution[1] - 1):
        for x in range(1, resolution[0] - 1):
            idx = indices[x][y]

            A[idx][indices[x - 1][y]] += 1
            A[idx][indices[x + 1][y]] += 1

            A[idx][indices[x][y - 1]] += 1
            A[idx][indices[x][y + 1]] += 1

            A[idx][indices[x][y]] += -4

            if x == 1 or x == resolution[0] - 2:
                A[idx + num_cells][indices[x - 1][y]] += 1
                A[idx + num_cells][indices[x][y]] += -2
                A[idx + num_cells][indices[x + 1][y]] += 1

                A[idx + 2 * num_cells][indices[x][y - 1]] += 0.5
                A[idx + 2 * num_cells][indices[x][y + 1]] += -0.5

            if y == 1 or y == resolution[1] - 2:
                A[idx + num_cells][indices[x][y - 1]] += 1
                A[idx + num_cells][indices[x][y]] += -2
                A[idx + num_cells][indices[x][y + 1]] += 1

                A[idx + 2 * num_cells][indices[x - 1][y]] += 0.5
                A[idx + 2 * num_cells][indices[x + 1][y]] += -0.5

            b[idx] = _laplace[x][y] * dx * dy
    return A, b


if __name__ == "__main__":
    main()
