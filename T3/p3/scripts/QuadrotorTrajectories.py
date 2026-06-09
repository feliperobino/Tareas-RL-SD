import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class QuadrotorTrajectory:

    _AXES = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, sample_time=0.01):
        self.sample_time = sample_time


    def _circle_pt(self, t, T, scale):
        w = 2.0 * np.pi / T
        return (scale * np.cos(w * t),
                scale * np.sin(w * t),
               -scale * w * np.sin(w * t),
                scale * w * np.cos(w * t))

    def _figure8_pt(self, t, T, scale):
        w = 2.0 * np.pi / T
        a     =  scale * np.sin(w * t)
        b     =  scale * np.sin(w * t) * np.cos(w * t)
        a_dot =  scale * w * np.cos(w * t)
        b_dot =  scale * w * (np.cos(w * t)**2 - np.sin(w * t)**2)
        return a, b, a_dot, b_dot

    def _square_pt(self, t, T, scale):
        seg_T  = T / 4.0
        spd    = scale / seg_T
        ct     = t % T
        st     = ct % seg_T
        seg    = int(np.floor(ct / seg_T))
        pos    = spd * st
        segs = [
            (0.,           pos,          0.,   spd ),  # up
            (-pos,         scale,       -spd,  0.  ),  # left
            (-scale,       scale - pos,  0.,  -spd ),  # down
            (-scale + pos, 0.,           spd,  0.  ),  # right
        ]
        return segs[seg]


    def circle(self, length=10.0, num_cycles=1, plane='xz',
               scaling=0.5, center=(0., 0., 1.)):
        """Circle in the specified plane, centred at `center`."""
        return self._build_planar(self._circle_pt, length, num_cycles,
                                  plane, scaling, center)

    def figure8(self, length=10.0, num_cycles=1, plane='xz',
                scaling=0.5, center=(0., 0., 1.)):
        """Lissajous figure-eight in the specified plane."""
        return self._build_planar(self._figure8_pt, length, num_cycles,
                                  plane, scaling, center)

    def square(self, length=10.0, num_cycles=1, plane='xz',
               scaling=0.5, center=(0., 0., 1.)):
        """Constant-speed square in the specified plane."""
        return self._build_planar(self._square_pt, length, num_cycles,
                                  plane, scaling, center)


    def helix(self, length=10.0, num_cycles=3,
              radius=0.5, rise=1.0, center=(0., 0., 0.5)):
        """Circular motion in xy + linear z rise."""
        times   = self._times(length)
        T       = length / num_cycles
        omega   = 2.0 * np.pi / T
        vz      = rise * num_cycles / length

        pos  = np.zeros((len(times), 3))
        vel  = np.zeros((len(times), 3))
        for i, t in enumerate(times):
            pos[i] = [center[0] + radius * np.cos(omega * t),
                      center[1] + radius * np.sin(omega * t),
                      center[2] + vz * t]
            vel[i] = [-radius * omega * np.sin(omega * t),
                       radius * omega * np.cos(omega * t),
                       vz]
        return pos, vel, np.linalg.norm(vel, axis=1)

    def lemniscate_3d(self, length=10.0, num_cycles=1,
                      scale_xy=0.5, scale_z=0.25, center=(0., 0., 1.)):
        times  = self._times(length)
        T      = length / num_cycles
        w      = 2.0 * np.pi / T

        pos  = np.zeros((len(times), 3))
        vel  = np.zeros((len(times), 3))
        for i, t in enumerate(times):
            pos[i] = [center[0] + scale_xy * np.sin(w * t),
                      center[1] + scale_xy * np.sin(w * t) * np.cos(w * t),
                      center[2] + scale_z  * np.sin(2 * w * t)]
            vel[i] = [scale_xy * w * np.cos(w * t),
                      scale_xy * w * (np.cos(w * t)**2 - np.sin(w * t)**2),
                      scale_z  * 2 * w * np.cos(2 * w * t)]
        return pos, vel, np.linalg.norm(vel, axis=1)

    def spiral_descent(self, length=10.0, num_cycles=4,
                       r0=0.8, r1=0.1, z0=2.0, z1=0.3):

        times  = self._times(length)
        T      = length / num_cycles
        omega  = 2.0 * np.pi / T
        vz     = (z1 - z0) / length
        vr     = (r1 - r0) / length

        pos  = np.zeros((len(times), 3))
        vel  = np.zeros((len(times), 3))
        for i, t in enumerate(times):
            r  = r0 + vr * t
            rd = vr
            pos[i] = [r * np.cos(omega * t),
                      r * np.sin(omega * t),
                      z0 + vz * t]
            vel[i] = [rd * np.cos(omega * t) - r * omega * np.sin(omega * t),
                      rd * np.sin(omega * t) + r * omega * np.cos(omega * t),
                      vz]
        return pos, vel, np.linalg.norm(vel, axis=1)


    def _times(self, length):
        return np.arange(0, length + self.sample_time, self.sample_time)

    def _build_planar(self, pt_fn, length, num_cycles, plane, scaling, center):
        ia  = self._AXES[plane[0]]
        ib  = self._AXES[plane[1]]
        ic  = [k for k in range(3) if k != ia and k != ib][0]
        T   = length / num_cycles
        times = self._times(length)

        pos  = np.zeros((len(times), 3))
        vel  = np.zeros((len(times), 3))
        for i, t in enumerate(times):
            a, b, ad, bd = pt_fn(t, T, scaling)
            pos[i, ia] = a + center[ia]
            pos[i, ib] = b + center[ib]
            pos[i, ic] = center[ic]
            vel[i, ia] = ad
            vel[i, ib] = bd
        return pos, vel, np.linalg.norm(vel, axis=1)



if __name__ == '__main__':

    gen     = QuadrotorTrajectory(sample_time=0.01)
    LENGTH  = 12.0
    t_vec   = gen._times(LENGTH)

    # Generate all trajectories
    trajs = {
        'Circle (xz)':      gen.circle    (length=LENGTH, num_cycles=2, plane='xz', scaling=0.5, center=(0., 0., 1.)),
        'Circle (xy)':      gen.circle    (length=LENGTH, num_cycles=2, plane='xy', scaling=0.5, center=(0., 0., 1.)),
        'Figure-8 (xz)':    gen.figure8   (length=LENGTH, num_cycles=2, plane='xz', scaling=0.5, center=(0., 0., 1.)),
        'Figure-8 (xy)':    gen.figure8   (length=LENGTH, num_cycles=2, plane='xy', scaling=0.5, center=(0., 0., 1.)),
        'Square (xz)':      gen.square    (length=LENGTH, num_cycles=2, plane='xz', scaling=0.5, center=(0., 0., 1.)),
        'Square (xy)':      gen.square    (length=LENGTH, num_cycles=2, plane='xy', scaling=0.5, center=(0., 0., 1.)),
        'Helix':            gen.helix     (length=LENGTH, num_cycles=3, radius=0.5, rise=1.0,  center=(0., 0., 0.5)),
        'Lemniscate 3D':    gen.lemniscate_3d(length=LENGTH, num_cycles=2, scale_xy=0.5, scale_z=0.25, center=(0., 0., 1.)),
        'Spiral descent':   gen.spiral_descent(length=LENGTH, num_cycles=4, r0=0.8, r1=0.1, z0=2.0, z1=0.3),
    }

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajs)))

    # Plot 1: position vs time
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig1.suptitle('Quadrotor Reference Trajectories — Position vs Time', fontsize=13)
    coord_ylabels = ['x [m]', 'y [m]', 'z [m]']
    for name, (pos, vel, spd), color in zip(trajs.keys(), trajs.values(), colors):
        t = t_vec[:len(pos)]
        for j, ax in enumerate(axes1):
            ax.plot(t, pos[:, j], linewidth=1.2, label=name, color=color)
    for j, ax in enumerate(axes1):
        ax.set_ylabel(coord_ylabels[j])
        ax.grid(True, alpha=0.3)
    axes1[-1].set_xlabel('t [s]')
    axes1[0].legend(fontsize=7, ncol=3, loc='upper right')
    plt.tight_layout()

    #Plot 2: velocity vs time
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle('Quadrotor Reference Trajectories — Velocity vs Time', fontsize=13)
    vel_ylabels = ['x_dot [m/s]', 'y_dot [m/s]', 'z_dot [m/s]']
    for name, (pos, vel, spd), color in zip(trajs.keys(), trajs.values(), colors):
        t = t_vec[:len(vel)]
        for j, ax in enumerate(axes2):
            ax.plot(t, vel[:, j], linewidth=1.2, label=name, color=color)
    for j, ax in enumerate(axes2):
        ax.set_ylabel(vel_ylabels[j])
        ax.grid(True, alpha=0.3)
    axes2[-1].set_xlabel('t [s]')
    axes2[0].legend(fontsize=7, ncol=3, loc='upper right')
    plt.tight_layout()


    # ── Plot 3: 3D shapes
    n     = len(trajs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig4  = plt.figure(figsize=(16, 5 * nrows))
    fig4.suptitle('Quadrotor Reference Trajectories — 3D Shapes', fontsize=13)
    for idx, (name, (pos, vel, spd)) in enumerate(trajs.items()):
        ax = fig4.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                linewidth=1.5, color=colors[idx])
        ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]],
                   color='black', s=20, zorder=5)   # start marker
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('x [m]', fontsize=7)
        ax.set_ylabel('y [m]', fontsize=7)
        ax.set_zlabel('z [m]', fontsize=7)
        ax.tick_params(labelsize=6)
    plt.tight_layout()


    plt.show()
