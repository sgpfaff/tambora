from galpy.potential import PlummerPotential
from galpy.df import isotropicPlummerdf
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from src.tambora.util import galpydfsampler
from src.tambora.simulation import Simulation
from multiprocessing import Process, Queue

def run_sim(dt, pos, vel, masses, q):
    sim = Simulation()
    sim.add_particles('stars', pos, vel, mass=masses)
    sim.run(t_end=100, dt=dt, dt_out=1, eps=0.6)
    ts = sim.times.copy()
    dEs = np.array([np.abs(sim.percent_dE(t2=t)) for t in np.arange(len(ts))])
    q.put((dt, ts, dEs))

pot = PlummerPotential(amp=1e10*u.Msun, b=2*u.kpc)
df = isotropicPlummerdf(pot=pot)

if __name__ == '__main__':
    pos, vel, masses = galpydfsampler(df, n=10000, m_total=1e10)

    dt_values = [0.5, 0.2, 0.1, 0.05, 0.01]
    processes = []
    q = Queue()

    for dt in dt_values:
        p = Process(target=run_sim, args=(dt, pos, vel, masses, q))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = {}
    times = {}
    while not q.empty():
        dt, ts, dEs = q.get()
        times[str(dt)] = ts
        results[str(dt)] = dEs
    for dt in dt_values:
        plt.plot(times[str(dt)], results[str(dt)], label=f'dt={dt}')
    plt.xlabel('Time [Myr]')
    plt.ylabel('|dE/E| [%]')
    plt.yscale('log')
    plt.legend()
    plt.savefig('energy_convergence.png', dpi=150)

    np.save('convergence_results.npy', results)