# test PAOFLOW Z2Pack

import z2pack
import tbmodels
import scipy.optimize as so
import matplotlib.pyplot as plt
plt.switch_backend('agg')
model = tbmodels.Model.from_wannier_files(
    hr_file='z2pack_hamiltonian.dat'
    )
system = z2pack.tb.System(model,bands=32)

def gap(k):
    eig = model.eigenval(k)
    return eig[32] - eig[31]

# Run the WCC calculations
# a small sphere around one Weyl point
#guess = so.minimize(gap, x0=[0.6239797, 0.70845137, 0])
#result_1 = z2pack.surface.run(
#    system=system,
#    surface=z2pack.shape.Sphere(guess.x, 0.05),
#    iterator=range(20, 101, 2),
#    min_neighbour_dist=1e-6
#    )
result_1 = z2pack.surface.run(
    system=system,
    surface=z2pack.shape.Sphere(center=(0.6239797, 0.70845137, 0), radius=0.007),
#    save_file='./results/res1.json',
#    load=True
)
# a small sphere around the other Weyl point
result_2 = z2pack.surface.run(
    system=system,
    surface=z2pack.shape.Sphere(center=(0.61197818, 0.70080602, 0), radius=0.007),
#    save_file='./results/res2.json',
#    load=True
)
# a bigger sphere around both Weyl points
result_3 = z2pack.surface.run(
    system=system,
    surface=z2pack.shape.Sphere(center=(0.61797894, 0.7046287, 0.), radius=0.014)
#    save_file='./results/res3.json',
#    load=True
)

# Combining the two plots
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 12))
for res, axis in zip([result_1, result_2, result_3], ax):
    z2pack.plot.chern(res, axis=axis)
plt.savefig('plots/plot.pdf', bbox_inches='tight')

print(('Chern number / Weyl chirality around WP1: {0}'.format(z2pack.invariant.chern(result_1))))
print(('Chern number / Weyl chirality around WP2: {0}'.format(z2pack.invariant.chern(result_2))))
print(('Chern number / Weyl chirality around both Weyl points: {0}'.format(z2pack.invariant.chern(result_3))))
