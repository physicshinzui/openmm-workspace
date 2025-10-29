import MDAnalysis as mda
from MDAnalysis.transformations import unwrap, center_in_box, wrap
from tqdm import tqdm

# 入力
u = mda.Universe('top.pdb', 'traj.dcd')

# 選択
protein = u.select_atoms('protein')
water   = u.select_atoms('resname TIP3 or resname HOH or resname SOL')
all_atoms = u.atoms


# For cubic box
trans = [
    unwrap(u.atoms),
    center_in_box(protein, center='geometry'),
    wrap(u.atoms, compound='residues')
]
u.trajectory.add_transformations(*trans)

# output
with mda.Writer('traj_fixed.dcd', all_atoms.n_atoms) as W:
    for ts in tqdm(u.trajectory):
        W.write(u)

