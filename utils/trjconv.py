import MDAnalysis as mda
import sys
import argparse
import warnings
from tqdm import tqdm

def extract_trajectory_subset(topology_file, trajectory_file, output_file, selection_string, step=1):
    """
    MDAnalysisを使用して、トラジェクトリから原子のサブセットを抽出し、
    さらにフレームを間引いて保存する．
    """
    
    warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.coordinates.DCD')

    print(f"Loading universe...")
    print(f"  Topology: {topology_file}")
    print(f"  Trajectory: {trajectory_file}")
    
    try:
        u = mda.Universe(topology_file, trajectory_file)
    except Exception as e:
        print(f"Error loading files: {e}", file=sys.stderr)
        return False

    print(f"Original trajectory has {u.trajectory.n_frames} frames and {u.atoms.n_atoms} atoms.")

    # 選択クエリで原子グループを作成
    print(f"Applying selection: '{selection_string}'")
    try:
        selected_atoms = u.select_atoms(selection_string)
    except Exception as e:
        print(f"Error during atom selection: {e}", file=sys.stderr)
        return False

    if selected_atoms.n_atoms == 0:
        print(f"Error: No atoms matched the selection '{selection_string}'.", file=sys.stderr)
        return False

    print(f"Selected {selected_atoms.n_atoms} atoms.")
    
    # 間引きステップの確認
    if step <= 0:
        print(f"Error: --step must be a positive integer (got {step}).", file=sys.stderr)
        return False
    
    if step > 1:
        print(f"Applying decimation: Saving 1 frame every {step} steps.")

    # Writer (書き出し機能) を準備
    try:
        with mda.Writer(output_file, selected_atoms.n_atoms) as W:
            print(f"Writing selected trajectory to {output_file}...")
            
            # トラジェクトリをスライシング [::step] で間引く
            total_frames_written = 0
            for ts in tqdm(u.trajectory[::step]):
                W.write(selected_atoms)
                total_frames_written += 1
            
            print(f"Wrote {total_frames_written} frames.")

    except Exception as e:
        print(f"Error during writing to {output_file}: {e}", file=sys.stderr)
        return False

    print("Done.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a subset of atoms and/or decimate (thin) a DCD trajectory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 必須の引数
    parser.add_argument(
        '-t', '--topology', 
        metavar='TOPO_FILE', 
        type=str, 
        required=True, 
        help="Input topology file (e.g., .psf, .pdb)"
    )
    parser.add_argument(
        '-f', '--trajectory', 
        metavar='TRAJ_FILE', 
        type=str, 
        required=True, 
        help="Input trajectory file (e.g., .dcd)"
    )
    parser.add_argument(
        '-o', '--output', 
        metavar='OUT_FILE', 
        type=str, 
        required=True, 
        help="Output trajectory file (e.g., subset.dcd)"
    )
    parser.add_argument(
        '-s', '--selection', 
        metavar='"SELECTION"', 
        type=str, 
        required=True, 
        help="Atom selection string (MDAnalysis syntax).\n"
             "Examples: 'protein', 'resname LIG', 'resid 10-20'"
    )
    
    # オプションの引数 (間引き用)
    parser.add_argument(
        '-st', '--step', 
        metavar='N', 
        type=int, 
        default=1, 
        help="Decimation step. Save one frame every N steps. (Default: 1, no decimation)"
    )
    
    args = parser.parse_args()

    success = extract_trajectory_subset(
        args.topology, 
        args.trajectory, 
        args.output, 
        args.selection,
        args.step
    )

    if not success:
        print("\nScript finished with errors.", file=sys.stderr)
        sys.exit(1)
