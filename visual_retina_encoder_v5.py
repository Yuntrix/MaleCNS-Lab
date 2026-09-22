"""Offline engineering visualizer for the V4 side-aware retina candidates.

No OBS, camera calibration, LIF, spikes, motor control or WORLD orientation is
used. Hex-to-display coordinates are arbitrary axial visualization coordinates.
"""
from pathlib import Path
import argparse
import math
import sys
import numpy as np
import pandas as pd


CELL_CODES = {"unknown": 0, "R1-R6": 1, "R7": 7, "R8": 8}


def block(title):
    print("\n" + "=" * 92 + "\n" + title + "\n" + "=" * 92, flush=True)


def axial_to_display(h1, h2, swap_axes=False, flip_x=False, flip_y=False, rotation=0):
    """Engineering display transform only; it has no biological orientation."""
    q, r = (float(h1), float(h2))
    if swap_axes:
        q, r = r, q
    x, y = q + 0.5 * r, (math.sqrt(3.0) / 2.0) * r
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    theta = math.radians(rotation)
    return x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)


class RetinaProjector:
    """Map normalized grayscale arrays to the nearest side-aware hex records."""
    def __init__(self, map_path=None, swap_axes=False, flip_x=False, flip_y=False, rotation=0):
        self.map_path = Path(map_path) if map_path else None
        self.swap_axes, self.flip_x, self.flip_y, self.rotation = swap_axes, flip_x, flip_y, rotation
        self.map = None
        self.load_map()

    def load_map(self):
        if self.map_path is None:
            self.map = None
            return self
        with np.load(self.map_path, allow_pickle=False) as z:
            self.map = {k: z[k].copy() for k in z.files}
        return self

    def get_eye_coordinates(self, side):
        if self.map is None:
            return np.empty((0, 2), dtype=float)
        code = {"LEFT": -1, "RIGHT": 1}[str(side).upper()]
        mask = self.map["coordinate_side"] == code
        return self.map["display_x"][mask], self.map["display_y"][mask]

    def normalize_hex_geometry(self):
        if self.map is None or len(self.map["hex1"]) == 0:
            return np.empty((0, 2), dtype=float)
        xy = np.array([axial_to_display(a, b, self.swap_axes, self.flip_x, self.flip_y, self.rotation)
                       for a, b in zip(self.map["hex1"], self.map["hex2"])])
        for side in (-1, 1):
            mask = self.map["coordinate_side"] == side
            if mask.any():
                lo, hi = xy[mask].min(axis=0), xy[mask].max(axis=0)
                span = np.where(hi > lo, hi - lo, 1.0)
                xy[mask] = (xy[mask] - lo) / span
        return xy

    def map_normalized_point_to_nearest_hex(self, x, y, side):
        if self.map is None or not (0 <= float(x) <= 1 and 0 <= float(y) <= 1):
            return None
        xy = self.normalize_hex_geometry()
        code = {"LEFT": -1, "RIGHT": 1}[str(side).upper()]
        mask = self.map["coordinate_side"] == code
        if not mask.any():
            return None
        indices = np.flatnonzero(mask)
        distance = (xy[mask, 0] - x) ** 2 + (xy[mask, 1] - y) ** 2
        return int(indices[int(np.argmin(distance))])

    def map_normalized_image_to_hex_values(self, image, side):
        image = np.asarray(image, dtype=float)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("image must be a non-empty 2D grayscale array")
        result = {}
        for i in np.flatnonzero(self.map["coordinate_side"] == {"LEFT": -1, "RIGHT": 1}[str(side).upper()]):
            x, y = self.map["normalized_x"][i], self.map["normalized_y"][i]
            px = min(image.shape[1] - 1, max(0, int(round(x * (image.shape[1] - 1)))))
            py = min(image.shape[0] - 1, max(0, int(round(y * (image.shape[0] - 1)))))
            result[(int(self.map["hex1"][i]), int(self.map["hex2"][i]))] = float(image[py, px])
        return result


def connected_components(coords):
    remaining = set(coords)
    sizes = []
    for start in list(remaining):
        if start not in remaining:
            continue
        remaining.remove(start); stack = [start]; size = 0
        while stack:
            q, r = stack.pop(); size += 1
            for nxt in ((q+1,r),(q-1,r),(q,r+1),(q,r-1),(q+1,r-1),(q-1,r+1)):
                if nxt in remaining:
                    remaining.remove(nxt); stack.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def safe_class(frame):
    c = frame.get("cell_class", "unknown").fillna("unknown").astype(str)
    return c.where(c.isin(CELL_CODES), "unknown")


def main(root=None, output=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    processed = root / "data/processed"
    source_reports = processed / "retina_reports"
    reports = Path(output) if output else source_reports
    npz_in = processed / "retina_hex_pair_candidates.npz"
    consensus_path = source_reports / "retina_hex_pair_consensus.csv"
    population_path = source_reports / "retina_hex_pair_population.csv"
    block("INPUT MAP")
    print(npz_in); print(consensus_path); print(population_path)
    if not npz_in.is_file() or not consensus_path.is_file():
        raise FileNotFoundError("V4 NPZ and consensus report are required")
    with np.load(npz_in, allow_pickle=False) as z:
        print("NPZ arrays:")
        for key in z.files:
            print(" ", key, z[key].shape, z[key].dtype)
        npz = {key: z[key].copy() for key in z.files}
    required = {"neuron_idx", "body_id", "laterality_code", "hex1", "hex2", "eligible_target_count", "dominance", "confidence_code", "subset_code"}
    if not required.issubset(npz):
        raise ValueError("NPZ missing required V4 fields: " + str(sorted(required - set(npz))))
    n = len(npz["neuron_idx"])
    dup_total = n - len(np.unique(npz["neuron_idx"]))
    print("Rows:", n, "duplicate neuron rows:", dup_total, "subset codes:", np.unique(npz["subset_code"]).tolist())
    consensus = pd.read_csv(consensus_path)
    needed = {"neuron_idx", "subset", "voting_mode", "strong_resolved", "cell_class", "sensory_laterality", "top_hex1", "top_hex2"}
    if not needed.issubset(consensus):
        raise ValueError("Consensus report missing required columns")
    selected = consensus[(consensus.subset == "all_direct") & (consensus.voting_mode == "side_pair") & (consensus.strong_resolved.astype(str).str.lower() == "true")].copy()
    selected = selected.drop_duplicates("neuron_idx", keep=False)
    if selected.empty:
        raise ValueError("No unique all_direct STRONG resolved side_pair rows")
    selected["cell_class"] = safe_class(selected)
    # Use consensus as the authoritative final population; NPZ supplies numeric fields.
    map_df = selected[["neuron_idx", "body_id", "cell_class", "sensory_laterality", "top_target_side", "top_hex1", "top_hex2", "dominance", "eligible_target_count"]].copy()
    map_df = map_df.rename(columns={"top_target_side": "laterality", "top_hex1": "hex1", "top_hex2": "hex2"})
    npz_all = pd.DataFrame({k: npz[k] for k in ("neuron_idx", "body_id", "dominance", "eligible_target_count", "subset_code")})
    npz_all = npz_all[npz_all.subset_code == 0].drop_duplicates("neuron_idx", keep=False)
    map_df = map_df.merge(npz_all[["neuron_idx", "body_id", "dominance", "eligible_target_count"]], on="neuron_idx", suffixes=("_csv", "_npz"), validate="one_to_one")
    map_df["dominance"] = map_df.dominance_csv
    map_df["eligible_target_count"] = map_df.eligible_target_count_csv.astype(int)
    map_df["body_id"] = map_df.body_id_csv.astype(np.int64)
    map_df = map_df[["laterality", "hex1", "hex2", "neuron_idx", "body_id", "cell_class", "dominance", "eligible_target_count"]]
    reports.mkdir(parents=True, exist_ok=True)
    map_df["hex1"] = map_df.hex1.astype(float).round().astype(int)
    map_df["hex2"] = map_df.hex2.astype(float).round().astype(int)
    block("DEDUPLICATION")
    print("NPZ duplicate rows (columnar duplicate):", dup_total)
    print("Final population uses all_direct + STRONG + resolved side_pair only.")
    print("Final projection candidates:", len(map_df), "unique neurons:", map_df.neuron_idx.nunique())
    block("FINAL SPATIAL POPULATION")
    print(map_df.groupby("laterality").size().to_string())
    map_df.to_csv(reports / "retina_v5_projection_population.csv", index=False, encoding="utf-8-sig")
    # Engineering geometry.
    xy = np.array([axial_to_display(a,b) for a,b in zip(map_df.hex1, map_df.hex2)])
    map_df["display_x"], map_df["display_y"] = xy[:,0], xy[:,1]
    norm = np.zeros_like(xy)
    for side, code in (("LEFT", -1), ("RIGHT", 1)):
        mask = map_df.laterality.eq(side).to_numpy(); lo, hi = xy[mask].min(0), xy[mask].max(0); span=np.where(hi>lo,hi-lo,1.0); norm[mask]=(xy[mask]-lo)/span
    map_df["normalized_x"], map_df["normalized_y"] = norm[:,0], norm[:,1]
    block("LEFT GEOMETRY")
    for side in ("LEFT", "RIGHT"):
        f=map_df[map_df.laterality.eq(side)]; print(side,"unique coords:",f[["hex1","hex2"]].drop_duplicates().shape[0],"hex1:",(f.hex1.min(),f.hex1.max()),"hex2:",(f.hex2.min(),f.hex2.max()))
    block("HEX GRID SANITY")
    sanity=[]
    for side in ("LEFT","RIGHT"):
        f=map_df[map_df.laterality.eq(side)]; coords=set(zip(f.hex1,f.hex2)); distances=[]
        for q,r in coords:
            neigh=[(q+1,r),(q-1,r),(q,r+1),(q,r-1),(q+1,r-1),(q-1,r+1)]
            distances += [math.sqrt(3) for x in neigh if x in coords]
        isolated=sum(not any(x in coords for x in ((q+1,r),(q-1,r),(q,r+1),(q,r-1),(q+1,r-1),(q-1,r+1))) for q,r in coords)
        components=connected_components(coords); holes=max(0,(f.hex1.max()-f.hex1.min()+1)*(f.hex2.max()-f.hex2.min()+1)-len(coords))
        row=dict(laterality=side,unique_coordinates=len(coords),isolated_coordinates=isolated,connected_components=len(components),largest_component=max(components,default=0),bounding_box_holes=holes,nearest_neighbor_distance_mean=float(np.mean(distances)) if distances else np.nan,nearest_neighbor_distance_min=float(np.min(distances)) if distances else np.nan)
        sanity.append(row); print(row)
    print("WARNING: bounding-box holes are not proof of missing hexes; axial lattice may be clipped.")
    block("CELL CLASS OCCUPANCY")
    coord = map_df.groupby(["laterality","hex1","hex2"])
    occupancy=coord.agg(sensory_neuron_count=("neuron_idx","nunique"),R1_R6=("cell_class",lambda s:int((s=="R1-R6").sum())),R7=("cell_class",lambda s:int((s=="R7").sum())),R8=("cell_class",lambda s:int((s=="R8").sum())),unknown=("cell_class",lambda s:int((s=="unknown").sum())),neuron_indices=("neuron_idx",lambda s: ",".join(map(str,sorted(set(s))))),body_ids=("body_id",lambda s:",".join(map(str,sorted(set(s)))))).reset_index()
    print(occupancy.groupby("laterality").sensory_neuron_count.describe().to_string())
    block("SYNTHETIC PROJECTION TEST")
    # map npz is built below; projector behavior is exercised with the same final map.
    map_path=processed/"retina_spatial_map_v1.npz"
    cell_code=map_df.cell_class.map(CELL_CODES).astype(np.int8).to_numpy()
    side_code=map_df.laterality.map({"LEFT":-1,"RIGHT":1}).astype(np.int8).to_numpy()
    np.savez_compressed(map_path,coordinate_side=side_code,hex1=map_df.hex1.to_numpy(np.int64),hex2=map_df.hex2.to_numpy(np.int64),display_x=map_df.display_x.to_numpy(float),display_y=map_df.display_y.to_numpy(float),normalized_x=map_df.normalized_x.to_numpy(float),normalized_y=map_df.normalized_y.to_numpy(float),neuron_idx=map_df.neuron_idx.to_numpy(np.int64),body_id=map_df.body_id.to_numpy(np.int64),cell_class_code=cell_code,dominance=map_df.dominance.to_numpy(float),eligible_target_count=map_df.eligible_target_count.to_numpy(np.int64),offsets=np.array([0,len(map_df)],dtype=np.int64),purpose=np.asarray("engineering_visualization_only_orientation_arbitrary_WORLD_uncalibrated"))
    projector=RetinaProjector(map_path)
    images={"full_dark":np.zeros((32,32)),"full_bright":np.ones((32,32)),"left_bright_half":np.pad(np.ones((32,16)),((0,0),(0,16))),"right_bright_half":np.pad(np.ones((32,16)),((0,0),(16,0))),"center_bright_spot":np.zeros((32,32))}
    images["center_bright_spot"][12:20,12:20]=1
    for name,image in images.items():
        vals={side:projector.map_normalized_image_to_hex_values(image,side) for side in ("LEFT","RIGHT")}; print(name,"nonzero LEFT/RIGHT:",sum(v>0 for v in vals["LEFT"].values()),sum(v>0 for v in vals["RIGHT"].values()))
    block("LIMITATIONS")
    print("hex1/hex2 -> display_x/display_y is standard axial engineering geometry only; orientation is arbitrary and not screen calibrated.")
    print("Nearest-neighbor distances use axial adjacency; holes/components describe the candidate set, not biological defects.")
    print("Synthetic images test sampling only; no firing rates, spikes, OBS, LIF, motor or server changes.")
    print("WORLD projection still needs camera/projector calibration and an explicit eye/screen transform.")
    block("GENERATED FILES")
    outputs=[reports/"retina_v5_projection_population.csv",reports/"retina_v5_hex_sanity.csv",reports/"retina_v5_coordinate_occupancy.csv",map_path]
    pd.DataFrame(sanity).to_csv(outputs[1],index=False,encoding="utf-8-sig"); occupancy.to_csv(outputs[2],index=False,encoding="utf-8-sig")
    for side in ("LEFT","RIGHT"):
        import matplotlib.pyplot as plt
        f=map_df[map_df.laterality.eq(side)]; plt.figure(figsize=(8,7)); plt.scatter(f.display_x,f.display_y,c=f.dominance,s=20+8*f.eligible_target_count,cmap="viridis"); plt.gca().set_aspect("equal"); plt.colorbar(label="dominance"); plt.title(f"{side} engineering hex map — orientation arbitrary / not screen calibrated"); plt.xlabel("display_x (engineering)"); plt.ylabel("display_y (engineering)"); path=reports/f"retina_{side.lower()}_hex_map.png"; plt.savefig(path,dpi=160,bbox_inches="tight"); plt.close(); outputs.append(path)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharex=False,sharey=False)
    for ax,side in zip(axes,("LEFT","RIGHT")):
        f=map_df[map_df.laterality.eq(side)]; ax.scatter(f.display_x,f.display_y,c=f.dominance,s=20+8*f.eligible_target_count,cmap="viridis"); ax.set_aspect("equal"); ax.set_title(side+" (orientation arbitrary)"); ax.set_xlabel("display_x")
    fig.suptitle("Bilateral engineering hex map — sides shown separately; not screen calibrated"); fig.tight_layout(); path=reports/"retina_bilateral_hex_map.png"; fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig); outputs.append(path)
    print("Final map rows:",len(map_df),"(expected near V4 STRONG, not hard-coded)")
    for p in outputs: print(p)
    return map_df, occupancy


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try: main(args.root,args.output)
    except Exception as error: print(f"ERROR: {type(error).__name__}: {error}",file=sys.stderr); sys.exit(1)
