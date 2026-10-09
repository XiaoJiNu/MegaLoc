#!/usr/bin/env python
"""Q3: re-evaluate MegaLoc top-1 snap against an RTK-anchored GT (not /map).

Per query bag: fit map<-ENU rigid (complex LS, same as reloc3r audit5) from
/ll/position/gps/gnss_fix, then per extracted frame (same step/sync logic as
reloc3r extract) compute cam center = fit(enu) + R_map_body @ v_lever, where
v_lever is the fixed cam offset in body frame recovered from pose.txt vs /map.
Reports /map-vs-RTK residual (GT noise floor) and MegaLoc err vs RTK GT.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_zc_megaloc import SCENES, OUT, seq_frames  # noqa: E402

BAGS = {  # query dir key -> mcap path
    ("mowing", "same_day_062224"): "/data/data/reLocationData/zc_data/20260921_062224.mcap",
    ("mowing", "cross_day_081533"): "/data/data/reLocationData/zc_data/20260920_081533.mcap",
    ("building_0923", "query_060415"): "/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap",
}
IMG_STEP, SYNC_MAX_MS = 5, 100.0


def main():
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    from scipy.spatial.transform import Rotation as Rot

    res = np.load  # shorthand
    out_rows = []
    for (scene, q), bag in BAGS.items():
        r = res(OUT / f"res__{scene}__{q}.npz")
        terr_map = r["terr"]
        _, db_t, _ = seq_frames(SCENES[scene]["db"])
        q_pngs, q_t, _ = seq_frames(SCENES[scene]["queries"][q])
        # lever arm v = cam offset in body frame: T_map_cam = T_map_body @ A -> A = T_mb^-1 @ T_mc ; v_body constant
        # recover A from first frames: A = inv(T_mb) @ T_mc. Re-read /map for t,R.
        map_t, map_p, map_R, gnss = [], [], [], []
        with open(bag, "rb") as f:
            rd = make_reader(f, decoder_factories=[DecoderFactory()])
            for s, ch, m, dec in rd.iter_decoded_messages(topics=["/map", "/ll/position/gps/gnss_fix", "/camera_01/color/image_raw"]):
                if ch.topic == "/map":
                    p = dec.pose.pose
                    map_t.append(m.log_time)
                    map_p.append((p.position.x, p.position.y, p.position.z))
                    map_R.append(Rot.from_quat((p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)).as_matrix())
                elif ch.topic == "/ll/position/gps/gnss_fix":
                    gnss.append((m.log_time, dec.latitude, dec.longitude))
        map_t = np.array(map_t)
        map_p, map_R = np.array(map_p), np.array(map_R)
        # frame timestamps: replay image topic with same step/sync rule
        ft, fj = [], []
        n_img = 0
        with open(bag, "rb") as f:
            rd = make_reader(f, decoder_factories=[DecoderFactory()])
            for s, ch, m, dec in rd.iter_decoded_messages(topics=["/camera_01/color/image_raw"]):
                n_img += 1
                if (n_img - 1) % IMG_STEP:
                    continue
                i = int(np.clip(np.searchsorted(map_t, m.log_time), 1, len(map_t) - 1))
                j = i if abs(map_t[i] - m.log_time) <= abs(map_t[i - 1] - m.log_time) else i - 1
                if abs(map_t[j] - m.log_time) / 1e6 > SYNC_MAX_MS:
                    continue
                ft.append(m.log_time)
                fj.append(j)
        fj = np.array(fj)
        assert len(fj) == len(q_pngs), f"frame count mismatch {len(fj)} vs {len(q_pngs)}"
        # lever arm (body frame): T_mc = T_mb @ A => A = inv(T_mb) @ T_mc, averaged
        T_mb = np.eye(4)[None].repeat(len(fj), 0)
        T_mb[:, :3, :3] = map_R[fj]
        T_mb[:, :3, 3] = map_p[fj]
        T_mc = np.zeros((len(fj), 4, 4)); T_mc[:, :3, :3] = [np.loadtxt(str(p).replace(".color.png", ".pose.txt"))[:3, :3] for p in q_pngs]
        T_mc[:, :3, 3] = q_t; T_mc[:, 3, 3] = 1
        A = np.linalg.inv(T_mb) @ T_mc
        v_lev = A[:, :3, 3].mean(0)
        assert np.linalg.norm(A[:, :3, 3] - v_lev) < 0.02, "lever not constant -> chain mismatch"
        # rigid fit map<-ENU (2D, complex LS)
        G = np.array(gnss)
        lat0, lon0 = G[0, 1], G[0, 2]
        e = (G[:, 2] - lon0) * math.cos(math.radians(lat0)) * 111320
        n = (G[:, 1] - lat0) * 110540
        jg = np.clip(np.searchsorted(map_t, G[:, 0]), 0, len(map_t) - 1)
        E = (e - e.mean()) + 1j * (n - n.mean())
        Mc = (map_p[jg, 0] - map_p[jg, 0].mean()) + 1j * (map_p[jg, 1] - map_p[jg, 1].mean())
        z = (E.conj() @ Mc) / (E.conj() @ E).real
        res_fit = np.abs(Mc - z * E)
        # RTK-anchored camera centers at frame times
        enu = np.empty((len(fj), 2))
        for i, j in enumerate(fj):
            k = np.argmin(np.abs(G[:, 0] - map_t[j]))
            enu[i] = (e[k], n[k])
        rtk = np.stack([
            z.real * (enu[:, 0] - e.mean()) - z.imag * (enu[:, 1] - n.mean()) + map_p[jg, 0].mean(),
            z.imag * (enu[:, 0] - e.mean()) + z.real * (enu[:, 1] - n.mean()) + map_p[jg, 1].mean(),
            np.zeros(len(fj)) + map_p[fj, 2].mean(),
        ], 1)
        cam_rtk = rtk + np.einsum("nij,j->ni", map_R[fj], v_lev)
        terr_rtk = np.linalg.norm(db_t[r["top1"]] - cam_rtk, axis=1)
        row = {"scene": scene, "query": q,
               "map_vs_rtk_resid_p50_m": float(np.median(res_fit)),
               "map_vs_rtk_resid_p95_m": float(np.percentile(res_fit, 95)),
               "terr_vs_map_med/p90": [float(np.median(terr_map)), float(np.percentile(terr_map, 90))],
               "terr_vs_rtk_med/p90": [float(np.median(terr_rtk)), float(np.percentile(terr_rtk, 90))],
               "terr<2m vs_map/vs_rtk": [float((terr_map < 2).mean()), float((terr_rtk < 2).mean())]}
        out_rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "followup_rtk.json").write_text(json.dumps(out_rows, indent=2))


if __name__ == "__main__":
    main()
