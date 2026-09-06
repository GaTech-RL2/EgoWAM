import zarr, numpy as np, sys
p = sys.argv[1]
root = zarr.open(p, "r")
for k in ["data/head_camera", "data/state", "data/action", "meta/episode_ends"]:
    a = root[k]
    print(f"  {k:22s} shape={a.shape} dtype={a.dtype} chunks={a.chunks}")
print("episode_ends[:5]:", root["meta/episode_ends"][:5], "n_episodes:", root["meta/episode_ends"].shape[0])
print("state[0]:", np.asarray(root["data/state"][0]))
print("action[0]:", np.asarray(root["data/action"][0]))
# verify action = state shifted by one within an episode
st = np.asarray(root["data/state"][:5])
ac = np.asarray(root["data/action"][:5])
print("state[1] == action[0] (shift-by-one check):", np.allclose(st[1], ac[0]))
print("head_camera[0] min/max:", int(root["data/head_camera"][0].min()), int(root["data/head_camera"][0].max()))
