import h5py, numpy as np
p="/coc/flash7/scratch/egowm/robotwin/dataset/stack_bowls_three/aloha-agilex_clean_50/data/episode0.hdf5"
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name:50s} shape={obj.shape} dtype={obj.dtype}")
with h5py.File(p,"r") as f:
    f.visititems(show)
    print("=== sample values ===")
    print("left_endpose[0]:", f["/endpose/left_endpose"][0])
    print("right_endpose[0]:", f["/endpose/right_endpose"][0])
    print("joint_action/left_gripper[0]:", f["/joint_action/left_gripper"][0])
    print("joint_action/right_gripper[0]:", f["/joint_action/right_gripper"][0])
    print("joint_action/vector[0]:", f["/joint_action/vector"][0])
    print("endpose/left_gripper[0]:", f["/endpose/left_gripper"][0])
    print("num frames (vector):", f["/joint_action/vector"].shape)
