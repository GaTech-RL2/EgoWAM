from egomimic.rldb.utils import S3RLDBDataset

ds = S3RLDBDataset(
    embodiment="eva_bimanual",
    mode="total",          # load all, then inspect both splits
    bucket_name="rldb",
    local_files_only=True,
    filters={"task": "cup_on_saucer", "scene": "1"},
    # use_future=True,
    # action_chunk=25
)
print("Train episode hashes:", sorted(ds.train_collections))
print("Valid episode hashes:", sorted(ds.valid_collections))
print("All loaded hashes:   ", sorted(ds.datasets.keys()))