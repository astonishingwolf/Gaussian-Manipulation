# to get trajectory file, run:
python DG-Mesh/dgmesh/render_trajectory.py --config DG-Mesh/dgmesh/configs/iphone/lamp.yaml \
    --start_checkpoint DG-Mesh/outputs/lamp/data_v4-2025-12-02_04-25-35/

# file should looks like
# DG-Mesh/outputs/lamp/rendering-traj-data-2025-12-02_22-49-40/deformed_xyz.npy

# to generate video with custom trajectory:
python DG-Mesh/dgmesh/render_custom_trajectory.py --config DG-Mesh/dgmesh/configs/iphone/lamp.yaml \
    --start_checkpoint DG-Mesh/outputs/lamp/data_v4-2025-12-02_04-25-35/ --custom_dxyz_path DG-Mesh/outputs/lamp/rendering-traj-data-2025-12-02_22-49-40/deformed_xyz.npy