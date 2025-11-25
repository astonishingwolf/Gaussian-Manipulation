python video-manipulation/preprocess/preprocess_video.py --input ball.mp4 --output optical_flows --visualize

python video-manipulation/preprocess/2_track_trajectories.py \
        --input data/optical_flows/ball \
        --use-flow-magnitude --magnitude-threshold 0.5
        
python video-manipulation/preprocess/3_analyze_trajectories_fft.py \
        --input data/optical_flows/ball/trajectories/trajectories.npy \
        --fps 150 --visualize