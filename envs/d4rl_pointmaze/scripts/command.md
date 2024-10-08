python envs/d4rl_pointmaze/scripts/generate_Twall_Maze2d_datasets.py  --save_images \
--data_dir './testdata'   --save_video --manual_maze 'm_maze2'


python envs/d4rl_pointmaze/scripts/generate_d4rl_datasets.py --data_dir './testdata'   --manual_maze 'm_maze2' \
--num_samples 2000

python envs/d4rl_pointmaze/scripts/generate_d4rl_datasets.py --data_dir '~/.d4rl_pointmass/datasets' --file_name 'mMaze2-v0.hdf5'  --manual_maze 'm_maze2' --num_samples 2000 --save_video

python envs/d4rl_pointmaze/scripts/generate_d4rl_datasets.py --data_dir '~/.d4rl_pointmass/datasets' --file_name 'random.hdf5'  --num_samples 2000 --save_video \
--fixed_maze_size 20


python envs/d4rl_pointmaze/scripts/generate_d4rl_datasets.py --data_dir '~/.d4rl_pointmass/datasets' --file_name 'test-v0.hdf5'  --manual_maze 'test_maze' --num_samples 2000 --save_video