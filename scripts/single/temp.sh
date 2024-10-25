

# maze
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --device "cuda:5"

# maze varh
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --device "cuda:5"

# maze value
python scripts/train/train_value.py --config 'config.single.varh.train_diff_var1'  --device "cuda:4"



# med
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-medium-v1 --device "cuda:7" 

# med varh
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-medium-v1 --device "cuda:7"

# med value 
python scripts/train/train_value.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-medium-v1 --device "cuda:6"


# large 
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-large-v1 --device "cuda:6"
python scripts/train/train_diffuser.py --config 'config.single.train_diff_store'  --dataset maze2d-large-v1 --device "cuda:6"

# large varh
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-large-v1 --device "cuda:5"

# large value
python scripts/train/train_value.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-large-v1 --device "cuda:5"




# large 
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-large-v1 --device "cuda:7" \
 --horizon 512

# large varh
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-large-v1 --device "cuda:6" \
 --horizon 512

# large value
python scripts/train/train_value.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-large-v1 --device "cuda:5" \
 --horizon 512


