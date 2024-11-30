from diffuser.datasets.bits_fun.gen_dataset.dynamics import eval_env

import diffuser.utils as utils
class Parser(utils.Parser):
    dataset: str = 'tamp_easy_78'
    config: str = 'config_bits.diffusion.train_diff'

args = Parser().parse_args('diffusion')

# model_save_dir = args.savepath + "_transformer/"
results_save_dir = args.savepath + "_transformer_results/generated_steps.txt"

all_length, converted_stage_action_num = eval_env(results_save_dir)

