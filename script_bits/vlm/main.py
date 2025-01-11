

# read first 100 cases for training, and following 900 for testing

import diffuser.utils as utils
from diffuser.datasets.bits_fun.bits_utils import *
from script_bits.vlm .utils.openai_completor import OpenAICompletor

from tqdm import tqdm

# Load custom_parser from diffuser.utils
class Parser(utils.Parser):
    dataset: str = 'tamp_easy_78'
    config: str = 'config_bits.diffusion.train_diff'


def get_first_n_lines(text, n):
    return "\n".join(text.split("\n", n)[:n])

def list_to_text(text_list, join_str="\n"):
    return join_str.join(text_list) + join_str

# ================ txt ================
# txt file reader
def read_txt_file(file_path):
    with open(file_path, 'r') as txt_file:
        return txt_file.read()
    
# txt file writer
def write_txt_file(file_path, data):
    with open(file_path, 'w') as txt_file:
        txt_file.write(data)

# ===================================== main ===================================

args = Parser().parse_args('diffusion')
results_save_dir = args.savepath + "_vlm_results/"

# 1. load the training tex
custom_texts = load_custom_texts(args.train_data_dir)

# 2. separate as list
train_num = 100
training_text = custom_texts[:train_num]
testing_text = custom_texts[train_num: train_num+1000]

# 3. for training, input all rows
training_text = list_to_text(training_text)

# 4. for testing, only input first 7 rows
testing_text = [get_first_n_lines(text, 7) for text in testing_text]

# 5. read texts
train_text = read_txt_file('script_bits/vlm/utils/prompts_train.txt')
infer_text = read_txt_file('script_bits/vlm/utils/prompts_infer.txt')

# 6. prompts

completor = OpenAICompletor()
completor.add_system(train_text)

train_prompt = f"\
    Examples are: \n \
    {training_text} \n \
"

ans = completor.answer(train_prompt)

print("======== train =========")
print(ans)

ans = completor.answer(infer_text)

print("======== infer =========")
print(ans)

#  7. start to test

# Ensure the result directory exists
os.makedirs(results_save_dir, exist_ok=True)

# remove files in the results_save_dir
for file in os.listdir(results_save_dir):
    os.remove(os.path.join(results_save_dir, file))

# Specify the output file path
output_file = os.path.join(results_save_dir, "generated_steps.txt")


for text in tqdm(testing_text):
    output_text = completor.answer(text)
    with open(output_file, "a") as f:  # Open in append mode ("a")
        f.write(output_text + "\n\n")

print(f"Generated steps for batch saved to {output_file}")



