from Firebolt.FireboltImageFlipper import FireboltImageFlipper
from Firebolt.FireboltBackground import FireboltBackground
from Firebolt.FireboltDatasetCreator import FireboltDatasetCreator
from FireboltDataSplitter import FireboltDataSplitter

# aug = FireboltImageFlipper(img_dir="C:/Users/William/Documents/C3P0 datasets/dataset2/img/img224x224/", input_json="C:/Users/William/Documents/C3P0 datasets/dataset2/dataset2.json",
#                            output_dir="C:/Users/William/Documents/C3P0 datasets/dataset2-flip/img/base/", output_json_path="C:/Users/William/Documents/C3P0 datasets/dataset2-flip/json/base.json")
# aug.flip()

bg = FireboltBackground(img_dir='C:/Users/William/Documents/C3P0 datasets/dataset2-flip/img/base/', input_json='C:/Users/William/Documents/C3P0 datasets/dataset2-flip/json/base.json',
                        output_dir='C:/Users/William/Documents/C3P0 datasets/dataset2-flip/img/bg/', output_json_path='C:/Users/William/Documents/C3P0 datasets/dataset2-flip/json/bg.json',
                        bg_dir='C:/Users/William/Documents/C3P0 datasets/dataset3/dataset/bg/', blanks_dir='C:/Users/William/Documents/C3P0 datasets/blanks/', num_blanks=516120, blank_bg='C:/Users/William/Documents/C3P0 datasets/backgrounds/')
bg.swap()

dataset = FireboltDatasetCreator('C:/Users/William/Documents/C3P0 datasets/dataset2-flip/img/bg/', 'C:/Users/William/Documents/C3P0 datasets/dataset2-flip/json/bg.json',
                                 'C:/Users/William/Documents/C3P0 datasets/dataset2-flip/dataset2/train/', '"C:/Users/William/Documents/C3P0 datasets/dataset2-flip/dataset2/train.json"')
dataset.create(True)

splitter = FireboltDataSplitter('C:/Users/William/Documents/C3P0 datasets/dataset2-flip/dataset2/train/', 'C:/Users/William/Documents/C3P0 datasets/dataset2-flip/dataset2/train.json')