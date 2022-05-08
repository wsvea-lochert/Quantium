import os
import json
from colorama import Fore
from Filch.FilchUtils import get_json_to_split


class FireboltDataSplitter:
    def __init__(self, json_path, out_dir):
        """
        Initialize the class.
        :param json_path: path to big json file.
        :param out_dir: path to output directory where the files will be saved.
        """
        self.json_path = json_path
        self.out_dir = out_dir
        self.samples, self.json_dict = get_json_to_split(self.json_path)
        self.split_value = int(len(self.samples))/8
        self.split1, self.split2, self.split3, self.split4, self.split5, self.split6, self.split7, self.split8 = {}, {}, {}, {}, {}, {}, {}, {}

    def split_data(self):
        """
        Split the data into 8 parts then run the __dump_json method to save the data into files.
        :return:
        """
        print(Fore.GREEN, f'Split value: {self.split_value}\n'
              f'Number of samples: {len(self.samples)}')

        for i in range(len(self.samples)):
            if i < self.split_value:
                self.split1[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 1 < i <= self.split_value * 2:
                self.split2[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 2 < i <= self.split_value * 3:
                self.split3[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 4 >= i > self.split_value * 3:
                self.split4[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 4 < i <= self.split_value * 5:
                self.split5[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 5 < i <= self.split_value * 6:
                self.split6[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 6 < i <= self.split_value * 7:
                self.split7[self.samples[i]] = self.json_dict[self.samples[i]]
            elif self.split_value * 8 >= i > self.split_value * 7:
                self.split8[self.samples[i]] = self.json_dict[self.samples[i]]

        print(Fore.CYAN, f'Number of samples in split 1: {len(self.split1)}\nNumber of samples in split 2: {len(self.split2)}\nNumber of samples in split 3: {len(self.split3)}\nNumber of samples in split 4: {len(self.split4)}')
        print(Fore.CYAN, f'Number of samples in split 5: {len(self.split5)}\nNumber of samples in split 6: {len(self.split6)}\nNumber of samples in split 7: {len(self.split7)}\nNumber of samples in split 8: {len(self.split8)}')

        self.__dump_json()

    def __dump_json(self):
        """
        Dump the data into files.
        :return:
        """
        with open(f'{self.out_dir}train_split1.json', 'w') as outfile:
            json.dump(self.split1, outfile, indent=2)

        with open(f'{self.out_dir}train_split2.json', 'w') as outfile:
            json.dump(self.split2, outfile, indent=2)

        with open(f'{self.out_dir}train_split3.json', 'w') as outfile:
            json.dump(self.split3, outfile, indent=2)

        with open(f'{self.out_dir}train_split4.json', 'w') as outfile:
            json.dump(self.split4, outfile, indent=2)

        with open(f'{self.out_dir}train_split5.json', 'w') as outfile:
            json.dump(self.split5, outfile, indent=2)

        with open(f'{self.out_dir}train_split6.json', 'w') as outfile:
            json.dump(self.split6, outfile, indent=2)

        with open(f'{self.out_dir}train_split7.json', 'w') as outfile:
            json.dump(self.split7, outfile, indent=2)

        with open(f'{self.out_dir}train_split8.json', 'w') as outfile:
            json.dump(self.split8, outfile, indent=2)
