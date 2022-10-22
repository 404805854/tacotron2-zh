#!/usr/bin/python
# coding=utf-8

import os
import argparse
import sys
from tqdm import tqdm


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--datasets', type=str, required=True,
                        help='datasets (phareses separated by |)')
    parser.add_argument('-t', '--training-files', type=str, required=True,
                        help='output train metadata')
    parser.add_argument('-v', '--validation-files', type=str, default="",
                        help='output validation metadata')
    return parser

from tacotron2.text import symbols as _letters
#_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890<># '


def normalize(str, trans_prosody=True, remove_punc=True):
    if trans_prosody:
        str = str.replace("# # # #", "# # #").replace("# # #", "#@").replace("# #", "#").replace("#@", "# # #")
    if remove_punc:
        arr = list(str)
        clean = []
        for iter in arr:
            if iter in _letters:
                clean.append(iter)
        return "".join(clean)
    else:
        return str


def main():
    parser = argparse.ArgumentParser(
        description='merge datasets metadata')
    parser = parse_args(parser)
    import sys
    from hparams import default_args

    args = default_args
    args.extend(sys.argv[1:])
    args, _ = parser.parse_known_args(args)

    dataset_list = args.datasets.split("|")
    training_data = []
    validation_data = []
    for dataset in tqdm(dataset_list):
        if not os.path.exists(os.path.join(dataset, "metadata.csv")):
            print(sys._getframe().f_code.co_name,
                  " dataset not found metadata.csv : ", dataset)
        else:
            with open(os.path.join(dataset, "metadata.csv"), encoding="utf8") as ifile:
                lines = ifile.readlines()
                if os.path.isabs(lines[0].strip().split("|")[0]):
                    training_data.extend(lines)
                else:
                    for line in tqdm(lines):
                        arr = line.strip().split("|")
                        arr[0] = os.path.abspath(os.path.join(dataset, arr[0]))
                        arr[-1] = normalize(arr[-1])
                        training_data.append("|".join(arr)+"\n")

        if args.validation_files != "":
            if not os.path.exists(os.path.join(dataset, "val_metadata.csv")):
                print(sys._getframe().f_code.co_name,
                      " dataset not found validation_data.csv : ", dataset)
            else:
                with open(os.path.join(dataset, "val_metadata.csv"), encoding="utf8") as ifile:
                    lines = ifile.readlines()
                    if os.path.isabs(lines[0].strip().split("|")[0]):
                        validation_data.extend(lines)
                    else:
                        for line in tqdm(lines):
                            arr = line.strip().split("|")
                            arr[0] = os.path.abspath(
                                os.path.join(dataset, arr[0]))
                            arr[-1] = normalize(arr[-1])
                            validation_data.append("|".join(arr)+"\n")
    with open(args.training_files, "w", encoding="utf8") as ofile:
        for line in tqdm(training_data):
            ofile.write(line)
    if args.validation_files != "":
        with open(args.validation_files, "w", encoding="utf8") as ofile:
            for line in tqdm(validation_data):
                ofile.write(line)


if __name__ == "__main__":
    main()
