#!/usr/bin/python
# coding=utf-8

import os
import argparse
import sys
from tqdm import tqdm

import torchaudio
from speechbrain.pretrained import EncoderClassifier

from libsvm.svmutil import svm_read_problem, svm_problem, svm_train, svm_save_model, svm_parameter, svm_predict

import torch

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device": "cuda"})


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-t', '--training-files', type=str, required=True,
                        help='output train metadata')
    parser.add_argument('-v', '--validation-files', type=str, default="",
                        help='output validation metadata')
    parser.add_argument('-m', '--method', type=str, default="mean",
                        help='[mean] or [svm]')
    return parser


def get_spk_id(path):
    return path.strip().split("/")[-2]


def get_spk_embedding(path):
    signal, _ = torchaudio.load(path)
    return classifier.encode_batch(signal)


def create_spk_dict(metadata, split="|"):
    spk_dict = {}
    for line in tqdm(metadata):
        arr = line.split(split)
        if len(arr) != 3:
            continue
        path = arr[0]
        spk_id = get_spk_id(path)
        spk_embedding = get_spk_embedding(path)
        if spk_id not in spk_dict.keys():
            spk_dict[spk_id] = []
        spk_dict[spk_id].append(spk_embedding)
    return spk_dict


def create_svm_train(args, path):
    with open(args.training_files, "r", encoding="utf8") as ifile:
        metadata = ifile.readlines()
    print("create spk dict data")
    spk_dict = create_spk_dict(metadata)
    label = 0
    spk_id_label = {}
    print("create svm train data")
    with open(path, "w", encoding="utf8") as ofile:
        for key, value in tqdm(spk_dict.items()):
            spk_id_label[key] = label
            for item in value:
                line = [str(label)]
                for i in range(0, len(item.view(-1))):
                    line.append("{}:{}".format(i+1, item.view(-1)[i]))
                ofile.write(" ".join(line)+"\n")
            label += 1
    return spk_id_label


def train_svm(train_path, scale_path, model_path):
    os.system("mv {} {}.ori && svm-scale -s {} {}.ori > {}".format(train_path,
              train_path, scale_path, train_path, train_path))
    print('create svm scale success')
    y, x = svm_read_problem(train_path)  # 读取训练集的数据
    prob = svm_problem(y, x)
    # param = svm_parameter('-c 8.0 -g 0.00048828125')  # 根据之前的结果设置参数
    # model = svm_train(y, x)  # 使用默认参数训练集的数据训练模型
    # model = svm_load_model('model_svm') # 读取文件中的模型
    # model = svm_train(prob, param)  # 使用指定参数训练
    model = svm_train(prob)
    svm_save_model(model_path, model)  # 将训练好的模型保存到文件 model_svm 中
    print('create svm model success')

    # print('Train:')
    # p_label, p_acc, p_val = svm_predict(y, x, model)
    # print("p_label: ", p_label)   # 每个数据对应的预测类别（标签）
    # print("p_acc: ", p_acc)  # 预测的精确度，均值和回归的平方相关系数
    # print("p_val: ", p_val)   # 在指定参数'-b 1'时将返回判定系数(判定的可靠程度)


def create_svm(args, train_path, scale_path, model_path):
    spk_id_label = create_svm_train(args, train_path)
    train_svm(train_path, scale_path, model_path)
    return spk_id_label


def get_trusted_tensors(args, train_path="./data/svm_train.txt", scale_path="./data/svm_scale.txt", model_path="./data/svm_model.txt", spk_folder="./data/spk_embeddings/"):
    spk_id_tensor = {}
    if not os.path.exists(spk_folder):
        os.mkdir(spk_folder)
    if args.method == "svm":
        spk_id_label = create_svm(args, train_path, scale_path, model_path)
        trusted_dict = {}
        with open(model_path, "r", encoding="utf8") as ifile:
            for line in ifile.readlines():
                arr = line.strip().split(" ")
                if len(arr) < 500:
                    continue
                try:
                    label = int(float(arr[0]))
                except:
                    continue
                if label not in trusted_dict.keys():
                    trusted_dict[label] = arr[-512:]
        for key, value in trusted_dict:
            if key in spk_id_label.keys():
                arr = []
                try:
                    for item in value:
                        arr.append(float(item.split(":")[-1]))
                except:
                    continue
                t = torch.Tensor(arr)
                path = os.path.abspath(os.path.join(spk_folder, key+".pt"))
                torch.save(t, path)
                spk_id_tensor[key] = path
    elif args.method == "mean":
        with open(args.training_files, "r", encoding="utf8") as ifile:
            metadata = ifile.readlines()
        print("create spk dict data")
        spk_dict = create_spk_dict(metadata)
        for key, value in spk_dict.items():
            t = torch.stack(value)
            # print(t.shape)
            t = torch.mean(t, dim=0)
            # print(t.shape)
            path = os.path.abspath(os.path.join(spk_folder, key+".pt"))
            torch.save(t, path)
            spk_id_tensor[key] = path
    return spk_id_tensor


def main():
    parser = argparse.ArgumentParser(
        description='merge datasets metadata')
    parser = parse_args(parser)
    import sys
    from hparams import default_args

    args = default_args
    args.extend(sys.argv[1:])
    args, _ = parser.parse_known_args(args)

    spk_id_tensor = get_trusted_tensors(args)
    with open(args.training_files, "r", encoding="utf8") as ifile:
        metadata = ifile.readlines()
    with open(args.training_files, "w", encoding="utf8") as ofile:
        for line in metadata:
            arr = line.strip().split("|")
            if len(arr) != 3:
                continue
            tensor_path = spk_id_tensor[get_spk_id(arr[0])]
            arr.insert(0, tensor_path)
            ofile.write("|".join(arr)+"\n")


if __name__ == "__main__":
    main()
