"""
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
You can be released from the terms, and requirements of the Academic public license by purchasing a commercial license.
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle
import os
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, average_precision_score, precision_recall_curve, roc_auc_score, auc
from scipy.special import softmax
import scipy
import sys

print(tf.__version__)

model_name = "SVGP"
#number of Epochs for NN training
EPOCHS = 1000
#number of inducing points for SVGP
M = 50

dataset_name_list = ["balance-scale", "blood", "abalone", "annealing", "car", "contrac", "mammographic", "miniboone",
                    "wine", "lenses","breast-cancer-wisc-prog","haberman-survival","post-operative","spectf","plant-texture",
                    "pima","synthetic-control","iris","breast-tissue","conn-bench-vowel-deterding","ozone","oocytes_trisopterus_states_5b",
                    "twonorm","audiology-std","heart-switzerland","musk-2","spambase","lung-cancer","molec-biol-promoter","congressional-voting",
                    "conn-bench-sonar-mines-rocks","breast-cancer-wisc-diag","thyroid","spect","optical","arrhythmia","oocytes_merluccius_nucleus_4d",
                    "credit-approval", "cylinder-bands", "energy-y1", "energy-y2", "hill-valley", "image-segmentation", "led-display", "magic",
                    "cardiotocography-3clases", "chess-krvk", "chess-krvkp", "connect-4",
                    "Phishing","messidor","Bioconcentration","Climate","yeast",
                    "adult", "bank", "cardiotocography-10clases",
                    "nursery","oocytes_trisopterus_nucleus_2f","low-res-spect","ilpd-indian-liver","statlog-image","flags","semeion",
                    "wall-following","soybean","zoo","hayes-roth","plant-margin","hepatitis","wine-quality-red","parkinsons","wine-quality-white","mushroom",
                    "monks-3","breast-cancer","pittsburg-bridges-REL-L","statlog-heart","statlog-landsat","fertility","monks-1","statlog-vehicle",
                    "vertebral-column-3clases","ionosphere","pittsburg-bridges-TYPE","acute-nephritis","libras","horse-colic","oocytes_merluccius_states_2f","breast-cancer-wisc",
                    "pittsburg-bridges-MATERIAL","statlog-shuttle","waveform","steel-plates","statlog-german-credit","trains","statlog-australian-credit",
                    "acute-inflammation","page-blocks","molec-biol-splice","seeds","titanic","ringnorm","musk-1","glass","pittsburg-bridges-T-OR-D",
                    "planning","dermatology","monks-2","ecoli","primary-tumor","waveform-noise","teaching","lymphography","balloons","heart-cleveland",
                    "pendigits","plant-shape","letter","tic-tac-toe","echocardiogram","vertebral-column-2clases","heart-va","heart-hungarian","pittsburg-bridges-SPAN"]

def print_acc(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS):
    acc_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        acc_list.append(exp_result["rio_test_acc"])
    print("rio_test_acc for {} {}: {}".format(framework_variant, algo_spec+add_info, np.mean(acc_list)))


def print_mean_score(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS):
    mean_correct_list_train = []
    mean_incorrect_list_train = []
    mean_correct_list_test = []
    mean_incorrect_list_test = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        train_check = (exp_result["train_labels"]==np.argmax(exp_result["train_NN_predictions"], axis=1))
        test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
        mean_correct_list_train.append(np.mean(exp_result["mean_train"].reshape(-1)[np.where(train_check)]))
        mean_incorrect_list_train.append(np.mean(exp_result["mean_train"].reshape(-1)[np.where(train_check == False)]))
        mean_correct_list_test.append(np.mean(exp_result["mean"].reshape(-1)[np.where(test_check)]))
        mean_incorrect_list_test.append(np.mean(exp_result["mean"].reshape(-1)[np.where(test_check == False)]))

    print("{} {} train mean_correct: {}, mean_incorrect: {}".format(framework_variant, algo_spec+add_info, np.mean(mean_correct_list_train), np.mean(mean_incorrect_list_train)))
    print("{} {} test mean_correct: {}, mean_incorrect: {}".format(framework_variant, algo_spec+add_info, np.mean(mean_correct_list_test), np.mean(mean_incorrect_list_test)))
    return np.mean(mean_correct_list_train), np.mean(mean_incorrect_list_train)

def analysis_false_positive(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, bias):
    precision_list_train = []
    recall_list_train = []
    precision_list_test = []
    recall_list_test = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        train_check = (exp_result["train_labels"]==np.argmax(exp_result["train_NN_predictions"], axis=1))
        test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
        mean_correct_train = np.mean(exp_result["mean_train"].reshape(-1)[np.where(train_check)])
        mean_incorrect_train = np.mean(exp_result["mean_train"].reshape(-1)[np.where(train_check == False)])
        mean_correct_test = np.mean(exp_result["mean"].reshape(-1)[np.where(test_check)])
        mean_incorrect_test = np.mean(exp_result["mean"].reshape(-1)[np.where(test_check == False)])
        threshold = np.mean(mean_incorrect_train) - bias
        print("train threshold: {}".format(threshold))
        mean_detected = exp_result["mean"].reshape(-1)[np.where(exp_result["mean"].reshape(-1)<threshold)]
        num_detected = len(mean_detected)
        check_detected = test_check[np.where(exp_result["mean"].reshape(-1)<threshold)]
        num_false_positive_detected = len(check_detected[np.where(check_detected == False)])
        num_false_positive_total = len(test_check[np.where(test_check == False)])
        precision_list_train.append(float(num_false_positive_detected)/(float(num_detected)+sys.float_info.epsilon))
        recall_list_train.append(float(num_false_positive_detected)/(float(num_false_positive_total)+sys.float_info.epsilon))
        #print("precision: {}".format(precision_list_train[-1]))
        #print("recall: {}".format(recall_list_train[-1]))
        threshold = np.mean(mean_incorrect_test) - bias
        print("test threshold: {}".format(threshold))
        mean_detected = exp_result["mean"].reshape(-1)[np.where(exp_result["mean"].reshape(-1)<threshold)]
        num_detected = len(mean_detected)
        check_detected = test_check[np.where(exp_result["mean"].reshape(-1)<threshold)]
        num_false_positive_detected = len(check_detected[np.where(check_detected == False)])
        num_false_positive_total = len(test_check[np.where(test_check == False)])
        precision_list_test.append(float(num_false_positive_detected)/(float(num_detected)+sys.float_info.epsilon))
        recall_list_test.append(float(num_false_positive_detected)/(float(num_false_positive_total)+sys.float_info.epsilon))
        #print("precision: {}".format(precision_list_test[-1]))
        #print("recall: {}".format(recall_list_test[-1]))
    print("{} {} train mean precision: {}, mean recall: {}".format(framework_variant, algo_spec+add_info, np.mean(precision_list_train), np.mean(recall_list_train)))
    print("{} {} test mean precision: {}, mean recall: {}".format(framework_variant, algo_spec+add_info, np.mean(precision_list_test), np.mean(recall_list_test)))
    return np.mean(precision_list_train), np.mean(recall_list_train), np.mean(precision_list_test), np.mean(recall_list_test)

def AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name, NN_info="64+64"):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
            continue
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = -exp_result["mean"].reshape(-1)
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = exp_result["mean"].reshape(-1)
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for {}: {}".format(framework_variant, AP_list))
    return AP_list

def AP_calculation_topN(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, top_num, metric_name, NN_info="64+64"):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
            continue
        if framework_variant == "GP_inputOnly":
            result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
            with open(result_file_name, 'rb') as result_file:
                exp_result = pickle.load(result_file)
            if metric_name == "AP-error" or metric_name == "AUPR-error":
                y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
                y_score_test = -exp_result["mean"].reshape(-1)
            else:
                y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
                y_score_test = exp_result["mean"].reshape(-1)
            if metric_name == "AP-error" or metric_name == "AP-success":
                AP_list.append(average_precision_score(y_true_test, y_score_test))
            elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
                precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                AP_list.append(auc(recall, precision))
            elif metric_name == "AUROC":
                AP_list.append(roc_auc_score(y_true_test, y_score_test))
            #print("AP_list for {}: {}".format(framework_variant, AP_list))
        else:
            trial_num = 10
            max_difference = -100
            AP_list_tmp = []
            difference_list_test_tmp = []
            difference_list_train_tmp = []
            noise_variance_list_tmp = []
            signal_noise_ratio_list_tmp = []
            for trial in range(trial_num):
                result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trail{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
                with open(result_file_name, 'rb') as result_file:
                    exp_result_tmp = pickle.load(result_file)
                test_check = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                if metric_name == "AP-error" or metric_name == "AUPR-error":
                    y_true_test = (exp_result_tmp["test_labels"]!=np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = -exp_result_tmp["mean"].reshape(-1)
                else:
                    y_true_test = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = exp_result_tmp["mean"].reshape(-1)
                if metric_name == "AP-error" or metric_name == "AP-success":
                    AP_list_tmp.append(average_precision_score(y_true_test, y_score_test))
                elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
                    precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                    AP_list_tmp.append(auc(recall, precision))
                elif metric_name == "AUROC":
                    AP_list_tmp.append(roc_auc_score(y_true_test, y_score_test))
                difference_list_test_tmp.append(exp_result_tmp["mean_correct_test"] - exp_result_tmp["mean_incorrect_test"])
                difference_list_train_tmp.append(exp_result_tmp["mean_correct_train"] - exp_result_tmp["mean_incorrect_train"])
                noise_variance_list_tmp.append(exp_result_tmp["hyperparameter"][-1])
                signal_noise_ratio_list_tmp.append((exp_result_tmp["hyperparameter"][1]+exp_result_tmp["hyperparameter"][3])/exp_result_tmp["hyperparameter"][-1])
                add_info = "+separate_opt"
                result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trail{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
                with open(result_file_name, 'rb') as result_file:
                    exp_result_tmp = pickle.load(result_file)
                test_check = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                if metric_name == "AP-error" or metric_name == "AUPR-error":
                    y_true_test = (exp_result_tmp["test_labels"]!=np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = -exp_result_tmp["mean"].reshape(-1)
                else:
                    y_true_test = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = exp_result_tmp["mean"].reshape(-1)
                if metric_name == "AP-error" or metric_name == "AP-success":
                    AP_list_tmp.append(average_precision_score(y_true_test, y_score_test))
                elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
                    precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                    AP_list_tmp.append(auc(recall, precision))
                elif metric_name == "AUROC":
                    AP_list_tmp.append(roc_auc_score(y_true_test, y_score_test))
                difference_list_test_tmp.append(exp_result_tmp["mean_correct_test"] - exp_result_tmp["mean_incorrect_test"])
                difference_list_train_tmp.append(exp_result_tmp["mean_correct_train"] - exp_result_tmp["mean_incorrect_train"])
                noise_variance_list_tmp.append(exp_result_tmp["hyperparameter"][-1])
                signal_noise_ratio_list_tmp.append((exp_result_tmp["hyperparameter"][1]+exp_result_tmp["hyperparameter"][3])/exp_result_tmp["hyperparameter"][-1])
            #print("AP_list_test_tmp for {}: {}".format(framework_variant, AP_list_tmp))
            #print("difference_test for {}: {}".format(framework_variant, difference_list_test_tmp))
            #print("difference_train for {}: {}".format(framework_variant, difference_list_train_tmp))
            #print("noise_variance_list_tmp for {}: {}".format(framework_variant, noise_variance_list_tmp))
            #print("signal_noise_ratio_list_tmp for {}: {}".format(framework_variant, signal_noise_ratio_list_tmp))
            AP_list.append(np.mean(np.sort(AP_list_tmp, axis=None)[-top_num:]))
        #print("AP_list_top{} for {}: {}".format(top_num, framework_variant, AP_list))
        #print("AP_list_top{} mean for {}: {}".format(top_num, framework_variant, np.mean(AP_list)))
    return AP_list

def AP_class_max(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name, NN_info="64+64"):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
            continue
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        softmax_predictions = np.max(softmax(exp_result["test_NN_predictions"], axis=1), axis=1).reshape(-1)
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = -softmax_predictions
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = softmax_predictions
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for class max: {}".format(AP_list))
    return AP_list

def AP_class_difference(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name, NN_info="64+64"):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
            continue
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
        softmax_predictions = softmax(exp_result["test_NN_predictions"], axis=1)
        softmax_predictions_sorted = np.sort(softmax_predictions)
        class_diff = softmax_predictions_sorted[:,-1] - softmax_predictions_sorted[:,-2]
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = -class_diff
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = class_diff
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for class difference: {}".format(AP_list))
    return AP_list


def AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name, NN_info="64+64"):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
            continue
        trial = 0
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trail{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),SOTA_dir_name,('{}_exp_info_'+SOTA_algo_name+'_{}_run{}.pkl').format(dataset_name, NN_info, run))
        try:
            with open(result_file_name, 'rb') as result_file:
                SOTA_exp_info = pickle.load(result_file)
        except:
            print("skip {} run{} for {}".format(dataset_name, run, SOTA_algo_name))
            continue
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            if SOTA_algo_name == "TrustScore":
                y_score_test = -(SOTA_exp_info["trust_score_test"].reshape(-1))
            else:
                y_score_test = -(SOTA_exp_info["moderator_test_NN_predictions"].reshape(-1))
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            if SOTA_algo_name == "TrustScore":
                y_score_test = SOTA_exp_info["trust_score_test"].reshape(-1)
            else:
                y_score_test = SOTA_exp_info["moderator_test_NN_predictions"].reshape(-1)
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for {}: {}".format(SOTA_algo_name, AP_list))
    return AP_list

for dataset_index in range(len(dataset_name_list)):

    dataset_name = dataset_name_list[dataset_index]

    NN_size = "64+64"
    layer_width = 64
    RUNS = 10
    dir_name = "Results"

    metric_name_list = ["AP-error", "AP-success", "AUPR-error", "AUPR-success", "AUROC"]
    metric_name = metric_name_list[4]

    NN_info = NN_size

    print("Showing Results for {}".format(dataset_name))
    print("Showing Results for dir: {}".format(dir_name))

    acc_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        acc_list.append(exp_info["NN_test_acc"])
    print("NN_test_acc: {}".format(np.mean(acc_list)))

    AP_list_dict = {}
    AP_mean_dict = {}

    kernel_type = "RBF"
    framework_variant = "GP_inputOnly"
    algo_spec = "moderator_residual_target"
    add_info = ""

    AP_list_dict[framework_variant+"+"+algo_spec+add_info] = AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    AP_mean_dict[framework_variant+"+"+algo_spec+add_info] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info])

    label_max_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
            continue
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        label_max_list.append(np.max(exp_result["train_labels"]))
    print("max labels: {}".format(label_max_list))

    AP_list_dict["class_max"] = AP_class_max(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    AP_mean_dict["class_max"] = np.mean(AP_list_dict["class_max"])
    AP_list_dict["class_difference"] = AP_class_difference(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    AP_mean_dict["class_difference"] = np.mean(AP_list_dict["class_difference"])

    kernel_type = "RBF+RBF"
    framework_variant = "GP"
    top_num = 3
    AP_list_dict[framework_variant+"+"+algo_spec+add_info] = AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    AP_mean_dict[framework_variant+"+"+algo_spec+add_info] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info])
    AP_list_dict[framework_variant+"+"+algo_spec+add_info+"topN"] = AP_calculation_topN(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, top_num, metric_name)
    AP_mean_dict[framework_variant+"+"+algo_spec+add_info+"topN"] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info+"topN"])

    add_info = "+separate_opt"
    AP_list_dict[framework_variant+"+"+algo_spec+add_info] = AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    AP_mean_dict[framework_variant+"+"+algo_spec+add_info] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info])

    SOTA_dir_name = "Results"
    SOTA_algo_name = "CondifNet"
    AP_list_dict[SOTA_algo_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)
    AP_mean_dict[SOTA_algo_name] = np.mean(AP_list_dict[SOTA_algo_name])
    SOTA_algo_name = "Introspection-Net"
    AP_list_dict[SOTA_algo_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)
    AP_mean_dict[SOTA_algo_name] = np.mean(AP_list_dict[SOTA_algo_name])
    SOTA_dir_name = "Results"
    SOTA_algo_name = "TrustScore"
    AP_list_dict[SOTA_algo_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)
    AP_mean_dict[SOTA_algo_name] = np.mean(AP_list_dict[SOTA_algo_name])

    info = "All_residual_target_multirun"
    AP_list_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Statistics','{}_{}_{}.pkl'.format(metric_name, dataset_name, info))
    with open(AP_list_file_name, 'wb') as result_file:
        pickle.dump(AP_list_dict, result_file)
    print(AP_mean_dict)
