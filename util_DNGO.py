"""
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
You can be released from the terms, and requirements of the Academic public license by purchasing a commercial license.
"""
from __future__ import absolute_import, division, print_function

import pandas as pd

import os
import numpy as np
import time
from scipy.io import arff

# file that contains functions to read dataset and run RIO variants

def load_UCI121(dataset_name):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'UCI121_data',dataset_name,'{}_py.dat'.format(dataset_name))
    normed_dataset = pd.read_csv(dataset_path, header=None, sep=",")
    label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'UCI121_data',dataset_name,'labels_py.dat')
    labels = pd.read_csv(label_path, header=None).astype(int)
    return normed_dataset.dropna(), labels.dropna()

def dataset_read(dataset_name):
    if dataset_name == "yacht":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','yacht_hydrodynamics.data')
        column_names = ['Longitudinal position of the center of buoyancy','Prismatic coefficient','Length-displacement ratio','Beam-draught ratio','Length-beam ratio','Froude number','Residuary resistance']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, sep=' +', engine='python')
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "ENB_heating":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','ENB2012_data.xlsx')
        raw_dataset = pd.read_excel(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('Y2')
    elif dataset_name == "ENB_cooling":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','ENB2012_data.xlsx')
        raw_dataset = pd.read_excel(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('Y1')
    elif dataset_name == "airfoil_self_noise":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','airfoil_self_noise.dat')
        column_names = ['Frequency','Angle of attack','Chord length','Free-stream velocity','Suction side displacement thickness','sound pressure']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, sep="\t")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "concrete":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Concrete_Data.xls')
        raw_dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "winequality-red":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','winequality-red.csv')
        raw_dataset = pd.read_csv(dataset_path, sep = ';').astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "winequality-white":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','winequality-white.csv')
        raw_dataset = pd.read_csv(dataset_path, sep = ';').astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "CCPP":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Combined_Cycle_Power_Plant.xlsx')
        raw_dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "CASP":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','CASP.csv')
        raw_dataset = pd.read_csv(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "SuperConduct":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','SuperConduct.csv')
        raw_dataset = pd.read_csv(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "slice_localization":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','slice_localization_data.csv')
        raw_dataset = pd.read_csv(dataset_path) + 0.01
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('patientId')
    elif dataset_name == "MSD":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','YearPredictionMSD.txt')
        raw_dataset = pd.read_csv(dataset_path, sep=",", header=None)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset = dataset.astype(float)
    elif dataset_name == "Climate":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','pop_failures.dat')
        raw_dataset = pd.read_csv(dataset_path, sep=' +', engine='python')
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "Bioconcentration":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Grisoni_et_al_2016_EnvInt88.csv')
        raw_dataset = pd.read_csv(dataset_path, sep="\t")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "messidor":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','messidor_features.arff')
        data = arff.loadarff(dataset_path)
        raw_dataset = pd.DataFrame(data[0]).astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "Phishing":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','PhishingData.arff')
        data = arff.loadarff(dataset_path)
        raw_dataset = pd.DataFrame(data[0]).astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset["Result"] += 1
    elif dataset_name == "yeast":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','yeast.data')
        raw_dataset = pd.read_csv(dataset_path, header=None, sep=' +', engine='python')
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop(0)
        class_dic = {}
        count_dic = {}
        for i in range(len(dataset[9].values)):
            if dataset[9].values[i] in class_dic:
                dataset.set_value(i, 9, class_dic[dataset[9].values[i]])
                count_dic[dataset[9].values[i]] += 1
            else:
                class_dic[dataset[9].values[i]] = len(class_dic)
                dataset.set_value(i, 9, len(class_dic)-1)
                count_dic[dataset[9].values[i]] = 1
        print("count_dic: {}".format(count_dic))
    return dataset

