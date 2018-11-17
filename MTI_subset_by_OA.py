#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script subsets the 2013 MTI dataset for only those PMIDs that
# are also in the PMC Open Access file list

import pandas as pd

oa_list = pd.read_csv("oa_file_list.csv")

mti_train = open("2013_MTI_ML_DataSet/PMIDs_train", "r")
mti_train = mti_train.readlines()
mti_train = pd.DataFrame({'PMID':mti_train})

mti_test = open("2013_MTI_ML_DataSet/PMIDs_test", "r")
mti_test = mti_test.readlines()
mti_test = pd.DataFrame({'PMID':mti_test})

mti_oaSubset_train = oa_list[(oa_list.PMID.isin(mti_train.PMID))]
mti_oaSubset_train.to_csv("2013_MTI_in_OA_train.csv")

mti_oaSubset_test = oa_list[(oa_list.PMID.isin(mti_test.PMID))]
mti_oaSubset_test.to_csv("2013_MTI_in_OA_test.csv")