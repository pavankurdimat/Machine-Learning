#---------------------------------------------------------------------#
#Program Name: main.py
#Description: This is the main program when all the other programs
#related to this project is called.
#---------------------------------------------------------------------#

import seaborn as sns
import matplotlib.pyplot as plt


from read_data import read_data
from describe_data import describe_data, display_null_counts
from clean_data import display_frequency_counts, remove_unwanted_columns
from eda import pair_plot, bar_chart
from fill_null_values import fill_null_values
from feature_engineering import feature_engineering
from prepare_data import prepare_data
from modelling import model_random_forests, model_linear_regression

import sys
import numpy as np

if __name__ == '__main__':

    print ("\n\n***************Project to Predict Used Car Prices Started***************")

    print ("\n\n-------------------Loading the Data Started-------------------")
    #Read Data from the file
    data = read_data('autos.csv')
    print("\n\nLoading the Data Completed....")

    print("\n\n-------------------Exploratory Analysis Started-------------------")
    #Describe Data : describe_data
    describe_data(data)

    #Describe Data : Display Null Values
    display_null_counts(data)

    #clean the data : Display frequency counts
    display_frequency_counts(data)

    # clean the data : Remove unwanted columns
    updated_data = remove_unwanted_columns(data)

    #Describe Data : describe_data
    describe_data(updated_data)

    #Describe Data : Display Null Values
    display_null_counts(updated_data)

    print("\n\n-------------------Preprocessing and Data Transformation Started-------------------")
    #eda : display pair_plot
    # pair_plot(updated_data)

    ##----------------------Fill Null Values-----------------------------------------##
    # notRepairedDamage
    # we can see 'nein' value is more frequent than 'ja' , so we can fillna with maxFreq
    #
    # fuelType
    # we can see only benzin and diesel are more frequent , for now lets fill it with 'benzin'
    #
    # gearbox
    # Null fuelTypes could be set to "not-declared" value
    #-----------------------------------------------------------
    #eds : barchart for notRepairedDamage
    bar_chart(updated_data['notRepairedDamage'], 'notRepairedDamage')
    #fill_null_values : fill no-value to the column notRepairedDamage
    fill_null_values(updated_data,'notRepairedDamage', 'nein' )
    # eds : barchart for notRepairedDamage
    bar_chart(updated_data['notRepairedDamage'], 'notRepairedDamage')
    # describe-data: display null counts
    display_null_counts(updated_data)
    # -----------------------------------------------------------
    # eds : barchart for vehicleType
    bar_chart(updated_data['vehicleType'], 'vehicleType')
    # fill_null_values : fill no-value to the column vehicleType
    fill_null_values(updated_data, 'vehicleType', 'no-value')
    # eds : barchart for vehicleType
    bar_chart(updated_data['vehicleType'], 'vehicleType')
    # describe-data: display null counts
    display_null_counts(updated_data)
    # -----------------------------------------------------------
    # eds : barchart for gearbox
    bar_chart(updated_data['gearbox'], 'gearbox')
    # fill_null_values : fill no-value to the column gearbox
    fill_null_values(updated_data, 'gearbox', 'no-value')
    # eds : barchart for gearbox
    bar_chart(updated_data['gearbox'], 'gearbox')
    # describe-data: display null counts
    display_null_counts(updated_data)
    # -----------------------------------------------------------
    # eds : barchart for model
    bar_chart(updated_data['model'], 'model')
    # fill_null_values : fill no-value to the column model
    fill_null_values(updated_data, 'model', 'no-value')
    # eds : barchart for model
    bar_chart(updated_data['model'], 'model')
    # describe-data: display null counts
    display_null_counts(updated_data)
    # -----------------------------------------------------------
    # eds : barchart for fuelType
    bar_chart(updated_data['fuelType'], 'fuelType')
    # fill_null_values : fill no-value to the column fuelType
    fill_null_values(updated_data, 'fuelType', 'benzin')
    # eds : barchart for fuelType
    bar_chart(updated_data['fuelType'], 'fuelType')
    # describe-data: display null counts
    display_null_counts(updated_data)

    #------------------------------FEATURE Engineering------------------------------#
    updated_data = feature_engineering(updated_data)

    #-----------------------------Prepare Data for Training------------------------#
    x_train, x_test, y_train, y_test = prepare_data(updated_data, 'price')

    # -----------------------------Modelling-Random Forests-------------------------#
    model_random_forests(x_train,y_train, x_test,y_test  )

    # -----------------------------Modelling-Linear Regression-------------------------#
    model_linear_regression(x_train, y_train, x_test, y_test)

    print('\n\n#----------------------THE END OF THE PROJECT----------------------#')