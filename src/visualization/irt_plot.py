import os
os.chdir('..')

import matplotlib.pyplot as plt  # allows for plotting
import seaborn as sns
import pandas as pd
import csv

from os.path import exists  # check that path exists
from os import mkdir  # directory operations
import src.visualization.constants as C


class IRTPlot:

    def generate_plots(self, headers: [str], barplot_name: str, barplot_data: [int],
                       facetplot_name: str, facetplot_data: [[int]]):
        """
        Generates barplot from provided parameters
        :param headers: labels for files and plots
        :param barplot_name: name for barplot
        :param barplot_data: barplot_data values
        :param facetplot_name: name for facetplot
        :param facetplot_data: data for facetplot
        """
        self.export_data(barplot_name, headers, barplot_data)
        self.barplot(barplot_name, x_label='Algorithms', y_label='Total IRT', img_title='Total IRTs')
        self.export_multi_data(facetplot_name, headers, facetplot_data)
        self.facetplot_lines(facetplot_name, facetplot_data, img_title='IRT')

    def facetplot_lines(self, name: str, img_title: str):
        """
        Creates a facetplot
        :param name: name for files
        :param img_title: title for save file
        """
        path = C.CSV_DIR
        sns.set(style='whitegrid')
        sns.set_palette(sns.color_palette(C.BAR_COLORS))
        df = pd.read_csv(path + name + '.csv')
        ax = sns.barplot(data=df)
        ax.set(title=img_title)

        self.save_plot(name)
        plt.show()

    def barplot(self, name: str, x_label: str, y_label: str, img_title: str):
        """
        Create a barplot from the parameterized data
        :param name: name for files
        :param x_label: x label's name
        :param y_label: y label's name
        :param img_title: title of saved file
        """
        path = C.CSV_DIR
        sns.set(style='whitegrid')
        sns.set_palette(sns.color_palette(C.BAR_COLORS))
        df = pd.read_csv(path + name + '.csv')
        ax = sns.barplot(data=df)
        ax.set(xlabel=x_label, ylabel=y_label, title=img_title)

        self.save_plot(name)
        plt.show()

    @staticmethod
    def export_multi_data(filename: str, headers: [str], data: [int]):
        """
        Creates CSVs from the data for later use with pandas dataframes
        :param filename: the csv filename
        :param headers: the data headers
        :param data: list of data
        """
        path = C.CSV_DIR

        with open(path + filename + '.csv', 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
            wr.writerow(headers)
            for d in data:
                wr.writerow(d)

    @staticmethod
    def export_data(filename: str, headers: [str], data: int):
        """
        Creates CSVs from the data for later use with pandas dataframes
        :param filename: the csv filename
        :param headers: the data headers
        :param data: the data to save
        """
        path = C.CSV_DIR
        with open(path + filename + '.csv', 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
            wr.writerow(headers)
            wr.writerow(data)

    @staticmethod
    def save_plot(title: str):
        """
        Saves plot to specified directory.
        :param title: title of plot
        """
        path = C.CSV_DIR

        # create directory if it doesn't exist
        if not exists(path):
            mkdir(path)

        # save plot to specified directory
        img = path + title + '.png'
        plt.savefig(img)
