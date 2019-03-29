
import matplotlib.pyplot as plt  # allows for plotting
import seaborn as sns
import pandas as pd
import numpy as np
import csv

from os.path import exists  # check that path exists
from os import mkdir  # directory operations
import src.visualization.constants as C


class IRTPlot:

    def generate_plots(self, headers: [str], barplot_name: str, barplot_data: [int],
                       lineplot_name: str, multi_data: [[int]]):
        """
        Generates plots from provided parameters
        :param headers: labels for files and plots
        :param barplot_name: name for barplot
        :param barplot_data: barplot_data values
        :param lineplot_name: name for lineplot
        :param multi_data: data for lineplot
        """
        self.export_data(barplot_name, headers, barplot_data)
        title = 'Total IRTs (' + str(C.MAX_ITERATIONS) + ' Steps)'
        self.barplot(barplot_name, y_label='Total IRT', img_title=title)
        self.lineplot(lineplot_name, headers, multi_data, img_title='IRT Paths')

    def lineplot(self, name: str, headers: [str], data: [[int]], img_title: str):
        """
        Creates a line plot
        :param name: name for files
        :param headers: labels for the plot
        :param data: if parameterized, the data is provided
        :param img_title: title for save file
        """
        total_lines = len(data)
        x = np.arange(0, len(data[0]))

        for line in range(total_lines):
            ax = sns.lineplot(x=x, y=data[line], color=C.IRT_COLORS[line], label=headers[line])

        ax.legend(loc='best')
        ax.set(xlabel='Steps', ylabel='IRT', title=img_title)
        self.save_plot(name)
        plt.show()

    def barplot(self, name: str, y_label: str, img_title: str):
        """
        Create a barplot from the parameterized data
        :param name: name for files
        :param y_label: y label's name
        :param img_title: title of saved file
        """
        path = C.TEST_DIR

        sns.set(style='whitegrid')
        sns.set_palette(sns.color_palette(C.IRT_COLORS))
        df = pd.read_csv(path + name + '.csv')
        ax = sns.barplot(data=df)
        ax.set(ylabel=y_label, title=img_title)

        self.save_plot(name)
        plt.show()

    @staticmethod
    def export_data(filename: str, headers: [str], data: int, text: bool=False):
        """
        Creates CSVs from the data for later use with pandas dataframes
        :param filename: the csv filename
        :param headers: the data headers
        :param data: the data to save
        :param text: True if data is composed of strings, False otherwise
        """
        path = C.TEST_DIR
        if not exists(path):
            mkdir(path)
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
        path = C.TEST_DIR

        # create directory if it doesn't exist
        if not exists(path):
            mkdir(path)

        # save plot to specified directory
        img = path + title + '.png'
        plt.savefig(img)
