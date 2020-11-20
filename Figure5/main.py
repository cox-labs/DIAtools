import json
import math
import os
from statistics import mean

import matplotlib.pyplot as plt


class ProteinRecord:
    def __init__(self, proteinIds, intensities, lfqs, species):
        self.proteinIds = proteinIds
        self.intensities = intensities
        self.lfqs = lfqs
        self.species = species


class ProteinRecords:
    def __init__(self, records):
        self.records = records

    def get_ratios(self, species, minValidValues, quantType):
        x = []
        y = []
        ids = []
        for record in self.records:
            if record.species != species:
                continue
            if quantType == 'LFQ':
                tmp0 = [lfq for lfq in record.lfqs[0] if lfq != 0]
                tmp1 = [lfq for lfq in record.lfqs[1] if lfq != 0]
            else:
                tmp0 = [intens for intens in record.intensities[0] if intens != 0]
                tmp1 = [intens for intens in record.intensities[1] if intens != 0]
            if len(tmp0) < minValidValues or len(tmp1) < minValidValues:
                continue
            x.append(math.log2(mean(tmp0) / mean(tmp1)))
            y.append(math.log2(mean(tmp0) + mean(tmp1)))
            ids.append(record.proteinIds)
        return x, y, ids

    def get_counts(self, species, quantType):
        x = []
        for record in self.records:
            if record.species != species:
                continue
            if quantType == 'LFQ':
                tmp0 = [lfq for lfq in record.lfqs[0] if lfq != 0]
                tmp1 = [lfq for lfq in record.lfqs[1] if lfq != 0]
            else:
                tmp0 = [intens for intens in record.intensities[0] if intens != 0]
                tmp1 = [intens for intens in record.intensities[1] if intens != 0]
            if len(tmp0) + len(tmp1) == 0:
                continue
            x.append(len(tmp0) + len(tmp1))
        return x

    @staticmethod
    def read_protein_groups(filename, groupA, groupB):
        records = []
        with open(filename) as file_fs:
            header = file_fs.readline().rstrip().split('\t')
            proteinIds_index = header.index('Protein IDs')
            intensities_a_indexes = [header.index("Intensity {}".format(i)) for i in groupA]
            intensities_b_indexes = [header.index("Intensity {}".format(i)) for i in groupB]
            lfqs_a_indexes = [header.index("LFQ intensity {}".format(i)) for i in groupA]
            lfqs_b_indexes = [header.index("LFQ intensity {}".format(i)) for i in groupB]
            species_index = header.index('Species')
            filter_indexes = [header.index(i) for i in ['Only identified by site', 'Reverse', 'Potential contaminant']]
            for line in file_fs:
                spl = line.rstrip().split('\t')
                if sum([spl[filter_index] == '+' for filter_index in filter_indexes]) > 0 or spl[species_index] == '':
                    continue
                records.append(
                    ProteinRecord(spl[proteinIds_index],
                                  [[float(spl[i]) for i in intensities_a_indexes],
                                   [float(spl[i]) for i in intensities_b_indexes]],
                                  [[float(spl[i]) for i in lfqs_a_indexes],
                                   [float(spl[i]) for i in lfqs_b_indexes]],
                                  spl[species_index]))
        return ProteinRecords(records)


def plot_box(data, ax, params):
    ecoli_x, _, _ = data.get_ratios(params["ecoli"]["name"], params["minValidValue"], 'LFQ')
    yeast_x, _, _ = data.get_ratios(params["yeast"]["name"], params["minValidValue"], 'LFQ')
    human_x, _, _ = data.get_ratios(params["human"]["name"], params["minValidValue"], 'LFQ')
    ecoli_x0, _, _ = data.get_ratios(params["ecoli"]["name"], params["minValidValue"], 'Intensity')
    yeast_x0, _, _ = data.get_ratios(params["yeast"]["name"], params["minValidValue"], 'Intensity')
    human_x0, _, _ = data.get_ratios(params["human"]["name"], params["minValidValue"], 'Intensity')
    ecoli_c = data.get_counts(params["ecoli"]["name"], 'Intensity')
    yeast_c = data.get_counts(params["yeast"]["name"], 'Intensity')
    human_c = data.get_counts(params["human"]["name"], 'Intensity')
    for x, x0, cnt, species, level in zip([ecoli_x, yeast_x, human_x], [ecoli_x0, yeast_x0, human_x0],
                                          [ecoli_c, yeast_c, human_c], ["ecoli", "yeast", "human"], [0, 0.75, 1.5]):
        print(f"{species}: {len(x)}/{len(x0)}/{len(cnt)}")
        ax.boxplot(x, vert=False, positions=[level], widths=0.8,
                   showfliers=True,
                   patch_artist=True,
                   boxprops=dict(facecolor='white', color=params[species]["color"]),
                   capprops=dict(color=params[species]["color"]),
                   whiskerprops=dict(color=params[species]["color"]),
                   flierprops=dict(marker='.',
                                   markerfacecolor=params[species]["color"],
                                   markeredgecolor=None,
                                   markeredgewidth=0,
                                   markersize=params["dotSize"],
                                   alpha=params["dotAlpha"]),
                   medianprops=dict(color=params[species]["color"]))
        ax.axvline(params[species]["expectedValue"], color="black", ls='--', lw=1.5)
    ax.axis('off')


def plot_scatter(data, ax, params):
    ecoli_x, ecoli_y, _ = data.get_ratios(params["ecoli"]["name"], params["minValidValue"], 'LFQ')
    yeast_x, yeast_y, _ = data.get_ratios(params["yeast"]["name"], params["minValidValue"], 'LFQ')
    human_x, human_y, _ = data.get_ratios(params["human"]["name"], params["minValidValue"], 'LFQ')
    for x, y, species in zip([ecoli_x, yeast_x, human_x], [ecoli_y, yeast_y, human_y], ["ecoli", "yeast", "human"]):
        ax.scatter(x, y, marker='.', c=params[species]["color"], edgecolors=None, linewidths=0,
                   s=params["dotSize"] ** 2,
                   label=params[species]["name"], alpha=params["dotAlpha"])
        ax.axvline(params[species]["expectedValue"], color="black", ls='--', lw=1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(11.5, 24.5)
    ax.set_yticks([12, 16, 20, 24])
    ax.set_yticklabels([12, 16, 20, 24])
    ax.set_xlim(-4.2, 4.2)
    ax.set_xticks([-4, -2, 0, 1, 4])
    ax.set_xticklabels([-4, -2, 0, 1, 4])
    ax.set_xlabel('Log2(ratio)')
    ax.set_ylabel('Log2(summed intensity)')


def plot(params):
    for instrumentStatus in ["SCIEX", "timsTOF"]:
        for libraryStatus in ["libraryDIA", "discoveryDIA"]:
            data = ProteinRecords.read_protein_groups(params[instrumentStatus][libraryStatus]["proteinGroup"],
                                                      params[instrumentStatus]["groupA"],
                                                      params[instrumentStatus]["groupB"])
            out_file = os.path.join(params["outputFolder"],
                                    f"{params[instrumentStatus][libraryStatus]['outputFile']}.pdf")
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), sharex=True, squeeze=True,
                                    gridspec_kw={'height_ratios': [1, 6]})
            plt.subplots_adjust(hspace=0.0)
            plot_box(data, axs[0], params)
            plot_scatter(data, axs[1], params)
            if os.path.isfile(out_file):
                os.remove(out_file)
            plt.savefig(out_file)


if __name__ == "__main__":
    with open("parameters.json", 'r') as parameters_fs:
        plot(json.loads(parameters_fs.read()))
