import json
import math
import os
from statistics import mean

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde


class NgsData:
    def __init__(self, file):
        self.data = []
        self.genes = []
        with open(file) as fs:
            header = fs.readline()
            spl = header.rstrip().split('\t')
            data_index = spl.index("Mean")
            gene_index = spl.index("Gene name")
            for line in fs:
                if len(line) == 0 or line[0] == '#':
                    continue
                spl = line.rstrip().split('\t')
                self.data.append(float(spl[data_index]))
                self.genes.append(spl[gene_index])


class ProteinGroup:
    def __init__(self, file, type, pg2gene, mergeGenes=True, makeLog2=True):
        self.files = []
        self.lfq = None
        self.ibaq = None
        self.genes = []
        if type == "MaxQuant":
            with open(file) as fs:
                header = fs.readline()
                spl = header.rstrip().split('\t')
                pg_index = spl.index("Protein IDs")
                files = []
                lfq_indexes = []
                ibaq_indexes = []
                for i in range(len(spl)):
                    if spl[i].startswith("LFQ intensity "):
                        lfq_indexes.append(i)
                        files.append(int(spl[i].split()[-1]))
                    if spl[i].startswith("iBAQ ") and spl[i] != "iBAQ peptides":
                        ibaq_indexes.append(i)
                tuples = sorted(enumerate(files), key=lambda x: x[1])
                self.files = [t[1] for t in tuples]
                order = [t[0] for t in tuples]
                self.lfq = [[] for i in self.files]
                self.ibaq = [[] for i in self.files]
                reject_names = ["Only identified by site", "Reverse", "Potential contaminant"]
                reject_indexes = [spl.index(i) for i in reject_names]
                for line in fs:
                    spl = line.rstrip().split('\t')
                    if sum([spl[reject_index] == '+' for reject_index in reject_indexes]):
                        continue
                    genes = list(set([pg2gene[pg] for pg in spl[pg_index].split(';') if pg in pg2gene]))
                    if len(genes) == 0:
                        continue
                    genes.sort()
                    self.genes.append(genes[0])
                    for i in range(len(lfq_indexes)):
                        self.lfq[i].append(float(spl[lfq_indexes[order[i]]]))
                    for i in range(len(ibaq_indexes)):
                        self.ibaq[i].append(float(spl[ibaq_indexes[order[i]]]))
        elif type == "Spectronaut":
            with open(file) as fs:
                header = fs.readline()
                spl = header.rstrip().split('\t')
                pg_index = spl.index("PG.ProteinGroups")
                indexes = []
                files = []
                for i in range(len(spl)):
                    if spl[i].endswith(".raw.PG.Quantity"):
                        indexes.append(i)
                        files.append(int(spl[i].split('.')[0].split('_')[-1]))
                tuples = sorted(enumerate(files), key=lambda x: x[1])
                self.files = [t[1] for t in tuples]
                order = [t[0] for t in tuples]
                self.ibaq = [[] for i in self.files]
                for line in fs:
                    spl = line.rstrip().split('\t')
                    genes = list(set([pg2gene[pg] for pg in spl[pg_index].split(';') if pg in pg2gene]))
                    if len(genes) == 0:
                        continue
                    genes.sort()
                    self.genes.append(genes[0])
                    for i in range(len(indexes)):
                        k = spl[indexes[order[i]]]
                        if k == "Filtered":
                            self.ibaq[i].append(0.0)
                        else:
                            self.ibaq[i].append(float(k))
        if mergeGenes:
            genes = list(set(self.genes))
            gene2idx = {genes[i]: i for i in range(len(genes))}
            ibaq = [[0.0 for j in range(len(genes))] for i in range(len(self.files))]
            for i in range(len(self.files)):
                for j in range(len(self.genes)):
                    ibaq[i][gene2idx[self.genes[j]]] += self.ibaq[i][j]
            self.ibaq = ibaq
            if type == "MaxQuant":
                lfq = [[0.0 for j in range(len(genes))] for i in range(len(self.files))]
                for i in range(len(self.genes)):
                    for j in range(len(self.files)):
                        lfq[j][gene2idx[self.genes[i]]] += self.lfq[j][i]
                self.lfq = lfq
            self.genes = genes
        if makeLog2:
            for i in range(len(self.ibaq)):
                for j in range(len(self.ibaq[i])):
                    self.ibaq[i][j] = np.log2(self.ibaq[i][j])
            if type == "MaxQuant":
                for i in range(len(self.lfq)):
                    for j in range(len(self.lfq[i])):
                        self.lfq[i][j] = np.log2(self.lfq[i][j])


def getProteinGroup2Gene(fastaFiles):
    pg2gene = {}
    pname = ""
    gname = []
    for file in fastaFiles:
        with open(file) as fs:
            for line in fs:
                if line[0] == '>':
                    if len(gname) == 1:
                        pg2gene[pname] = gname[0]
                    pname = line.split('|')[1]
                    gname = [i[3:] for i in line.split(" ") if i.startswith("GN=")]
    if len(gname) == 1:
        pg2gene[pname] = gname[0]
    return pg2gene


def makeColours(vals):
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    colours = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
    return colours


def plot3ef(maxquant_data, maxquant_selection, spectronaut_data, spectronaut_selection, figSize, dotSize, outputFolder):
    for k, data, selection, title, measure, file_name in \
            [
                (0, maxquant_data.lfq, maxquant_selection, "MaxDIA", "LFQ intensity", "e"),
                (1, spectronaut_data.ibaq, spectronaut_selection, "Spectronaut", "Intensity", "f")
            ]:
        fig, ax = plt.subplots(1, 1, figsize=(figSize, figSize))
        yx = [(data[selection[0]][i], data[selection[1]][i])
              for i in range(len(data[selection[0]]))
              if not np.isinf(data[selection[0]][i]) and not np.isinf(data[selection[1]][i])]
        minv = yx[0][0]
        maxv = yx[0][0]
        for y, x in yx:
            minv = min(minv, min(x, y))
            maxv = max(maxv, max(x, y))
        xs = [x for x, y in yx]
        ys = [y for x, y in yx]
        values = np.vstack([xs, ys])
        z = gaussian_kde(values)(values)
        idx = z.argsort()
        ax.scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=dotSize)
        ax.set_title(title)
        ax.set_xlim((minv, maxv))
        ax.set_ylim((minv, maxv))
        ax.set_ylabel("Replicate {}, log2 {}".format(str(selection[0] + 1), measure))
        ax.set_xlabel("Replicate {}, log2 {}".format(str(selection[1] + 1), measure))
        r = list(range(math.ceil(minv), math.floor(maxv), 3))
        ax.set_xticks(r)
        ax.set_yticks(r)
        # ax.axline((1, 1), slope=1, ls="--", color="black", lw=1)
        # a, b = np.polyfit(xs, ys, 1)
        # X_plot = np.linspace(minv, maxv, 100)
        # plt.plot(X_plot, a * X_plot + b, '-')
        ax.text(minv + 0.1 * (maxv - minv), maxv - 0.1 * (maxv - minv),
                "Pearson $R$={:.3f}".format(scipy.stats.pearsonr(xs, ys)[0]), verticalalignment='top')
        fig.tight_layout()
        file = os.path.join(outputFolder, f"{file_name}.pdf")
        if os.path.isfile(file):
            os.remove(file)
        plt.savefig(file)


def plot3g(corr_table, selections, figSize, file):
    fig, ax = plt.subplots(1, 1, figsize=(figSize, figSize))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "darkblue"])
    im = ax.imshow(corr_table, cmap=cmap)
    for x, y in selections:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson $R^2$", rotation=-90, va="bottom")
    ax.set_axis_off()
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    for i in range(corr_table.shape[0]):
        im.axes.text(i, i, str(i + 1), **kw)
    ax.arrow(-1, -1, 0, 5, head_width=0.5, head_length=0.5, fc='k', ec='k')
    ax.text(-1, 5.5, 'Spectronaut', verticalalignment='top', horizontalalignment='center', fontsize=12, rotation=90)
    ax.arrow(-1, -1, 5, 0, head_width=0.5, head_length=0.5, fc='k', ec='k')
    ax.text(5.5, -1, 'MaxDIA', verticalalignment='center', horizontalalignment='left', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()
    if os.path.isfile(file):
        os.remove(file)
    plt.savefig(file)


def printDots(xs, ys, genes):
    for x, y, gene in zip(xs, ys, genes):
        if x > 37 and 20 < y < 24:
            print("First dot: {}".format(gene))
    for x, y, gene in zip(xs, ys, genes):
        if x < 26 and 26 < y:
            print("Second dot: {}".format(gene))


def plot3h(maxquant_data, spectronaut_data, min_count, figSize, dotSize, file):
    data = {}
    names = ["MaxDIA", "Spectronaut"]
    labels = ["log2 mean iBAQ intensity", "log2 mean intensity"]
    for name, raw_data, raw_genes in \
            [
                (names[0], maxquant_data.ibaq, maxquant_data.genes),
                (names[1], spectronaut_data.ibaq, spectronaut_data.genes)
            ]:
        data[name] = {}
        for i in range(len(raw_genes)):
            values = [raw_data[j][i] for j in range(len(raw_data)) if
                      not np.isinf(raw_data[j][i])]
            if len(values) < min_count:
                continue
            data[name][raw_genes[i]] = values
    xs = []
    ys = []
    genes = []
    for gene, values in data[names[0]].items():
        if gene in data[names[1]]:
            xvalues = data[names[0]][gene]
            yvalues = data[names[1]][gene]
            x0 = mean(xvalues)
            y0 = mean(yvalues)
            xs.append(x0)
            ys.append(y0)
            genes.append(gene)
    fig, ax = plt.subplots(1, 1, figsize=(figSize, figSize))
    values = np.vstack([xs, ys])
    z = gaussian_kde(values)(values)
    idx = z.argsort()
    ax.scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=dotSize)
    ax.set_title("Pearson R={:.3f}".format(scipy.stats.pearsonr(xs, ys)[0]))
    ax.set_xlabel("{}, {}".format(names[0], labels[0]))
    ax.set_ylabel("{}, {}".format(names[1], labels[1]))
    quantile = 0.005
    minx, maxx = sorted(xs)[int(quantile * len(xs))], sorted(xs)[int((1 - quantile) * len(xs))]
    ax.set_xlim(minx, maxx)
    ax.set_xticks(list(range(math.ceil(minx), math.floor(maxx), 3)))
    fig.tight_layout()
    if os.path.isfile(file):
        os.remove(file)
    plt.savefig(file)


def plot3ij(maxquant_data, spectronaut_data, min_count, ngs_data, figSize, dotSize, out_folder):
    data = {}
    raws = [maxquant_data, spectronaut_data]
    names = ["MaxDIA", "Spectronaut"]
    labels = ["log2 mean LFQ intensity", "log2 mean intensity"]
    for name, raw_data, genes in [(names[0], maxquant_data.ibaq, maxquant_data.genes),
                                  (names[1], spectronaut_data.ibaq, spectronaut_data.genes)]:
        data[name] = {}
        for i in range(len(genes)):
            values = [raw_data[j][i] for j in range(len(raw_data)) if
                      not np.isinf(raw_data[j][i])]
            if len(values) < min_count:
                continue
            data[name][genes[i]] = values
    MaxDIA = []
    Spectronaut = []
    ngs = []
    genes = []
    ngs_gene2idx = {ngs_data.genes[i]: i for i in range(len(ngs_data.genes))}
    for gene, values in data[names[0]].items():
        if gene in data[names[1]] and gene in ngs_gene2idx:
            xvalues = data[names[0]][gene]
            yvalues = data[names[1]][gene]
            MaxDIA.append(mean(xvalues))
            Spectronaut.append(mean(yvalues))
            ngs.append(ngs_data.data[ngs_gene2idx[gene]])
            genes.append(gene)
    for xs, ys, i, file_name in [(MaxDIA, ngs, 0, 'i'), (Spectronaut, ngs, 1, 'j')]:
        fig, ax = plt.subplots(1, 1, figsize=(figSize, figSize))
        values = np.vstack([xs, ys])
        z = gaussian_kde(values)(values)
        idx = z.argsort()
        ax.scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=dotSize)
        ax.set_xlabel("{}, {}".format(names[i], labels[i]))
        ax.set_ylabel("Caltech_HepG2_cell_PE_i200, RPKM")
        quantile = 0.01
        minx, maxx = sorted(xs)[int(quantile * len(xs))], sorted(xs)[int((1 - quantile) * len(xs))]
        ax.set_title("Pearson R={:.3f}".format(scipy.stats.pearsonr(xs, ys)[0]))
        fig.tight_layout()
        file = os.path.join(out_folder, f"{file_name}.pdf")
        if os.path.isfile(file):
            os.remove(file)
        plt.savefig(file)


def plot(params):
    ngsData = NgsData(params["NgsDataFile"])
    pg2gene = getProteinGroup2Gene(params["fastaFiles"])
    mqData = ProteinGroup(params["MaxQuantFile"], "MaxQuant", pg2gene)
    snData = ProteinGroup(params["SpectronautFile"], "Spectronaut", pg2gene)

    n = len(mqData.files)
    corr = [[0 for j in range(n)] for i in range(n)]
    minx = 1.0
    maxquant_corr = []
    spectronaut_corr = []
    for i in range(n):
        for j in range(i + 1, n):
            xy = [(mqData.lfq[i][k], mqData.lfq[j][k]) for k in range(len(mqData.lfq[i])) if
                  not np.isinf(mqData.lfq[i][k]) and not np.isinf(mqData.lfq[j][k])]
            corr[i][j] = scipy.stats.pearsonr([x for x, y in xy], [y for x, y in xy])[0] ** 2
            minx = min(corr[i][j], minx)
            maxquant_corr.append((i, j, corr[i][j]))
    maxquant_med_corr = sorted(maxquant_corr, key=lambda x: x[2])[len(maxquant_corr) // 2]
    for i in range(1, n):
        for j in range(0, i):
            xy = [(snData.ibaq[i][k], snData.ibaq[j][k]) for k in
                  range(len(snData.ibaq[i])) if
                  not np.isinf(snData.ibaq[i][k]) and not np.isinf(snData.ibaq[j][k])]
            corr[i][j] = scipy.stats.pearsonr([x for x, y in xy], [y for x, y in xy])[0] ** 2
            minx = min(corr[i][j], minx)
            spectronaut_corr.append((i, j, corr[i][j]))
    spectronaut_med_corr = sorted(spectronaut_corr, key=lambda x: x[2])[len(spectronaut_corr) // 2]
    for i in range(n):
        corr[i][i] = minx
    plot3ef(mqData, (maxquant_med_corr[0], maxquant_med_corr[1]), snData,
            (spectronaut_med_corr[0], spectronaut_med_corr[1]),
            params["figSize"], params["dotSize"], params["outputFolder"])
    plot3g(np.array(corr),
           [(maxquant_med_corr[1], maxquant_med_corr[0]), (spectronaut_med_corr[1], spectronaut_med_corr[0])],
           params["figSize"], os.path.join(params["outputFolder"], "g.pdf"))
    plot3h(mqData, snData, params["minCount"], params["figSize"], params["dotSize"], os.path.join(params["outputFolder"], "i.pdf"))
    plot3ij(mqData, snData, params["minCount"], ngsData, params["figSize"], params["dotSize"], params["outputFolder"])


if __name__ == "__main__":
    with open("parameters.json", 'r') as parameters_fs:
        plot(json.loads(parameters_fs.read()))
