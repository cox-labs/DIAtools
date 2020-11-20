import json
import math
import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde


class NgsRecord:
    def __init__(self, rpkm, geneName):
        self.rpkm = rpkm
        self.geneName = geneName

    @staticmethod
    def read_rpkm(file, geneNameColumn, rpkmColumns):
        records = []
        with open(file) as fs:
            header = fs.readline()
            spl = header.rstrip().split('\t')
            rpkm_indexes = [spl.index(i) for i in rpkmColumns]
            geneName_index = spl.index(geneNameColumn)
            for line in fs:
                if len(line) == 0 or line[0] == '#':
                    continue
                spl = line.rstrip().split('\t')
                records.append(
                    NgsRecord(
                        mean([float(spl[i]) for i in rpkm_indexes]),
                        spl[geneName_index]
                    ))
        return records


class ProteomicRecord:
    def __init__(self, intensities, lfqs, geneNames, proteinIds):
        self.intensities = intensities
        self.lfqs = lfqs
        self.geneNames = geneNames
        self.proteinIds = proteinIds

    @staticmethod
    def read(file, experiments):
        records = []
        with open(file) as fs:
            header = fs.readline()
            spl = header.rstrip().split('\t')
            intensities_indexes = [spl.index("Intensity " + i) for i in experiments]
            lfqs_indexes = [spl.index("LFQ intensity " + i) for i in experiments]
            geneNames_index = spl.index("Gene names")
            proteinGroups_index = spl.index("Protein IDs")
            reject_names = ["Only identified by site", "Reverse", "Potential contaminant"]
            reject_indexes = [spl.index(i) for i in reject_names]
            for line in fs:
                spl = line.rstrip().split('\t')
                if sum([spl[reject_index] == '+' for reject_index in reject_indexes]):
                    continue
                intensities = []
                for i in intensities_indexes:
                    if spl[i] == "0":
                        intensities.append(-1)
                    else:
                        intensities.append(math.log2(float(spl[i])))
                lfqs = []
                for i in lfqs_indexes:
                    if spl[i] == "0":
                        lfqs.append(-1)
                    else:
                        lfqs.append(math.log2(float(spl[i])))
                records.append(ProteomicRecord(
                    intensities, lfqs,
                    spl[geneNames_index].split(';'),
                    spl[proteinGroups_index].split(';')
                ))
        return records


def makeColours(vals):
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    colours = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
    return colours


def plot6b(records, params, out_file):
    plt.style.use('classic')
    fig, axs = plt.subplots(2, 2, figsize=(params["figSize"], params["figSize"]), sharex=True, sharey=True,
                            gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
    names = ["Replicate 1", "Replicate 2", "Replicate 3"]
    for i0, j0, i, j in [(0, 0, 0, 2), (1, 0, 1, 2), (1, 1, 1, 0)]:
        axs[i0][j0].xaxis.grid(True, which='major')
        axs[i0][j0].yaxis.grid(True, which='major')
        xs_log10 = []
        ys_log10 = []
        xs = []
        ys = []
        for r in records:
            if r.lfqs[i] != -1 and r.lfqs[i] > 5 and r.lfqs[j] != -1 and r.lfqs[j] > 5:
                xs.append(2 ** r.lfqs[i])
                ys.append(2 ** r.lfqs[j])
                xs_log10.append(math.log10(2 ** r.lfqs[i]))
                ys_log10.append(math.log10(2 ** r.lfqs[j]))
        values = np.vstack([xs_log10, ys_log10])
        z = gaussian_kde(values)(values)
        idx = z.argsort()
        axs[i0][j0].scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=params["dotSize"])
        axs[i0][j0].text(10 ** 7, 10 ** 10, "R={:.3f}".format(scipy.stats.pearsonr(xs_log10, ys_log10)[0]))
        axs[i0][j0].set_yscale('log')
        axs[i0][j0].set_ylim(10 ** 6, 10 ** 11)
        axs[i0][j0].set_xscale('log')
        axs[i0][j0].set_xlim(10 ** 6, 10 ** 11)
        if j0 == 0:
            axs[i0][j0].set_ylabel(names[i] + ", LFQ")
        if i0 == 1:
            axs[i0][j0].set_xlabel(names[j] + ", LFQ")
    axs[0][1].axis('off')
    if os.path.isfile(out_file):
        os.remove(out_file)
    plt.savefig(out_file)


def bools2int(bs):
    s = 0
    for i in range(len(bs)):
        if bs[i]:
            s += 2 ** i
    return s


def int2bools(s, n):
    bs = []
    for i in range(n):
        bs.append((s & 2 ** i) > 0)
    return bs


def plot6c(data, libraries, colors, width, ymax, ylabel, out_file):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.01},
                            figsize=(5, 5), constrained_layout=True)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].axes.get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(True)
    axs[1].set_frame_on(False)
    xs = list(range(len(data)))
    sortedxs = sorted(data.items(), key=lambda kv: sum(kv[1]), reverse=True)
    for i in range(len(sortedxs)):
        if sortedxs[i][0] in colors:
            axs[0].bar([i], sum(sortedxs[i][1]), width, color=colors[sortedxs[i][0]])
        else:
            axs[0].bar([i], sum(sortedxs[i][1]), width, color="black")
        y = sum(sortedxs[i][1])
        axs[0].text(i, y, str(y), ha='center', va='bottom')
    axs[0].set_ylim(0, ymax)
    axs[0].set_ylabel(ylabel)
    ys = list(range(len(libraries)))
    for i in range(len(ys)):
        if i % 2 == 0:
            axs[1].axhspan(ys[i] - 0.5, ys[i] + 0.5, color='whitesmoke')
    markersize = 8
    for j in range(len(xs)):
        bools = int2bools(sortedxs[j][0], len(ys))
        mini = len(ys)
        maxi = -1
        for i in range(len(ys)):
            if bools[i]:
                mini = min(mini, i)
                maxi = max(maxi, i)
        if maxi != -1:
            axs[1].plot([xs[j], xs[j]], [ys[mini], ys[maxi]], color='black', linewidth=2.0)
        for i in range(len(ys)):
            if bools[i]:
                axs[1].plot([xs[j]], [ys[i]], 'o', color='black', markersize=markersize)
            else:
                axs[1].plot([xs[j]], [ys[i]], 'o', color='lightgrey', markersize=markersize)
    axs[1].set_ylim(ys[0] - 0.5, ys[-1] + 0.5)
    axs[1].set_yticks(ys)
    axs[1].set_yticklabels(libraries)
    if os.path.isfile(out_file):
        os.remove(out_file)
    plt.savefig(out_file)


def plot6d(data, minx, maxx, colors, order, out_file):
    names = list(range(minx, maxx + 1))
    x = list(range(maxx + 1 - minx))
    tmp = [0 for i in range(minx, maxx + 1)]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), constrained_layout=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for k in order:
        ax.bar(x, data[k], bottom=tmp, color=colors[k], width=1)
        for j in range(len(tmp)):
            tmp[j] += data[k][j]
    ax.set_ylim(0, 2500)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("Gene expression, Log2 RPKM")
    ax.set_ylabel("Gene count")
    if os.path.isfile(out_file):
        os.remove(out_file)
    plt.savefig(out_file)


def plot6cd(sDDA, fDDA, discovery, ngs, params):
    colors = {bools2int(i["comb"]): i["color"] for i in params["colorOrder"]}
    minx = params["ngsData"]["rpkmMin"]
    maxx = params["ngsData"]["rpkmMax"]
    gene_bool = {record.geneName: [False, False, False, True] for record in ngs}  # sDDA fDDA discovery NGS
    gene_rpkm = {record.geneName: record.rpkm for record in ngs}
    dataset = [sDDA, fDDA, discovery]
    for i in range(len(dataset)):
        for record in dataset[i]:
            for geneName in record.geneNames:
                if geneName in gene_bool:
                    gene_bool[geneName][i] = True
    cntsPerRpkm = {}
    for geneName, bools in gene_bool.items():
        x = int(round(gene_rpkm[geneName]))
        if x < minx or x > maxx:
            continue
        i = bools2int(bools)
        if i not in cntsPerRpkm:
            cntsPerRpkm[i] = [0 for i in range(minx, maxx + 1)]
        cntsPerRpkm[i][x - minx] += 1
    order = [bools2int(i["comb"]) for i in params["colorOrder"]]
    plot6c(cntsPerRpkm, params["order"], colors, 0.8, 6500, "Gene count", os.path.join(params["outputFolder"], "c.pdf"))
    plot6d(cntsPerRpkm, minx, maxx, colors, order, os.path.join(params["outputFolder"], "d.pdf"))


def plot(params):
    ngs_data = NgsRecord.read_rpkm(params["ngsData"]["file"],
                                   params["ngsData"]["geneNameColumn"],
                                   params["ngsData"]["rpkmColumns"])

    singleShotLibrary = ProteomicRecord.read(params["fractionatedDIA"]["Single-Shot DDA Library"],
                                             params["experiments"])
    fractionatedLibrary = ProteomicRecord.read(params["fractionatedDIA"]["Fractionated DDA Library"],
                                               params["experiments"])
    discoveryLibrary = ProteomicRecord.read(params["fractionatedDIA"]["Discovery Library"], params["experiments"])
    plot6b(discoveryLibrary, params, os.path.join(params["outputFolder"], "b.pdf"))
    plot6cd(singleShotLibrary, fractionatedLibrary, discoveryLibrary, ngs_data, params)


if __name__ == "__main__":
    with open("parameters.json", 'r') as parameters_fs:
        plot(json.loads(parameters_fs.read()))
