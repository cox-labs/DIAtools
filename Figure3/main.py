import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from statistics import mean, median


MIN_COUNT = 6
FIG_SIZE = 7
DOT_SIZE = 8
mean_median = "mean"
target_file = "proteinGroups.txt" #peptides.txt


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
                self.lfq  = [[] for i in self.files]
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
    # http://education.expasy.org/student_projects/isotopident/htdocs/aa-list.html
    aas = "ARNDCEQGHILKMFPSTWYV"
    masses = [71.0788, 156.1875, 114.1038, 115.0886, 103.1388, 129.1155, 128.1307, 57.0519, 137.1411, 113.1594,
              113.1594, 128.1741, 131.1926, 147.1766, 97.1167, 87.0782, 101.1051, 186.2132, 163.1760, 99.1326]
    aa2mass = {aa: mass for aa, mass in zip(aas, masses)}
    pg2gene = {}
    pg2mass = {}
    pname = ""
    gname = []
    seq = ""
    for file in fastaFiles:
        with open(file) as fs:
            for line in fs:
                if line[0] == '>':
                    if len(gname) == 1:
                        pg2gene[pname] = gname[0]
                        pg2mass[pname] = sum([aa2mass[aa] for aa in seq if aa in aa2mass]) - (len(seq)-1)*18
                    pname = line.split('|')[1]
                    gname = [i[3:] for i in line.split(" ") if i.startswith("GN=")]
                    seq = ""
                else:
                    seq += line.rstrip()
    if len(gname) == 1:
        pg2gene[pname] = gname[0]
        pg2mass[pname] = sum([aa2mass[aa] for aa in seq if aa in aa2mass]) - (len(seq) - 1) * 18
    return pg2gene, pg2mass


def makeColours(vals):
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    colours = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
    return colours


def plot3fg(maxquant_data, maxquant_selection, spectronaut_data, spectronaut_selection, out_folder):
    for k, data, selection, title, measure, file_name in \
            [
                (0, maxquant_data.lfq, maxquant_selection, "MaxDIA", "LFQ intensity", "f"),
                (1, spectronaut_data.ibaq, spectronaut_selection, "Spectronaut", "Intensity", "g")
            ]:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE, FIG_SIZE))
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
        ax.scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=DOT_SIZE)
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
        ax.text(minv + 0.1 * (maxv - minv), maxv - 0.1 * (maxv - minv), "Pearson $R^2$={:.3f}".format(
            round(scipy.stats.pearsonr([x for x, y in yx], [y for x, y in yx])[0] ** 2, 3)), verticalalignment='top')
        fig.tight_layout()
        file = "{}//{}.{}".format(out_folder, file_name, "pdf")
        if os.path.isfile(file):
            os.remove(file)
        plt.savefig(file)


def plot3h(corr_table, selections, out_folder):
    file_name = 'h'
    fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE, FIG_SIZE))
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
    file = "{}//{}.{}".format(out_folder, file_name, "pdf")
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


def plot3i(maxquant_data, spectronaut_data, min_count, out_folder):
    file_name = "i"
    data = {}
    names = ["MaxDIA", "Spectronaut"]
    if mean_median == "mean":
        w = "mean"
        f = mean
    else:
        w = "median"
        f = median
    labels = ["log2 {} iBAQ intensity".format(w), "log2 {} intensity".format(w)]
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
            x0 = f(xvalues)
            y0 = f(yvalues)
            xs.append(x0)
            ys.append(y0)
            genes.append(gene)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE, FIG_SIZE))
    values = np.vstack([xs, ys])
    z = gaussian_kde(values)(values)
    idx = z.argsort()
    ax.scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=DOT_SIZE)
    npxs = np.array([[x] for x in xs])
    npys = np.array([[y] for y in ys])
    #reg_b = LinearRegression(fit_intercept=False).fit(npxs, npys)
    reg_ab = LinearRegression(fit_intercept=True).fit(npxs, npys)
    ax.set_title("R^2={:.3f} y={}*x+{}".format(
        round(reg_ab.score(npxs, npys), 3), round(reg_ab.coef_[0][0], 2), round(reg_ab.intercept_[0], 2))
    )
    ax.set_xlabel("{}, {}".format(names[0], labels[0]))
    ax.set_ylabel("{}, {}".format(names[1], labels[1]))
    quantile = 0.005
    minx, maxx = sorted(xs)[int(quantile*len(xs))], sorted(xs)[int((1-quantile)*len(xs))]
    miny, maxy = reg_ab.coef_[0][0] * minx + reg_ab.intercept_[0], reg_ab.coef_[0][0] * maxx + reg_ab.intercept_[0]
    X_plot = np.linspace(minx, maxx, 100)
    plt.plot(X_plot, reg_ab.coef_[0][0] * X_plot + reg_ab.intercept_[0], '--', c='black')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks(list(range(math.ceil(minx), math.floor(maxx), 3)))
    ax.set_yticks(list(range(math.ceil(miny), math.floor(maxy), 3)))
    fig.tight_layout()
    file = "{}//{}.{}".format(out_folder, file_name, "pdf")
    if os.path.isfile(file):
        os.remove(file)
    plt.savefig(file)


def plot3jk(maxquant_data, spectronaut_data, min_count, ngs_data, out_folder):
    data = {}
    raws = [maxquant_data, spectronaut_data]
    names = ["MaxDIA", "Spectronaut"]
    if mean_median == "mean":
        w = "mean"
        f = mean
    else:
        w = "median"
        f = median
    labels = ["log2 {} LFQ intensity".format(w), "log2 {} intensity".format(w)]
    for name, raw_data, genes in [(names[0], maxquant_data.ibaq, maxquant_data.genes), (names[1], spectronaut_data.ibaq, spectronaut_data.genes)]:
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
            MaxDIA.append(f(xvalues))
            Spectronaut.append(f(yvalues))
            ngs.append(ngs_data.data[ngs_gene2idx[gene]])
            genes.append(gene)
    for xs, ys, i, file_name in [(MaxDIA, ngs, 0, 'j'), (Spectronaut, ngs, 1, 'k')]:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE, FIG_SIZE))
        values = np.vstack([xs, ys])
        z = gaussian_kde(values)(values)
        idx = z.argsort()
        #axs[i].set(aspect='equal')
        ax.scatter(np.array(xs)[idx], np.array(ys)[idx], color=makeColours(z[idx]), s=DOT_SIZE)
        ax.set_xlabel("{}, {}".format(names[i], labels[i]))
        ax.set_ylabel("Caltech_HepG2_cell_PE_i200, RPKM")
        quantile = 0.01
        minx, maxx = sorted(xs)[int(quantile*len(xs))], sorted(xs)[int((1-quantile)*len(xs))]
        miny, maxy = -1, sorted(ys)[int((1-quantile)*len(ys))]
        npxs = np.array([[x] for x in xs])
        npys = np.array([[y] for y in ys])
        reg_ab = LinearRegression(fit_intercept=True).fit(npxs, npys)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xticks(list(range(math.ceil(minx), math.floor(maxx), 3)))
        ax.set_yticks(list(range(math.ceil(miny), math.floor(maxy), 3)))
        ax.set_title("Pearson R={:.3f}".format(round(reg_ab.score(npxs, npys)**0.5, 3)))
        X_plot = np.linspace(minx, maxx, 100)
        plt.plot(X_plot, reg_ab.coef_[0][0] * X_plot + reg_ab.intercept_[0], '--', c='black')
        fig.tight_layout()

        file = "{}//{}.{}".format(out_folder, file_name, "pdf")
        if os.path.isfile(file):
            os.remove(file)
        plt.savefig(file)


# def write4ProteomicRuler(maxquant_data, spectronaut_data, min_count, file_out, pg2gene, pg2mass):
#     data = {}
#     raws = [maxquant_data, spectronaut_data]
#     names = ["MaxDIA", "Spectronaut"]
#     if mean_median == "mean":
#         w = "mean"
#         f = mean
#     else:
#         w = "median"
#         f = median
#     for j in range(len(names)):
#         name = names[j]
#         raw = raws[j]
#         for i in range(len(raw.genes)):
#             values = [raw.data[j][i] for j in range(len(raw.data)) if
#                       not np.isinf(raw.data[j][i])]
#             if raw.genes[i] not in data:
#                 data[raw.genes[i]] = [[] for i in range(len(names))]
#             data[raw.genes[i]][j] = values
#     gene2pg = {}
#     for pg, gene in pg2gene.items():
#         if gene not in gene2pg:
#             gene2pg[gene] = []
#         gene2pg[gene].append(pg)
#     with open(file_out, "w") as fs:
#         fs.write("{}.{}\t{}.{}\t{}\t{}\t{}\n".format(names[0], w, names[1], w, "gene", "proteins", "weight"))
#         for gene, d in data.items():
#             if len(d[0]) < min_count:
#                 a = 'NaN'
#             else:
#                 a = f(d[0])
#             if len(d[1]) < min_count:
#                 b = 'NaN'
#             else:
#                 b = f(d[1])
#             if a == 'NaN' and b == 'NaN':
#                 continue
#             # weight = mean([pg2mass[pg] for pg in gene2pg[gene]]) / 1000
#             weight = [pg2mass[pg] for pg in gene2pg[gene]][0] / 1000
#             proteins = ";".join(gene2pg[gene])
#             fs.write("{}\t{}\t{}\t{}\t{}\n".format(a, b, gene, proteins, weight))


def plot(maxquant_data, spectronaut_data, ngs_data, out_folder):
    n = len(maxquant_data.files)
    corr = [[0 for j in range(n)] for i in range(n)]
    minx = 1.0
    maxquant_corr = []
    spectronaut_corr = []
    for i in range(n):
        for j in range(i + 1, n):
            xy = [(maxquant_data.lfq[i][k], maxquant_data.lfq[j][k]) for k in range(len(maxquant_data.lfq[i])) if
                  not np.isinf(maxquant_data.lfq[i][k]) and not np.isinf(maxquant_data.lfq[j][k])]
            corr[i][j] = scipy.stats.pearsonr([x for x, y in xy], [y for x, y in xy])[0] ** 2
            minx = min(corr[i][j], minx)
            maxquant_corr.append((i, j, corr[i][j]))
    maxquant_med_corr = sorted(maxquant_corr, key=lambda x: x[2])[len(maxquant_corr) // 2]
    for i in range(1, n):
        for j in range(0, i):
            xy = [(spectronaut_data.ibaq[i][k], spectronaut_data.ibaq[j][k]) for k in
                  range(len(spectronaut_data.ibaq[i])) if
                  not np.isinf(spectronaut_data.ibaq[i][k]) and not np.isinf(spectronaut_data.ibaq[j][k])]
            corr[i][j] = scipy.stats.pearsonr([x for x, y in xy], [y for x, y in xy])[0] ** 2
            minx = min(corr[i][j], minx)
            spectronaut_corr.append((i, j, corr[i][j]))
    spectronaut_med_corr = sorted(spectronaut_corr, key=lambda x: x[2])[len(spectronaut_corr) // 2]
    for i in range(n):
        corr[i][i] = minx
    plot3fg(maxquant_data, (maxquant_med_corr[0], maxquant_med_corr[1]), spectronaut_data, (spectronaut_med_corr[0], spectronaut_med_corr[1]), out_folder)
    plot3h(np.array(corr), [(maxquant_med_corr[1], maxquant_med_corr[0]), (spectronaut_med_corr[1], spectronaut_med_corr[0])], out_folder)
    plot3i(maxquant_data, spectronaut_data, MIN_COUNT, out_folder)
    plot3jk(maxquant_data, spectronaut_data, MIN_COUNT, ngs_data, out_folder)


if __name__ == "__main__":
    root_folder = ""
    out_folder = ""
    ngs_file = "{}\\ngs\\Caltech_HepG2_cell_PE_i200.txt".format(root_folder)
    pg2gene, pg2mass = getProteinGroup2Gene(
        ["{}\\fasta\\{}".format(root_folder, file) for file in ["UP000005640_9606.fasta", "UP000005640_9606_additional.fasta"]])
    result = {
        "MaxQuant":    "{}\\MaxQuant.40021621\\{}".format(root_folder, target_file),
        "Spectronaut": "{}\\Spectronaut.13\\{}".format(root_folder, target_file)
    }
    data = {k: ProteinGroup(v, k, pg2gene) for k, v in result.items()}
    ngs_data = NgsData(ngs_file)
    plot(data["MaxQuant"], data["Spectronaut"], ngs_data, out_folder)
    # with open("{}\\{}".format(root_folder, "pg2mass.txt"), "w") as fs:
    #     fs.write("{}\t{}\n".format("protein", "mass"))
    #     for pg, mass in pg2mass.items():
    #         fs.write("{}\t{}\n".format(pg, round(mass/1000, 1)))
    #data = {k: ProteinGroup(v, k, pg2gene, makeLog2=True) for k, v in result.items()}
    #plot3e(data["MaxQuant"], data["Spectronaut"], 6)
    #write4ProteomicRuler(data["MaxQuant"], data["Spectronaut"], 6, "{}\\{}".format(root_folder, "forProteomicRuler.txt"), pg2gene, pg2mass)