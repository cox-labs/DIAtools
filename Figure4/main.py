import json
import os

from matplotlib import pyplot as plt


class EvidencePeptideRecord:
    def __init__(self, pep, taxonomy, reverse):
        self.pep = pep
        self.taxonomy = taxonomy
        self.reverse = reverse


class EvidencePeptideTable:
    def __init__(self, file):
        self.evidences = []
        with open(file) as fs:
            header = fs.readline().rstrip().split('\t')
            n = len(header)
            indexes = {i: header.index(i) for i in ['PEP', 'Taxonomy IDs', 'Reverse', 'Sequence']}
            for line in fs:
                spl = line.rstrip().split('\t')
                if len(spl) != n:
                    continue
                self.evidences.append(
                    EvidencePeptideRecord(float(spl[indexes['PEP']]),
                                          spl[indexes['Taxonomy IDs']],
                                          spl[indexes['Reverse']] == '+'))

    @staticmethod
    def cut(data, maxvalue=0.05):
        for i in range(len(data) - 1, -1, -1):
            if data[i] < maxvalue:
                return data[:i]
        return data

    def internalFDR(self, params, maxvalue=0.05):
        humanId = params["human"]["id"]
        evidences = [self.evidences[i] for i in range(len(self.evidences))
                     if self.evidences[i].taxonomy == humanId]
        evidences.sort(key=lambda x: x.pep)
        fdrs = []
        cnts = []
        nreverse = 0
        for evidence in evidences:
            if evidence.reverse:
                nreverse += 1
            fdrs.append(nreverse / max(nreverse, len(fdrs) - nreverse + 1))
            cnts.append(len(fdrs) - nreverse)
        fdrs = self.cut(fdrs, maxvalue)
        return fdrs, cnts[:len(fdrs)]

    def externalFDR(self, params, maxvalue=0.05):
        humanId = params["human"]["id"]
        maizeId = params["maize"]["id"]
        fileSizeFactor = params["fileSizeFactor"]
        evidences = [self.evidences[i] for i in range(len(self.evidences))
                     if not self.evidences[i].reverse]
        evidences.sort(key=lambda i: i.pep)
        fdrs = []
        cnts = []
        nhuman = 0
        nmaize = 0
        for evidence in evidences:
            if evidence.taxonomy == humanId:
                nhuman += 1
            elif evidence.taxonomy == maizeId:
                nmaize += 1
            else:
                continue
            x = nmaize / fileSizeFactor
            fdrs.append(x / max(x, nhuman))
            cnts.append(nhuman)
        fdrs = self.cut(fdrs, maxvalue)
        return fdrs, cnts[:len(fdrs)]


class ProteinRecord:
    def __init__(self, score, species, reverse):
        self.score = score
        self.species = species
        self.reverse = reverse


class ProteinTable:
    def __init__(self, file):
        self.proteins = []
        with open(file) as fs:
            header = fs.readline().rstrip().split('\t')
            n = len(header)
            indexes = {i: header.index(i) for i in ['Gene names', 'Score', 'Species', 'Reverse']}
            for line in fs:
                spl = line.rstrip().split('\t')
                if len(spl) != n:
                    continue
                self.proteins.append(
                    ProteinRecord(float(spl[indexes['Score']]),
                                  spl[indexes['Species']],
                                  spl[indexes['Reverse']] == '+'))

    @staticmethod
    def cut(data, maxvalue=0.05):
        for i in range(100, len(data)):
            if data[i] >= maxvalue:
                return data[:i]
        return data

    def internalFDR(self, params, maxvalue=0.05):
        humanSpecies = params["human"]["speciesName"]
        proteins = [self.proteins[i] for i in range(len(self.proteins))
                    if self.proteins[i].species == humanSpecies]
        proteins.sort(key=lambda x: x.score, reverse=True)
        fdrs = []
        cnts = []
        nreverse = 0
        for protein in proteins:
            if protein.reverse:
                nreverse += 1
            fdrs.append(nreverse / max(nreverse, len(fdrs) - nreverse + 1))
            cnts.append(len(fdrs) - nreverse)
        fdrs = self.cut(fdrs, maxvalue)
        return fdrs, cnts[:len(fdrs)]

    def externalFDR(self, params, maxvalue=0.05):
        humanSpecies = params["human"]["speciesName"]
        maizeSpecies = params["maize"]["speciesName"]
        fileSizeFactor = params["fileSizeFactor"]
        proteins = [self.proteins[i] for i in range(len(self.proteins))
                    if not self.proteins[i].reverse]
        proteins.sort(key=lambda i: i.score, reverse=True)
        fdrs = []
        cnts = []
        nhuman = 0
        nmaize = 0
        for protein in proteins:
            if protein.species == humanSpecies:
                nhuman += 1
            elif protein.species == maizeSpecies:
                nmaize += 1
            else:
                continue
            x = nmaize / fileSizeFactor
            fdrs.append(x / max(x, nhuman))
            cnts.append(nhuman)
        fdrs = self.cut(fdrs, maxvalue)
        return fdrs, cnts[:len(fdrs)]


def plot_fdr_plot(mlStatus, libraryStatus, params, out_file):
    # miny = 1000
    # maxy = 3000000
    edge = 0.002
    minx = 0 - edge
    maxx = 0.05
    psms = ['PSM001_PROTEIN100', 'PSM100_PROTEIN100']
    colors = ["blue", "green", "red"]
    names = ["Matches", "Peptides", "Proteins"]
    files = ["evidence", "peptides", "proteinGroups"]
    classes = [EvidencePeptideTable, EvidencePeptideTable, ProteinTable]
    fig, ax = plt.subplots(1, 1, figsize=(params["figureSize"], params["figureSize"]))
    cnts = []
    for index in range(len(classes)):
        if names[index] == "Proteins":
            fullPath = f"{params[mlStatus][libraryStatus][psms[0]][files[index]]}"
        else:
            fullPath = f"{params[mlStatus][libraryStatus][psms[1]][files[index]]}"
        if not os.path.exists(fullPath):
            continue
        print(fullPath)
        data = classes[index](fullPath)
        print("Internal001")
        internalFDRx0, internalFDRy0 = data.internalFDR(params, 0.01)
        externalFDRx0, externalFDRy0 = data.externalFDR(params, 0.01)
        print(f"Internal: {len(internalFDRx0)}")
        print(f"External: {len(externalFDRx0)}")
        cnts.append(len(internalFDRx0))
        print("Internal005")
        internalFDRx, internalFDRy = data.internalFDR(params)
        print("External005")
        externalFDRx, externalFDRy = data.externalFDR(params)
        ax.plot(internalFDRx, internalFDRy, color=colors[index],
                label=f'{names[index]}/Internal FDR', linewidth=2, linestyle="-")
        ax.plot(externalFDRx, externalFDRy, color=colors[index],
                label=f'{names[index]}/External FDR', linewidth=2, linestyle=":")
    ax.grid(which='major', color="grey", alpha=0.5, linewidth=0.15, linestyle="-")
    ax.grid(which='minor', color="grey", alpha=0.5, linewidth=0.15, linestyle="-")
    for j in range(len(cnts)):
        ax.text(0.01, cnts[j], str(cnts[j]), color=colors[j], horizontalalignment="right")
    ax.axvline(x=0.01, color="black", alpha=0.5, linewidth=1.0, linestyle="-")
    ax.set_xlabel('Estimated FDR')
    ax.set_xlim((minx, maxx))
    ax.set_ylabel('Count')
    # ax.set_ylim((miny, maxy))
    ax.set_yscale('log')
    ax.set_title(libraryStatus)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    print(f"Done: {libraryStatus}")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    if os.path.isfile(out_file):
        os.remove(out_file)
    plt.savefig(out_file)


def plot(params):
    for mlStatus in ["ML", "noML"]:
        for libraryStatus in ["libraryDIA", "discoveryDIA"]:
            plot_fdr_plot(
                mlStatus,
                libraryStatus,
                params,
                os.path.join(params["outputFolder"],
                             "{}.{}".format(params[mlStatus][libraryStatus]["outputFile"], "pdf")))


if __name__ == "__main__":
    with open('parameters.json', 'r') as parameters_fs:
        plot(json.loads(parameters_fs.read()))
