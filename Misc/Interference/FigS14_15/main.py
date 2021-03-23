import json
import math
import os
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np


class RatioPoint:
    def __init__(self, taxonomy, ratioSamples, ratio, intensity):
        self.taxonomy = taxonomy
        self.ratioSamples = ratioSamples
        self.ratio = ratio
        self.intensity = intensity


class GeneRatioPoint(RatioPoint):
    def __init__(self, geneName, taxonomy, ratioSamples, ratio, intensity):
        super().__init__(taxonomy, ratioSamples, ratio, intensity)
        self.geneName = geneName

    @staticmethod
    def readLfqData(filename, species2protein2record, groupColumns, minValidValue):
        species2protein2gene = {species: {protein: species2protein2record[species][protein].geneName for protein in
                                          species2protein2record[species]} for species in species2protein2record}
        records = []
        with open(filename) as file_fs:
            header = file_fs.readline().rstrip().split('\t')
            proteinIds_index = header.index('Protein IDs')
            lfq_indexes = {}
            for groupName in groupColumns:
                lfq_indexes[groupName] = {}
                for groupReplicateName in groupColumns[groupName]:
                    lfq_indexes[groupName][groupReplicateName] = \
                        header.index(f"LFQ intensity {groupReplicateName}")
            species_index = header.index('Species')
            filter_indexes = [header.index(i) for i in ['Only identified by site', 'Reverse', 'Potential contaminant']]
            for line in file_fs:
                spl = line.rstrip().split('\t')
                if sum([spl[filter_index] == '+' for filter_index in filter_indexes]) > 0 or spl[species_index] == '':
                    continue
                lfqs = {}
                for groupName in groupColumns:
                    lfqs[groupName] = {}
                    for groupReplicateName in groupColumns[groupName]:
                        lfqs[groupName][groupReplicateName] = \
                            float(spl[lfq_indexes[groupName][groupReplicateName]])
                species = spl[species_index]
                geneNames = list(set([species2protein2gene[species][proteinId]
                                      for proteinId in spl[proteinIds_index].split(';')
                                      if proteinId in species2protein2gene[species]]))
                if len(geneNames) == 1:
                    samples = list(groupColumns.keys())
                    for i in range(len(samples)):
                        for j in range(i + 1, len(samples)):
                            groupA = samples[i]
                            tmpA = [lfqs[groupA][groupReplicateName] for groupReplicateName in
                                    lfqs[groupA]
                                    if lfqs[groupA][groupReplicateName] != 0]
                            groupB = samples[j]
                            tmpB = [lfqs[groupB][groupReplicateName] for groupReplicateName in
                                    lfqs[groupB]
                                    if lfqs[groupB][groupReplicateName] != 0]
                            if len(tmpA) < minValidValue or len(tmpB) < minValidValue:
                                continue
                            records.append(GeneRatioPoint(geneNames[0], spl[species_index], f"{groupA}_{groupB}",
                                                          math.log2(statistics.mean(tmpA) / statistics.mean(tmpB)),
                                                          statistics.mean(
                                                              [statistics.mean(tmpA), statistics.mean(tmpB)])))
        return records

    @staticmethod
    def correctOnSpeciesSamples(records, subtractionSpecies):
        subtractionFactor0 = {}
        for record in records:
            if record.taxonomy == subtractionSpecies:
                if record.ratioSamples not in subtractionFactor0:
                    subtractionFactor0[record.ratioSamples] = []
                subtractionFactor0[record.ratioSamples].append(record.ratio)
        subtractionFactor = {}
        for sample, data in subtractionFactor0.items():
            subtractionFactor[sample] = statistics.median(data)
        for record in records:
            record.ratio = record.ratio - subtractionFactor[record.ratioSamples]

    @staticmethod
    def writeJson(records, filename):
        with open(filename, "w") as file:
            for record in records:
                file.write(json.dumps(record.__dict__) + "\n")

    @staticmethod
    def readJson(filename):
        records = []
        with open(filename) as file:
            for line in file:
                records.append(GeneRatioPoint(**(json.loads(line))))
        return records


class PeptideRatioPoint(RatioPoint):
    def __init__(self, modPeptideSequence, peptideSequence, charge, taxonomy, ratioSamples, ratio, intensity):
        super().__init__(taxonomy, ratioSamples, ratio, intensity)
        self.modPeptideSequence = modPeptideSequence
        self.peptideSequence = peptideSequence
        self.charge = charge

    @staticmethod
    def readAvantGardeFile(filename, modPeptideSequenceColumn, peptideSequenceColumn, chargeColumn, taxonomyColumn,
                           ratioSamplesColumn, ratioColumn, intensityColumn, taxonomyTranslation):
        peptideRecords = []
        with open(filename) as file:
            header = file.readline().rstrip().split()
            modPeptideSequenceIndex = header.index(modPeptideSequenceColumn)
            peptideSequenceIndex = header.index(peptideSequenceColumn)
            chargeIndex = header.index(chargeColumn)
            taxonomyIndex = header.index(taxonomyColumn)
            ratioSamplesIndex = header.index(ratioSamplesColumn)
            ratioIndex = header.index(ratioColumn)
            intensityIndex = header.index(intensityColumn)
            for line in file:
                spl = line.rstrip().split()
                if spl[ratioIndex] == "NA" or spl[intensityIndex] == "NA":
                    continue
                peptideRecords.append(
                    PeptideRatioPoint(
                        spl[modPeptideSequenceIndex],
                        spl[peptideSequenceIndex],
                        float(spl[chargeIndex]),
                        taxonomyTranslation[spl[taxonomyIndex]],
                        spl[ratioSamplesIndex],
                        float(spl[ratioIndex]),
                        float(spl[intensityIndex])
                    )
                )
        return peptideRecords

    @staticmethod
    def convert2geneRatioPoints(peptideRatioPoints, species2protein2record):
        gene2ratioSamples2peptideRatioPoints = {}
        cnt0 = 0
        cnt1 = 0
        for peptideRatioPoint in peptideRatioPoints:
            protein2record = species2protein2record[peptideRatioPoint.taxonomy]
            proteinNames = FastaRecord.searchPeptide(
                protein2record,
                peptideRatioPoint.peptideSequence
            )
            geneNames = set()
            for proteinName in proteinNames:
                geneNames.add(protein2record[proteinName].geneName)
            if len(geneNames) == 0:
                cnt0 += 1
            elif len(geneNames) == 1:
                geneName = list(geneNames)[0]
                ratioSamples = peptideRatioPoint.ratioSamples
                if geneName not in gene2ratioSamples2peptideRatioPoints:
                    gene2ratioSamples2peptideRatioPoints[geneName] = {}
                if ratioSamples not in gene2ratioSamples2peptideRatioPoints[geneName]:
                    gene2ratioSamples2peptideRatioPoints[geneName][ratioSamples] = []
                gene2ratioSamples2peptideRatioPoints[geneName][ratioSamples].append(peptideRatioPoint)
            else:
                cnt1 += 1
        result = []
        for geneName, ratioSamples2peptideRatioPoints in gene2ratioSamples2peptideRatioPoints.items():
            for ratioSample, points in ratioSamples2peptideRatioPoints.items():
                ratios = [point.ratio for point in points]
                intensities = [point.intensity for point in points]
                result.append(GeneRatioPoint(geneName, points[0].taxonomy, ratioSample,
                                             statistics.median(ratios), statistics.median(intensities)))
        print(cnt0, cnt1, len(peptideRatioPoints))
        return result


class FastaRecord:
    def __init__(self, proteinName, geneName, sequence):
        self.proteinName = proteinName
        self.geneName = geneName
        self.sequence = sequence

    @staticmethod
    def readFastaFiles(filenames):
        proteinPattern = re.compile(r">(tr|sp)\|(.*)\|")
        genePattern = re.compile(r"GN=([^\s]*)")
        protein2record = {}
        for filename in filenames:
            with open(filename) as file:
                proteinName = ""
                geneName = ""
                sequence = []
                for line in file:
                    if line.startswith(">"):
                        if proteinName != "" and proteinName is not None and geneName is not None:
                            protein2record[proteinName] = FastaRecord(proteinName, geneName, str.join("", sequence))
                        t = proteinPattern.search(line)
                        if t:
                            proteinName = t.group(2)
                        else:
                            proteinName = None
                        t = genePattern.search(line)
                        if t:
                            geneName = t.group(1)
                        else:
                            geneName = None
                        sequence.clear()
                    else:
                        sequence.append(line.rstrip())
                if proteinName != "" and proteinName is not None and geneName is not None:
                    protein2record[proteinName] = FastaRecord(proteinName, geneName, str.join("", sequence))
        return protein2record

    @staticmethod  # TODO too slow solution; use BWT instead
    def searchPeptide(protein2record, peptide):
        proteinNames = []
        for proteinName, fastaRecord in protein2record.items():
            if peptide in fastaRecord.sequence:
                proteinNames.append(proteinName)
        return proteinNames


def plot_box(data, ax, params, species):
    samples = list(params["input"]["data"]["samples"].keys())
    k = 0
    labels = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            y = [v.ratio for v in data if v.ratioSamples == f"{samples[i]}_{samples[j]}"]
            label = f"{samples[i]}/{samples[j]}"
            labels.append(label)
            color = params["input"]["data"]["ratios"]["colors"][label]
            ax.boxplot(y, vert=True, positions=[k], widths=0.8,
                       showfliers=True,
                       patch_artist=True,
                       boxprops=dict(facecolor='white', color=color),
                       capprops=dict(color=color),
                       whiskerprops=dict(color=color),
                       flierprops=dict(marker='.',
                                       markerfacecolor=color,
                                       markeredgecolor=None,
                                       markeredgewidth=0,
                                       markersize=params["plot"]["dotSize"],
                                       alpha=params["plot"]["dotAlpha"]),
                       medianprops=dict(color=color))
            expectedRatio = math.log2(params["input"]["data"]["samples"][samples[i]]["proportions"][species] /
                                      params["input"]["data"]["samples"][samples[j]]["proportions"][species])
            ax.axhline(expectedRatio, color=color, ls='--', lw=0.5)
            k += 1
    ax.set_ylim(-5.2, 5.2)
    ax.set_yticks([])
    # ax.set_xlim(14, 36)
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=90)


def plot_scatter(data, ax, params, species):
    samples = list(params["input"]["data"]["samples"].keys())
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sampleRatioName = f"{samples[i]}_{samples[j]}"
            x = [math.log2(v.intensity) for v in data if v.ratioSamples == sampleRatioName]
            y = [v.ratio for v in data if v.ratioSamples == sampleRatioName]
            label = f"{samples[i]}/{samples[j]}"
            color = params["input"]["data"]["ratios"]["colors"][label]
            ax.scatter(x, y, marker='.', c=color,
                       edgecolors=None, linewidths=0,
                       s=params["plot"]["dotSize"] ** 2,
                       label=label, alpha=params["plot"]["dotAlpha"])
            expectedRatio = math.log2(params["input"]["data"]["samples"][samples[i]]["proportions"][species] /
                                      params["input"]["data"]["samples"][samples[j]]["proportions"][species])
            ax.axhline(expectedRatio, color=color, ls='--', lw=0.5)
    ax.text(15, -5, species, horizontalalignment="left", verticalalignment="bottom")
    ax.set_ylim(-5.2, 5.2)
    ax.set_yticks([-3, -1, 0, 1, 3])
    ax.set_yticklabels([-3, -1, 0, 1, 3])
    ax.set_ylabel('Log2(ratio)')
    ax.set_xlim(14, 36)
    ax.set_xticks([15, 20, 25, 30, 35])
    ax.set_xticklabels([15, 20, 25, 30, 35])
    ax.set_xlabel('Log2(ratio)')
    ax.set_xlabel('Log2(mean intensity)')


def plot(points, params, out_file):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 6), squeeze=True,
                            gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    speciesList = list(params["input"]["species"].keys())
    for i in range(len(speciesList)):
        data = [point for point in points if point.taxonomy == speciesList[i]]
        plot_scatter(data, axs[i][0], params, speciesList[i])
        plot_box(data, axs[i][1], params, speciesList[i])
    if os.path.isfile(out_file):
        os.remove(out_file)
    plt.savefig(out_file)


# TODO remove 3 repeats == copy-pasta
def plotMetricsValidRatios(ax, avantGardePoints, maxquantPoints, params):
    ax.grid(True)
    speciesList = list(params["input"]["species"].keys())
    samplesList = list(params["input"]["data"]["samples"].keys())
    avantGardeResult = []
    maxquantResult = []
    labels = []
    for k in range(len(speciesList)):
        avantGardeData = [point for point in avantGardePoints if point.taxonomy == speciesList[k]]
        maxquantPointsData = [point for point in maxquantPoints if point.taxonomy == speciesList[k]]
        for i in range(len(samplesList)):
            for j in range(i + 1, len(samplesList)):
                sampleRatioName = f"{samplesList[i]}_{samplesList[j]}"
                avantGardeY = [v.ratio for v in avantGardeData if v.ratioSamples == sampleRatioName]
                maxquantY = [v.ratio for v in maxquantPointsData if v.ratioSamples == sampleRatioName]
                avantGardeResult.append(len(avantGardeY))
                maxquantResult.append(len(maxquantY))
                labels.append(f"{sampleRatioName}")
        avantGardeResult.append(0)
        maxquantResult.append(0)
        labels.append(" ")
    X = np.arange(len(labels))
    ax.bar(X - 0.20, avantGardeResult, color="#f1ad1e", width=0.35)
    ax.bar(X + 0.20, maxquantResult, color="#207ca5", width=0.35)
    ax.set_xticks(X)
    ax.set_xticklabels(labels)
    ax.set_ylabel('# Valid Ratios')


def plotMetricsStandardDeviation(ax, avantGardePoints, maxquantPoints, params):
    ax.grid(True)
    speciesList = list(params["input"]["species"].keys())
    samplesList = list(params["input"]["data"]["samples"].keys())
    avantGardeResult = []
    maxquantResult = []
    labels = []
    for k in range(len(speciesList)):
        avantGardeData = [point for point in avantGardePoints if point.taxonomy == speciesList[k]]
        maxquantPointsData = [point for point in maxquantPoints if point.taxonomy == speciesList[k]]
        for i in range(len(samplesList)):
            for j in range(i + 1, len(samplesList)):
                sampleRatioName = f"{samplesList[i]}_{samplesList[j]}"
                avantGardeY = [v.ratio for v in avantGardeData if v.ratioSamples == sampleRatioName]
                maxquantY = [v.ratio for v in maxquantPointsData if v.ratioSamples == sampleRatioName]
                avantGardeResult.append(statistics.stdev(avantGardeY))
                maxquantResult.append(statistics.stdev(maxquantY))
                labels.append(f"{sampleRatioName}")
        avantGardeResult.append(0)
        maxquantResult.append(0)
        labels.append(" ")
    X = np.arange(len(labels))
    ax.bar(X - 0.20, avantGardeResult, color="#f1ad1e", width=0.35)
    ax.bar(X + 0.20, maxquantResult, color="#207ca5", width=0.35)
    ax.set_xticks(X)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Standard Deviation')


def plotMetricsExpectedActual(ax, avantGardePoints, maxquantPoints, params):
    ax.grid(True)
    speciesList = list(params["input"]["species"].keys())
    samplesList = list(params["input"]["data"]["samples"].keys())
    avantGardeResult = []
    maxquantResult = []
    labels = []
    for k in range(len(speciesList)):
        avantGardeData = [point for point in avantGardePoints if point.taxonomy == speciesList[k]]
        maxquantPointsData = [point for point in maxquantPoints if point.taxonomy == speciesList[k]]
        for i in range(len(samplesList)):
            for j in range(i + 1, len(samplesList)):
                sampleRatioName = f"{samplesList[i]}_{samplesList[j]}"
                avantGardeY = statistics.median([v.ratio for v in avantGardeData if v.ratioSamples == sampleRatioName])
                maxquantY = statistics.median(
                    [v.ratio for v in maxquantPointsData if v.ratioSamples == sampleRatioName])
                expectedRatio = math.log2(
                    params["input"]["data"]["samples"][samplesList[i]]["proportions"][speciesList[k]] /
                    params["input"]["data"]["samples"][samplesList[j]]["proportions"][speciesList[k]])
                avantGardeResult.append(abs(avantGardeY - expectedRatio))
                maxquantResult.append(abs(maxquantY - expectedRatio))
                labels.append(f"{sampleRatioName}")
        avantGardeResult.append(0)
        maxquantResult.append(0)
        labels.append(" ")
    X = np.arange(len(labels))
    ax.bar(X - 0.20, avantGardeResult, color="#f1ad1e", width=0.35)
    ax.bar(X + 0.20, maxquantResult, color="#207ca5", width=0.35)
    ax.set_xticks(X)
    ax.set_xticklabels(labels)
    ax.set_ylabel('|Expected-Observed Median|')


def plotMetrics(avantGardePoints, maxquantPoints, params, outFile):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 6), squeeze=True)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plotMetricsValidRatios(axs[0], avantGardePoints, maxquantPoints, params)
    plotMetricsStandardDeviation(axs[1], avantGardePoints, maxquantPoints, params)
    plotMetricsExpectedActual(axs[2], avantGardePoints, maxquantPoints, params)
    if os.path.isfile(outFile):
        os.remove(outFile)
    plt.savefig(outFile)


def readParameters(filename):
    with open(filename, 'r') as parametersFileStream:
        parameters = json.loads(parametersFileStream.read())
    return parameters


def readAvantGardeResults(jsonFilename, txtFilename, columns, species):
    if os.path.exists(jsonFilename):
        geneRatioPoints = GeneRatioPoint.readJson(jsonFilename)
    else:
        peptideRatioPoints = PeptideRatioPoint.readAvantGardeFile(
            txtFilename,
            columns["modPeptideSequence"],
            columns["peptideSequence"],
            columns["charge"],
            columns["taxonomy"],
            columns["ratioSamples"],
            columns["ratio"],
            columns["intensity"],
            species
        )
        geneRatioPoints = PeptideRatioPoint.convert2geneRatioPoints(peptideRatioPoints, fastaRecords)
        GeneRatioPoint.writeJson(geneRatioPoints, jsonFilename)
    return geneRatioPoints


def readMaxQuantResults(jsonFilename, txtFilename, groupColumns, minValidValue):
    if os.path.exists(jsonFilename):
        geneRatioPoints = GeneRatioPoint.readJson(jsonFilename)
    else:
        geneRatioPoints = GeneRatioPoint.readLfqData(txtFilename, fastaRecords, groupColumns, minValidValue)
        GeneRatioPoint.writeJson(geneRatioPoints, jsonFilename)
    return geneRatioPoints


def intersectGenes(avantGardeResults, maxQuantResults):
    tmp = {}
    for record in avantGardeResults:
        if record.ratioSamples not in tmp:
            tmp[record.ratioSamples] = set()
        tmp[record.ratioSamples].add(record.geneName)
    intersection = {key: set() for key in tmp}
    for record in maxQuantResults:
        if record.geneName in tmp[record.ratioSamples]:
            intersection[record.ratioSamples].add(record.geneName)

    avantGardeResults0 = []
    for record in avantGardeResults:
        if record.geneName in intersection[record.ratioSamples]:
            avantGardeResults0.append(record)

    maxQuantResults0 = []
    for record in maxQuantResults:
        if record.geneName in intersection[record.ratioSamples]:
            maxQuantResults0.append(record)

    return avantGardeResults0, maxQuantResults0


if __name__ == "__main__":
    parametersFile = "parameters.json"
    correctionSpecies = "Homo sapiens"

    parameters = readParameters(parametersFile)
    fastaRecords = {species: FastaRecord.readFastaFiles(parameters["input"]["species"][species]["fasta"])
                    for species in parameters["input"]["species"]}

    avantGardeName = "avant-garde"
    avantGardeResults = readAvantGardeResults(
        parameters["input"]["data"]["avant-garde"]["files"][avantGardeName]["json"],
        parameters["input"]["data"]["avant-garde"]["files"][avantGardeName]["txt"],
        parameters["input"]["data"]["avant-garde"]["columns"],
        parameters["input"]["data"]["avant-garde"]["species"]
    )
    GeneRatioPoint.correctOnSpeciesSamples(avantGardeResults, correctionSpecies)
    plot(avantGardeResults, parameters, parameters["output"][avantGardeName])

    maxQuantName = "tq001.mixed.mc1"
    maxQuantResults = readMaxQuantResults(
        parameters["input"]["data"]["maxquant"][maxQuantName]["json"],
        parameters["input"]["data"]["maxquant"][maxQuantName]["txt"],
        {
            groupName: parameters["input"]["data"]["samples"][groupName]["columns"]
            for groupName in parameters["input"]["data"]["samples"]
        },
        parameters["plot"]["minValidValue"]

    )
    GeneRatioPoint.correctOnSpeciesSamples(maxQuantResults, correctionSpecies)
    plot(maxQuantResults, parameters, parameters["output"][maxQuantName])

    plotMetrics(avantGardeResults, maxQuantResults, parameters, parameters["output"]["common.metrics"])
