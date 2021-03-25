import argparse
import datetime
import json
import logging
import os
import os.path
import re
import shutil

import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global TrainingFile
TrainingFile = ""


# Recording Losses after each iteration and saving it in file : output/train_information.txt
class RecordLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        file_training = open(TrainingFile, 'w')
        file_training.write("train_loss\tval_loss\n")
        file_training.close()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        file_training = open(TrainingFile, 'a')
        file_training.write(str(logs.get('loss')) + "\t" + str(logs.get('val_loss')) + "\n")
        file_training.close()


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, metrics, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.metrics = metrics


def getData(percentageTraining, randomSeed,
            trainingFiles, sequenceColumn, retentionTimeColumn,
            hyperParameters):
    dictData = dict()

    df_training = pd.DataFrame()
    for tf in trainingFiles:
        file_name = str(tf).replace('"', '')
        data = pd.read_csv(file_name, sep='\t')
        aux = data[[sequenceColumn, retentionTimeColumn]]
        df_training = df_training.append(aux, ignore_index=True).drop_duplicates()

    tokenizer = Tokenizer(num_words=None, char_level=True)
    tokenizer.fit_on_texts(df_training[sequenceColumn].values)
    dictionary = tokenizer.word_index

    X = tokenizer.texts_to_sequences(df_training[sequenceColumn].values)
    X = pad_sequences(X, maxlen=50)

    trainSize = int(percentageTraining) / 100.0

    maxy = max(df_training[retentionTimeColumn])
    miny = min(df_training[retentionTimeColumn])

    X_train, X_test_aux, Y_train, Y_test_aux = train_test_split(X, df_training[retentionTimeColumn],
                                                                test_size=trainSize, random_state=randomSeed)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_test_aux, Y_test_aux,
                                                                  test_size=0.50, random_state=randomSeed)

    X_train = X_train
    dictData["X_train"] = X_train
    dictData["X_test"] = X_test
    dictData["X_validation"] = X_validation

    dictData["Y_train"] = (Y_train - miny) / maxy
    dictData["Y_test"] = (Y_test - miny) / maxy
    dictData["Y_validation"] = (Y_validation - miny) / maxy

    print(f"Minimum/Maximum value {miny}/{maxy}")

    # getting Hyper Parameters
    dicParameters = dict()
    dicParameters["BatchSize"] = int(hyperParameters["batchSize"])
    dicParameters["LearningRate"] = float(hyperParameters["learningRate"])
    dicParameters["EmbeddingInput"] = int(hyperParameters["embeddingInput"])
    dicParameters["EmbeddingOutput"] = int(hyperParameters["embeddingOutput"])
    dicParameters["Dropout"] = int(hyperParameters["dropout"])
    dicParameters["LSTMUnits"] = int(hyperParameters["dimensionLSTMUnit"])
    dicParameters["RecurrentDropout"] = float(hyperParameters["recurrentDropout"])
    dicParameters["Epochs"] = int(hyperParameters["epochs"])
    dicParameters["Dictionary"] = dictionary

    return dictData, dicParameters, miny, maxy


def CreateRNN(data, hyperPar, tempFolder):
    batch_size = hyperPar["BatchSize"]
    embed_input = hyperPar["EmbeddingInput"]
    embed_output = hyperPar["EmbeddingOutput"]
    learn_rate = hyperPar["LearningRate"]
    lstm_out = hyperPar["LSTMUnits"]
    hdropout = hyperPar["Dropout"]
    hrecurrentdrop = hyperPar["RecurrentDropout"]
    hepochs = hyperPar["Epochs"]
    ### Changing the value of the global variable
    global TrainingFile
    TrainingFile = os.path.join(tempFolder, 'train_information.tsv')
    dictionaryFilename = os.path.join(tempFolder, 'dictionary.json')
    with open(dictionaryFilename, 'w') as outfile:
        json.dump(hyperPar["Dictionary"], outfile)

    ### RRN model
    model = Sequential()
    model.add(Embedding(embed_input, embed_output, input_length=50))
    model.add(Bidirectional(LSTM(lstm_out, return_sequences=True), name='Layer_Bidirectional_1'))
    model.add(
        LSTM(lstm_out, dropout=hdropout, recurrent_dropout=hrecurrentdrop, return_sequences=True, name='Layer_LSTM_1'))
    model.add(LSTM(lstm_out, dropout=hdropout, recurrent_dropout=hrecurrentdrop, name='Layer_LSTM_2'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='linear', name="LastLayer"))
    keras.optimizers.Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='mae', optimizer='adam')
    metrics_callback = MetricsCallback(validation_data=(data['X_validation'], data['Y_validation']), metrics=['mae'])

    # Folder for saving weights Model
    filepath = os.path.join(tempFolder, "weights-{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True,
                                 mode='auto', period=10)
    history = model.fit(data['X_train'], data['Y_train'], epochs=hepochs, batch_size=batch_size, verbose=2,
                        validation_data=(data['X_validation'], data['Y_validation']),
                        callbacks=[metrics_callback, checkpoint, RecordLosses])

    predictions = model.predict(data['X_test'], batch_size, verbose=1)
    now = str(str(datetime.datetime.now())[:20]).replace(" ", "").replace(":", "").replace(".", "")

    # Saving the model
    modelFilename = os.path.join(tempFolder, 'model.h5')
    model.save(modelFilename)
    # Saving testing value
    with open(os.path.join(tempFolder, 'test_information.tsv'), 'w') as file_testing:
        file_testing.write("real_value\tpredicted_value\n")
        x = np.array(data['Y_test'].values)  # *maxNum
        y = predictions.flatten()  # *maxNum
        for idx, val in enumerate(x):
            file_testing.write(str(x[idx]) + "\t" + str(y[idx]) + "\n")
    return modelFilename, dictionaryFilename


class Model:
    def __init__(self, modelFilename, dictionaryFilename):
        self.model = keras.models.load_model(modelFilename)
        self.dictionary = keras.models.load_model(dictionaryFilename)

    def predict_sequence(self, sequence, padding=50):
        sequence = list(sequence.strip())
        featurevector = [[0]] * padding
        i = padding - 1
        for a in sequence:
            featurevector[i] = [self.dictionary[a]]
            i = i - 1
        return self.model.predict(np.asarray([featurevector]).reshape(1, -1))


class Spectrum:
    def __init__(self, sequence, charge, intensities, types):
        self.sequence = sequence
        self.charge = charge
        self.intensities = intensities
        self.types = types
        self.proteinGroups = []
        self.rt = 0
        self.startPosition = 0
        self.missedCleavages = 0

    atom2mass = {
        'C': 12.0,
        'H': 1.00782503223,
        'O': 15.99491461956,
        'N': 14.00307400486,
        'S': 31.97207100
    }

    aa2formula = {
        'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2},
        'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2},
        'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3},
        'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4},
        # 'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},
        'C': {'C': 5, 'H': 10, 'N': 2, 'O': 3, 'S': 1},
        'Q': {'C': 5, 'H': 10, 'N': 2, 'O': 3},
        'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4},
        'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2},
        'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2},
        'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2},
        'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2},
        'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2},
        'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},
        'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2},
        'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2},
        'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3},
        'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3},
        'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2},
        'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3},
        'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2},
    }

    aa2mass = {aa: (sum([atom2mass[atom] * cnt for atom, cnt in formula.items()]) - 2 * atom2mass['H'] - atom2mass['O'])
               for aa, formula in aa2formula.items()}

    def get_masses(self):
        masses = []
        for ion in self.types:
            ionType = ion[0]
            ionPosition = int(ion[1:]) - 1
            mass = 0.0
            if ionType == 'y':
                mass += sum([self.aa2mass[aa] for aa in self.sequence[-(ionPosition + 1):]])
                mass += 3 * self.atom2mass['H'] + self.atom2mass['O']
            elif ionType == 'b':
                mass += sum([self.aa2mass[aa] for aa in self.sequence[:(ionPosition + 1)]])
                mass += self.atom2mass['H']
            else:
                raise Exception('Ions should by either y either b type')
            masses.append(mass)
        return masses

    @staticmethod
    def readDeepMassPrediction(filename, columns):
        return []

    @staticmethod
    def readWinnerPrediction(filename, columns):
        return []

    @staticmethod
    def _mergeFragmentsProsit(fragments):
        type2intensity = {}
        intensities = []
        types = []
        for fragment in fragments:
            if fragment[1] not in type2intensity:
                type2intensity[fragment[1]] = 0.0
            type2intensity[fragment[1]] += fragment[0]
        for t, i in type2intensity.items():
            types.append(t)
            intensities.append(i)
        return intensities, types

    @staticmethod
    def readPrositPrediction(filename, columns):
        records = []
        with open(filename, 'r') as ifs:
            header = ifs.readline().rstrip().split(',')
            fragmentIntensityIndex = header.index(columns["intensity"])
            fragmentTypeIndex = header.index(columns["type"])
            fragmentNumberIndex = header.index(columns["number"])
            peptideSequenceIndex = header.index(columns["sequence"])
            peptideChargeIndex = header.index(columns["charge"])
            peptideSequence = ""
            peptideCharge = ""
            fragments = []
            for line in ifs:
                spl = line.rstrip().split(',')
                if peptideSequence != spl[peptideSequenceIndex] or peptideCharge != spl[peptideChargeIndex]:
                    if peptideSequence != "":
                        intensities, types = Spectrum._mergeFragmentsProsit(fragments)
                        records.append(Spectrum(peptideSequence, peptideCharge, intensities, types))
                    peptideSequence = spl[peptideSequenceIndex]
                    peptideCharge = int(spl[peptideChargeIndex])
                    fragments.clear()
                fragments.append(
                    (
                        100000 * float(spl[fragmentIntensityIndex]),
                        f"{spl[fragmentTypeIndex]}{spl[fragmentNumberIndex]}"
                    )
                )
            if peptideSequence != "":
                intensities, types = Spectrum._mergeFragmentsProsit(fragments)
                records.append(Spectrum(peptideSequence, peptideCharge, intensities, types))
        return records

    @staticmethod
    def retentionTimePrediction(model, spectra, minRt, maxRt):
        sequences2rt = {}
        for spectrum in spectra:
            sequences2rt[spectrum.sequence] = 0.0
        for sequence in sequences2rt:
            sequences2rt[sequence] = (maxRt - minRt) * model.predict_sequence(sequence) + minRt
        for spectrum in spectra:
            spectrum.rt = sequences2rt[spectrum.sequence]

    @staticmethod
    def proteinGroupAssignment(spectra, fastaFiles, fastaIdParseRule, proteaseParseRule,
                               missedCleavageMin, missedCleavageMax,
                               lengthMin, lengthMax):
        peptides = {}
        for fastaFile in fastaFiles:
            record2sequence = {}
            name = ""
            with open(fastaFile) as file:
                for line in file:
                    if line.startswith('>'):
                        name = re.match(fastaIdParseRule, line).groups()[0]
                        record2sequence[name] = ""
                    else:
                        record2sequence[name] += line.rstrip()
            for name, sequence in record2sequence.items():
                inds = [-1] + [m.start(0) for m in re.finditer(proteaseParseRule, sequence)]
                for missedCleavage in range(missedCleavageMin, missedCleavageMax + 1):
                    for i in range(len(inds) - missedCleavage - 1):
                        start = inds[i] + 1
                        end = inds[i + missedCleavage + 1] + 1
                        if lengthMin <= (end - start) <= lengthMax:
                            s = sequence[start:end]
                            if s in peptides:
                                peptides[s].append((name, start, missedCleavage))
                            else:
                                peptides[s] = [(name, start, missedCleavage)]
        for spectrum in spectra:
            p = peptides[spectrum.sequence]
            spectrum.proteinGroups = [i[0] for i in p]
            spectrum.startPosition = p[0][1]
            spectrum.missedCleavages = p[0][2]

    @staticmethod
    def writeOutput(spectraList, outMsmsFile, outEvidenceFile, outPeptidesFile, fragmentation, massAnalyzer):
        with open(outMsmsFile, 'w') as msmsFile, \
                open(outEvidenceFile, 'w') as evidenceFile, \
                open(outPeptidesFile, 'w') as peptidesFile:
            msmsFile.write(
                '\t'.join(
                    [
                        "Fragmentation",
                        "Mass analyzer",
                        "Retention time",
                        "PEP",
                        "Score",
                        "Matches",
                        "Intensities",
                        "Masses",
                        "id"
                    ]
                ) + '\n')
            evidenceFile.write(
                '\t'.join(
                    [
                        "Sequence",
                        "Modified sequence",
                        "Proteins",
                        "MS/MS IDs",
                        "Raw file",
                        "Type",
                        "Reverse",
                        "Charge",
                        "Calibrated retention time",
                        "Calibrated retention time start",
                        "Calibrated retention time finish",
                        "Retention time",
                        "Intensity"
                    ]
                ) + '\n')
            peptidesFile.write(
                '\t'.join(
                    [
                        "Sequence",
                        "Unique (Proteins)",
                        "Leading razor protein",
                        "Start position",
                        "Missed cleavages"
                    ]
                ) + '\n')
            cnt = 0
            selectedPeptides = set()
            for spectrum in spectraList:
                msmsFile.write(
                    '\t'.join(
                        [
                            fragmentation,
                            massAnalyzer,
                            str(round(spectrum.rt, 3)),
                            "0.0",
                            "150.00",
                            ';'.join(spectrum.types),
                            ';'.join([str(round(i, 2)) for i in spectrum.intensities]),
                            ';'.join([str(round(i, 5)) for i in spectrum.get_masses()]),
                            str(cnt)
                        ]
                    ) + '\n')
                evidenceFile.write(
                    '\t'.join(
                        [
                            spectrum.sequence,
                            "_{}_".format(spectrum.sequence),
                            ";".join(spectrum.proteinGroups),
                            str(cnt),
                            "a.raw",
                            "MULTI-MSMS",
                            "",
                            str(spectrum.charge),
                            str(round(spectrum.rt, 3)),
                            str(round(spectrum.rt - 0.3, 3)),
                            str(round(spectrum.rt + 0.3, 3)),
                            str(round(spectrum.rt, 3)),
                            "10000"
                        ]
                    ) + '\n')
                if spectrum.sequence not in selectedPeptides:
                    peptidesFile.write(
                        '\t'.join(
                            [
                                spectrum.sequence,
                                "yes" if len(spectrum.proteinGroups) == 1 else "no",
                                ";".join(spectrum.proteinGroups),
                                str(spectrum.startPosition),
                                str(spectrum.missedCleavages)
                            ]
                        ) + '\n')
                    selectedPeptides.add(spectrum.sequence)
                cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor', choices=['deepmass', 'winner', 'prosit'])
    parser.add_argument('parameters', type=argparse.FileType('r'))
    args = parser.parse_args()
    predictor = args.predictor

    try:
        parameters = json.loads(args.parameters.read())

        tempFolder = parameters["output"]["tmpFolder"]
        if os.path.exists(tempFolder):
            shutil.rmtree(tempFolder)
        os.mkdir(tempFolder)

        handler = logging.FileHandler(os.path.join(tempFolder, 'postprocess.log'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        record_losses = RecordLosses()

        logger.info('GetData started')
        data, hyperPar, miny, maxy = getData(
            parameters["input"]["RTprediction"]["LSTMconfig"]["percentageTraining"],
            parameters["input"]["RTprediction"]["LSTMconfig"]["randomSeed"],
            parameters["input"]["RTprediction"]["evidence"]["trainingFiles"],
            parameters["input"]["RTprediction"]["evidence"]["columns"]["sequence"],
            parameters["input"]["RTprediction"]["evidence"]["columns"]["retentionTime"],
            parameters["input"]["RTprediction"]["LSTMconfig"]["hyperParameters"]
        )
        logger.info('GetData done')

        logger.info('ML training started')
        modelFilename, dictionaryFilename = CreateRNN(data, hyperPar, tempFolder)
        logger.info('ML training done')

        logger.info('Read MSMS prediction started')
        if predictor == "deepmass":
            spectra = Spectrum.readDeepMassPrediction(
                parameters["input"]["file"],
                parameters["input"]["predictors"]["deepmass"]["columns"]
            )
        elif predictor == "winner":
            spectra = Spectrum.readWinnerPrediction(
                parameters["input"]["file"],
                parameters["input"]["predictors"]["winner"]["columns"]
            )
        else:
            spectra = Spectrum.readPrositPrediction(
                parameters["input"]["file"],
                parameters["input"]["predictors"]["prosit"]["columns"]
            )
        logger.info('Read MSMS prediction done')

        logger.info('Predict RT started')
        Spectrum.retentionTimePrediction(Model(modelFilename, dictionaryFilename), spectra, miny, maxy)
        logger.info('Predict RT done')

        logger.info('Protein assignment started')
        Spectrum.proteinGroupAssignment(
            spectra,
            parameters["input"]["fasta"]["files"],
            parameters["input"]["fasta"]["idParseRule"]
        )
        logger.info('Protein assignment done')

        logger.info('Writing output started')
        Spectrum.writeOutput(
            spectra,
            parameters["output"]["msmsFile"],
            parameters["output"]["evidenceFile"],
            parameters["output"]["peptidesFile"],
            parameters["input"]["fragmentation"],
            parameters["input"]["massAnalyzer"]
        )
        logger.info('Writing output done')
    except Exception as e:
        logger.info(str(e))
        logger.info('>>> ERROR')
        print('ERROR: Check error log')
