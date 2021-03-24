import argparse
import json
import re


def digestPeptides(fastaFiles,
                   lengthMin, lengthMax,
                   missedCleavageMin, missedCleavageMax,
                   fastaIdParseRule, proteaseParseRule):
    peptides = set()
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
                        peptides.add(sequence[start:end])
    return peptides


def writePrositInput(peptides, outputFile,
                     chargeMin, chargeMax,
                     collisionEnergy):
    with open(outputFile, 'w') as file:
        file.write(f"modified_sequence,collision_energy,precursor_charge\n")
        for peptide in peptides:
            for charge in range(chargeMin, chargeMax + 1):
                file.write(f"{peptide},{collisionEnergy},{charge}\n")


def writeDeepMassInput(peptides, outputFile,
                       chargeMin, chargeMax,
                       fragmentation, massAnalyzer):
    with open(outputFile, 'w') as file:
        file.write(f"ModifiedSequence,Charge,Fragmentation,MassAnalyzer\n")
        for peptide in peptides:
            for charge in range(chargeMin, chargeMax + 1):
                file.write(f"{peptide},{charge},{fragmentation},{massAnalyzer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor', choices=['deepmass', 'winner', 'prosit'])
    parser.add_argument('parameters', type=argparse.FileType('r'))
    args = parser.parse_args()
    predictor = args.predictor

    try:
        parameters = json.loads(args.parameters.read())

        peptides = digestPeptides(
            parameters["input"]["fastaFiles"],
            parameters["input"]["length"]["min"],
            parameters["input"]["length"]["max"],
            parameters["input"]["missedCleavage"]["min"],
            parameters["input"]["missedCleavage"]["max"],
            parameters["input"]["fastaIdParseRule"],
            parameters["input"]["proteaseParseRule"]
        )
        if predictor == "deepmass" or predictor == "winner":
            writeDeepMassInput(
                peptides,
                parameters["output"]["file"],
                parameters["input"]["charge"]["min"],
                parameters["input"]["charge"]["max"],
                parameters["input"]["deepmass"]["fragmentation"],
                parameters["input"]["deepmass"]["massAnalyzer"]
            )
        else:
            writePrositInput(
                peptides,
                parameters["output"]["file"],
                parameters["input"]["charge"]["min"],
                parameters["input"]["charge"]["max"],
                parameters["input"]["prosit"]["collisionEnergy"]
            )
    except Exception as e:
        print(f"Exception: {e}")
