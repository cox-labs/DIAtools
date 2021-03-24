import argparse
import json
import re



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
