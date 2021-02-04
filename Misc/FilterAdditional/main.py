import json
import re
import sys

parse_expression = ">.*\|(.*)\|"

def read_protein_names(filename_fasta, pattern):
    name2sequence = {}
    name = ""
    sequence = ""
    with open(filename_fasta) as fs:
        for line in fs:
            if line[0] == '>':
                if sequence != "":
                    name2sequence[name] = sequence
                    sequence = ""
                name = pattern.match(line).group(1)
            else:
                sequence += line.rstrip()
        if sequence != "":
            name2sequence[name] = sequence
    return name2sequence


def filter_library(params):
    bname2sequence = read_protein_names(params["input"]["fasta"]["base_fasta"], re.compile(parse_expression))
    peptides = set()
    with open(params["input"]["library"]["peptides"], "r") as peptides_in, \
            open(params["output"]["library"]["peptides"], "w") as peptides_out:
        header = peptides_in.readline()
        peptides_out.write(header)
        header_spl = header.split("\t")
        sequence_index = header_spl.index("Sequence")
        unique_index = header_spl.index("Unique (Proteins)")
        proteins_index = header_spl.index("Leading razor protein")
        start_position_index = header_spl.index("Start position")
        for line in peptides_in:
            spl = line.rstrip().split('\t')
            ps = []
            for p in spl[proteins_index].split(';'):
                if p in bname2sequence:
                    ps.append(p)
            if len(ps) >= 1:
                spl[proteins_index] = ";".join(ps)
                if len(ps) == 1:
                    spl[unique_index] = "yes"
                else:
                    spl[unique_index] = "no"
                spl[start_position_index] = str(bname2sequence[ps[0]].index(spl[sequence_index]) + 1)
                peptides_out.write("\t".join(spl) + "\n")
                peptides.add(spl[sequence_index])
    with open(params["input"]["library"]["evidence"], "r") as evidence_in, \
            open(params["input"]["library"]["msms"], "r") as msms_in, \
            open(params["output"]["library"]["evidence"], "w") as evidence_out, \
            open(params["output"]["library"]["msms"], "w") as msms_out:
        header = evidence_in.readline()
        evidence_out.write(header)
        msms_out.write(msms_in.readline())
        header_spl = header.split("\t")
        sequence_index = header_spl.index("Sequence")
        proteins_index = header_spl.index("Proteins")
        for evidence_line in evidence_in:
            msms_line = msms_in.readline()
            evidence_spl = evidence_line.rstrip().split('\t')
            ps = []
            for p in evidence_spl[proteins_index].split(';'):
                if p in bname2sequence:
                    ps.append(p)
            if len(ps) >= 1 and evidence_spl[sequence_index] in peptides:
                evidence_spl[proteins_index] = ";".join(ps)
                evidence_out.write("\t".join(evidence_spl) + "\n")
                msms_out.write(msms_line)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        parameters_file = "parameters.json"
    else:
        parameters_file = sys.argv[1]
    with open(parameters_file, 'r') as parameters_fs:
        filter_library(json.loads(parameters_fs.read()))