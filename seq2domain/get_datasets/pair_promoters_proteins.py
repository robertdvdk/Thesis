"""
Module-level docstring
"""
# Import statements
import os
import re
import sys
import parse_files

# Global variables
SPECIES_ID_TO_NAME = \
    {'amel5': 'APIME', 'araTha1': 'ARATH', 'canFam3': 'CANLF',
     'ce6': 'CAEEL', 'danRer7': 'DANRE', 'dm6': 'DROME',
     'galGal5': 'CHICK', 'hg38': 'HUMAN', 'mm10': 'MOUSE',
     'pfa2': 'PLAF7', 'rheMac8': 'MACMU', 'rn6': 'RAT',
     'sacCer3': 'YEAST', 'spo2': 'SCHPO', 'zm3': 'MAIZE'}

# Function definitions
def parse_epd(filename):
    """Parses EPDnew fasta files and adds them to a dictionary

     Args:
        filename::str
            Name of the EPDnew fasta file

    Returns:
        seq_dict::dict
            A dictionary of {gene id: [promoter sequence]}
    """
    epd_dict = {}
    with open(filename, 'r') as fopen:
        for line in fopen:
            if line.startswith('>'):
                full_id = line.split()[1]
                # Example: convert LOC408911_1 to LOC408911 (the gene ID)
                if full_id.endswith('_'):
                    curr_id = line.split()[1][:-1]
                else:
                    curr_id = line.split()[1][:-2]
                epd_dict[curr_id] = [""]
            else:
                epd_dict[curr_id][0] += line.strip()
    return epd_dict

def get_epd_ids(raw_dir):
    """Generates a dictionary containing all gene IDs in the EPD files

    Args:
        raw_dir::str
            The directory containing the epd fasta files

    Returns:
        epd_ids_dict::dict
            A dictionary of {species_name: {gene ID: [promoter sequence]}}
    """
    epd_dir = f"{raw_dir}/epd"
    if not os.path.exists(epd_dir):
        print(f"Please download all EPDnew files and put them inside"
              f"{raw_dir}/epd/(kingdom), where kingdom is the kingdom as "
              f"listed on EPDnew.")
    epd_dict = {}
    for kingdom in os.listdir(epd_dir):
        for species_filename in os.listdir(f"{epd_dir}/{kingdom}"):
            # Get species id; discard file extension
            species_id = species_filename.split(".")[0]
            species_name = SPECIES_ID_TO_NAME[species_id]
            epd_filename = f"{epd_dir}/{kingdom}/{species_filename}"
            epd_dict[species_name] = parse_epd(epd_filename)
    return epd_dict

def pair_genbank_ids(processed_dir, epd_dict):
    """Gets the GenBank (RefSeq) IDs corresponding to each promoter sequence

    Args:
        processed_dir::str
            The directory where processed files are stored
        epd_dict::dict
            A dictionary of {species_name: {gene ID: [promoter sequence]}}

    Returns:
        epd_dict::dict
            A dictionary of {species_name:
                {gene ID: [UniProt ID, GenBank ID, promoter sequence]}}
    """
    idmapping_file = f"{processed_dir}/id_map/relevant_idmappings.txt"
    with open(idmapping_file, 'r') as fopen:
        curr_genbank_id, curr_epd_id = '', ''
        for line in fopen:
            prot_id, mapping_type, mapped_id = line.strip().split('\t')
            if mapping_type == 'STRING' and curr_species == 'MAIZE':
                # Example: from 4577.GRMZM2G127789_P01 get GRMZM2G127789
                m = re.search(r'\..*_', line)
                mapped_id = m.group(0)[1:-1]
            elif mapping_type == 'EMBL-CDS' and curr_species == 'PLAF7':
                # EMBL-CDS contains version numbers after the dot
                mapped_id = mapped_id.split('.')[0]
            elif mapping_type == 'Ensembl' and curr_species == 'MACMU':
                # Ensembl contains version numbers after the dot
                mapped_id = mapped_id.split('.')[0]

            if mapping_type == 'UniProtKB-ID':
                # Example line: P32234  UniProtKB-ID    128UP_DROME
                # where P32234 is uniprot_id, and DROME is curr_species
                curr_species = mapped_id.split('_')[1]
                curr_genbank_id = ''
                curr_epd_id = ''
                uniprot_id = prot_id
            elif mapping_type == 'RefSeq' and not curr_genbank_id:
                # Store the first encountered RefSeq ID
                curr_genbank_id = mapped_id
            elif mapped_id in epd_dict[curr_species] and not curr_epd_id:
                # Store the first ID that's found in the EPD dict
                curr_epd_id = mapped_id
            if curr_genbank_id and curr_epd_id and len(epd_dict[curr_species][curr_epd_id]) == 1:
                epd_dict[curr_species][curr_epd_id].insert(0, curr_genbank_id)
                epd_dict[curr_species][curr_epd_id].insert(0, uniprot_id)

    for species, sequences in epd_dict.items():
        # Remove entries without Uniprot ID or GenBank ID
        keys_to_delete = []
        for gene_id, rest in sequences.items():
            if len(rest) != 3:
                keys_to_delete.append(gene_id)
        for key in keys_to_delete:
            del sequences[key]
    return epd_dict

def pair_protein_sequences(epd_dict, raw_dir, min_length=0, max_length=1000000):
    """Adds the protein sequences to the epd dict

    Args:
        epd_dict::dict
            A dictionary of {species_name:
                {gene ID: [UniProt ID, GenBank ID, promoter sequence]}}
        raw_dir::str
            Directory containing the raw files
        min_length::int
            Minimum length of a protein sequences for it to be kept
        max_length::int
            Maximum length of a protein sequence for it to be kept

    Returns:
        epd_dict::dict
            A dictionary of {species_name:
                {gene ID: [UniProt ID, GenBank ID, promoter sequence, protein sequence]}}
    """
    epd_dir = f"{raw_dir}/epd"
    genbank_dir = f"{raw_dir}/genbank/eukaryotes"
    for kingdom in os.listdir(epd_dir):
        for species_filename in os.listdir(epd_dir + "/" + kingdom):
            species_id = species_filename.split(".")[0]
            species = SPECIES_ID_TO_NAME[species_id]
            genbank_file = f"{genbank_dir}/{kingdom}/{species_id}.gpff"
            genbank_dict = parse_files.parse_genbank(genbank_file)
            for gene_id, (uniprot_id, genbank_id, prom_seq) in epd_dict[species].items():
                if genbank_id in genbank_dict:
                    prot_seq = genbank_dict[genbank_id]
                    if len(prot_seq) < min_length or len(prot_seq) > max_length:
                        continue
                    epd_dict[species][gene_id].append(genbank_dict[genbank_id])
    return epd_dict


def main():
    if len(sys.argv) != 3:
        print('Incorrect usage. Usage: python3 pair_promoters_proteins.py {raw directory} {processed directory}')
    raw_dir = sys.argv[1]
    processed_dir = sys.argv[2]

    # Generate dictionary containing all relevant information
    # prom ID, uniprot ID, prom seq, prot seq, genbank ID
    epd_dict = get_epd_ids(raw_dir)
    epd_dict = pair_genbank_ids(processed_dir, epd_dict)
    epd_dict = pair_protein_sequences(epd_dict, f'{raw_dir}/epd', min_length=100, max_length=1000)

    # Write output files
    for species, promoters in epd_dict.items():
        with open(f'/home/klis004/nbk_lustre/processed_data/sequences/{species}.txt', 'w') as fopen:
            for gene_id, rest in promoters.items():
                if len(rest) != 4:
                    continue
                uniprot_id = rest[0]
                genbank_id = rest[1]
                prom_seq = rest[2]
                prot_seq = rest[3]
                entry = f'GENEID: {gene_id}_{species}\nUPID: {uniprot_id}\nGBID: {genbank_id}\nPROMSEQ: {prom_seq}\nPROTSEQ: {prot_seq}\n'
                fopen.write(entry)

if __name__ == "__main__":
    """The main function of this module"""
    main()
