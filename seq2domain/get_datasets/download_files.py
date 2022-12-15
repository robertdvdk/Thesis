"""
Downloads the required files: GenBank, id_mapping, and interpro domains
Does not download required EPD files, as their fasta files are not
accessible by FTP.

Author: Robert van der Klis

Usage: python3 download_files.py {output directory}
"""

# Import statements
import os
import sys

# Function definitions
def download_file(download_dir, url, outfile):
    """Downloads and unzips the specified file in the specified directory

    Args:
        download_dir::str
            The directory in which to store the downloaded data
        url::str
            The url of the file to download
        outfile::str
            The name of the file after unzipping

    Returns:
        None
    """
    if not os.path.exists(download_dir):
        os.system(f"mkdir {download_dir}")
    if not os.path.exists(f"{download_dir}/{outfile}"):
        if url.endswith('.gz'):
            os.system(f"wget -O {download_dir}/{outfile}.gz {url}")
            os.system(f"gunzip {download_dir}/{outfile}.gz")
        else:
            os.system(f"wget -O {download_dir}/{outfile} {url}")

def main():
    """Calls the download_file function for all required files"""
    if len(sys.argv) != 2:
        print('Incorrect usage. Usage: python3 download_files.py {output directory}')
        sys.exit()
    main_dir = sys.argv[1]
    genbank_datasets = {
        'animals':
            {
                'amel5': 'https://ftp.ncbi.nih.gov/genomes/refseq/invertebrate/Apis_mellifera/all_assembly_versions/suppressed/GCF_000002195.4_Amel_4.5/GCF_000002195.4_Amel_4.5_protein.gpff.gz',
                'canFam3': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Canis_lupus_familiaris/all_assembly_versions/GCF_000002285.3_CanFam3.1/GCF_000002285.3_CanFam3.1_protein.gpff.gz',
                'ce6': 'https://ftp.ncbi.nih.gov/genomes/refseq/invertebrate/Caenorhabditis_elegans/all_assembly_versions/GCF_000002985.6_WBcel235/GCF_000002985.6_WBcel235_protein.gpff.gz',
                'danRer7': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_other/Danio_rerio/all_assembly_versions/GCF_000002035.4_Zv9/GCF_000002035.4_Zv9_protein.gpff.gz',
                'dm6': 'https://ftp.ncbi.nih.gov/genomes/refseq/invertebrate/Drosophila_melanogaster/all_assembly_versions/GCF_000001215.4_Release_6_plus_ISO1_MT/GCF_000001215.4_Release_6_plus_ISO1_MT_protein.gpff.gz',
                'galGal5': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_other/Gallus_gallus/all_assembly_versions/suppressed/GCF_000002315.4_Gallus_gallus-5.0/GCF_000002315.4_Gallus_gallus-5.0_protein.gpff.gz',
                'hg38': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_protein.gpff.gz',
                'mm10': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Mus_musculus/all_assembly_versions/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_protein.gpff.gz',
                'rheMac8': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Macaca_mulatta/all_assembly_versions/suppressed/GCF_000772875.2_Mmul_8.0.1/GCF_000772875.2_Mmul_8.0.1_protein.gpff.gz',
                'rn6': 'https://ftp.ncbi.nih.gov/genomes/refseq/vertebrate_mammalian/Rattus_norvegicus/all_assembly_versions/suppressed/GCF_000001895.5_Rnor_6.0/GCF_000001895.5_Rnor_6.0_protein.gpff.gz'
            },
        'fungi':
            {
                'sacCer3': 'https://ftp.ncbi.nih.gov/genomes/refseq/fungi/Saccharomyces_cerevisiae/all_assembly_versions/GCF_000146045.2_R64/GCF_000146045.2_R64_protein.gpff.gz',
                'spo2': 'https://ftp.ncbi.nih.gov/genomes/refseq/fungi/Schizosaccharomyces_pombe/all_assembly_versions/GCF_000002945.1_ASM294v2/GCF_000002945.1_ASM294v2_protein.gpff.gz'
            },
        'invertebrates':
            {
                'pfa2': 'https://ftp.ncbi.nih.gov/genomes/refseq/protozoa/Plasmodium_falciparum/all_assembly_versions/suppressed/GCF_000002765.4_ASM276v2/GCF_000002765.4_ASM276v2_protein.gpff.gz'
            },
        'plants':
            {
                'araTha1': 'https://ftp.ncbi.nih.gov/genomes/refseq/plant/Arabidopsis_thaliana/all_assembly_versions/GCF_000001735.3_TAIR10/GCF_000001735.3_TAIR10_protein.gpff.gz',
                'zm3': 'https://ftp.ncbi.nih.gov/genomes/refseq/plant/Zea_mays/all_assembly_versions/suppressed/GCF_000005005.1_B73_RefGen_v3/GCF_000005005.1_B73_RefGen_v3_protein.gpff.gz'
            }
    }
    idmapping_file = 'https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/idmapping/idmapping.dat.gz'
    interpro_file = 'https://ftp.ebi.ac.uk/pub/databases/interpro/current/protein2ipr.dat.gz'

    for kingdom, species in genbank_datasets.items():
        for species_id, url in species.items():
            download_file(f"{main_dir}/genbank/eukaryotes/{kingdom}/", url,
                          f"{species_id}.gpff")
    download_file(f"{main_dir}/interpro", interpro_file, "protein2ipr.dat")
    download_file(f"{main_dir}/id_map", idmapping_file, "idmapping.dat")

if __name__ == "__main__":
    """The main function of this module"""
    main()