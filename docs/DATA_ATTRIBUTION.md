# Dataset sources and attribution

The included IIoTAPT and Darknet tables are processed derivatives used by this
project. Binary and limited-label tables have distinct label schemas and must
not be interchanged. Their original byte order is preserved for the supplied
row indices. No raw packet captures are included.

## IIoTAPT / CICAPT-IIoT

Source: [CIC IIoT Dataset 2024](https://www.unb.ca/cic/datasets/iiot-dataset-2024.html).

Please cite the dataset and the paper listed by its provider:

E. Ghiasvand, S. Ray, S. Iqbal, S. Dadkhah, and A. Ghorbani,
“Resilience Against APTs: A Provenance-Based IIoT Dataset for Cybersecurity Research.”
The provider also lists “CICAPT-IIOT: A provenance-based APT attack dataset for
IIoT environment,” preprint, July 2024, by the same authors.

The [CIC dataset FAQ](https://www.unb.ca/cic/datasets/index.html) states:
“With any of our datasets, you may redistribute, republish, and mirror our
datasets in any form. However, any use or redistribution of the data must
include a citation to the dataset and the research paper listed.”

## Darknet / CICDarknet2020

Source: [CIC-Darknet2020](https://www.unb.ca/cic/datasets/darknet2020.html).

Please cite CICDarknet2020 and:

Arash Habibi Lashkari, Gurdip Kaur, and Abir Rahali,
“DIDarknet: A Contemporary Approach to Detect and Characterize the Darknet
Traffic using Deep Image Learning,” 10th International Conference on
Communication and Network Security, Tokyo, Japan, November 2020.
[DOI: 10.1145/3442520.3442521](https://doi.org/10.1145/3442520.3442521).

The dataset page explicitly permits redistribution, republication, and mirroring
in any form with citation to the dataset and the listed paper.

## CloudDDoS / BCCC-cPacket-Cloud-DDoS-2024

Source: [BCCC-cPacket-Cloud-DDoS-2024](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/cloud-ddos-attacks-bccc-cpacket-cloud-ddos-2024/).

MohammadMoein Shafi, Arash Habibi Lashkari, Vicente Rodriguez, and Ron Nevo,
“Toward Generating a New Cloud-Based Distributed Denial of Service (DDoS)
Dataset and Cloud Intrusion Traffic Characterization,” Information 15(4), 195,
2024. [DOI: 10.3390/info15040195](https://doi.org/10.3390/info15040195).

The source page provides a dataset-request link. No explicit redistribution
permission was identified on the inspected BCCC dataset and request pages;
CloudDDoS traffic tables are therefore not included in this public package.
Its exact processed-file hashes and feature schema remain in `configs/`.

Provider statements above were checked on 2026-10-21. These dataset statements
are separate from the third-party software licenses.
