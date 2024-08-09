"""
Script to run a full contant geometry sweep (n_contacts x contact_length)

The sweep of contact length and contact spacing is done by calling the 
monopolar_sweep_n_contacts.py script with new parameters.

The parameters were chosen to step down contact height while keeping
a common span of contacts along a 9-10 mm range, starting at the 
tip.  This is important becase
1. Post-hoc analysis will be done in a a common cylindrical space
   defined in the  CustomExport parametes in the input json.
2. Current tends to run along the metal contacts, so keeping the 
   active contact centered on a bunch of similar contacts is an
   import control.

These values were selected by hand by stepping up the number of levels:
levels/stim_level         5/3    7/4    9/5     11/6     13/7
target (L+gap)(~9.5/N)    1.9    1.35   1.05    0.86     0.73 
L                         1.5    1.1    0.9     0.7      0.55
spacing/gap               0.5    0.25   0.2     0.2      0.2
Total span (nL + (n-1)s)  9.5    9.2    9.7     9.7      9.55
"""
%run scripts/test_n_segments.py -l 1.50 -c 0.50 -v 5 3 -s 1 3 5 7 9 -M 0.02 -E 0.02 -V -d /Volumes/sim_data/results/  -D test_L150_C050_E00
%run scripts/test_n_segments.py -l 1.10 -c 0.25 -v 7 4 -s 1 3 5 7 9 -M 0.02 -E 0.02 -V -d /Volumes/sim_data/results/  -D test_L110_C025_E00
%run scripts/test_n_segments.py -l 0.90 -c 0.20 -v 9 5 -s 1 3 5 7 9 -M 0.02 -E 0.02 -V -d /Volumes/sim_data/results/  -D test_L090_C020_E00
%run scripts/test_n_segments.py -l 0.70 -c 0.20 -v 11 6 -s 1 3 5 7 9 -M 0.02 -E 0.02 -V -d /Volumes/sim_data/results/  -D test_L070_C020_E00
%run scripts/test_n_segments.py -l 0.55 -c 0.20 -v 13 7 -s 1 3 5 7 9 -M 0.02 -E 0.02 -V -d /Volumes/sim_data/results/  -D test_L055_C020_E00

%run scripts/test_n_segments.py -l 1.50 -c 0.50 -v 5 3 -s 1 3 5 7 9 -M 0.02 -E 0.02 -e 0.1 -V -d /Volumes/sim_data/results/  -D test_L150_C050_E01
%run scripts/test_n_segments.py -l 1.10 -c 0.25 -v 7 4 -s 1 3 5 7 9 -M 0.02 -E 0.02 -e 0.1 -V -d /Volumes/sim_data/results/  -D test_L110_C025_E01
%run scripts/test_n_segments.py -l 0.90 -c 0.20 -v 9 5 -s 1 3 5 7 9 -M 0.02 -E 0.02 -e 0.1 -V -d /Volumes/sim_data/results/  -D test_L090_C020_E01
%run scripts/test_n_segments.py -l 0.70 -c 0.20 -v 11 6 -s 1 3 5 7 9 -M 0.02 -E 0.02 -e 0.1 -V -d /Volumes/sim_data/results/  -D test_L070_C020_E01
%run scripts/test_n_segments.py -l 0.55 -c 0.20 -v 13 7 -s 1 3 5 7 9 -M 0.02 -E 0.02 -e 0.1 -V -d /Volumes/sim_data/results/  -D test_L055_C020_E01

