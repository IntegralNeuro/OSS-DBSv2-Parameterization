A quick description how to build custom electrodes
1) Switch to the main branch of OSS-DBS v2
2) Copy scripts from /custom_electrode/main to /ossdbs/ossdbs/electrodes
3) In /custom_electrode/stim_folder/oss-dbs_parameters.json, find entry Electrode -> CustomParameters and modify the parameters
4) 
	a) in best_electrode_ever.py, find entries "customization parameters" and modify
	b) IMPORTANT: modify "_n_contacts" under the second entry of "customization parameters"; the equation is provided.
5) Modify /custom_electrode/stim_folder/oss-dbs_parameters.json 
	a) make sure settings for all contacts are described. Number of contacts should be equal to "_n_contacts"!
	b) modify other settings as you see appropriate. "ExportElectrode" might not work ATM. "CalcAxonActivation" should be run via Lead-DBS, native support will be added before 04/27/2024.
	c) "ExportElectrode" will actually export the electrode but break the stimulation. There is an issue with default meshing for export (not the same as for FEM),
6) Use ossdbs oss-dbs_parameters.json --loglevel 10 for simulation info (debug mode)
7) Checkout results in /custom_electrode/stim_folder/Results_rh/
	a) distribution of E-field metrics (V/mm) and mesh quality can be visualized in Paraview (.vtu files)
	b) E-field (V/m) and VTA are also stored in .nii file, same space as /custom_electrode/stim_folder/segmask.nii 
	c) Field components are stored in /custom_electrode/stim_folder/Results_rh/E_field_Lattice.csv.
	d) .nii and .csv outputs are defined by "Lattice" parameter in /custom_electrode/stim_folder/oss-dbs_parameters.json 
8) In /custom_electrode/stim_folder/brain_structures, you will find STN and internal capsule, which you can use to "score" E-field distribution
