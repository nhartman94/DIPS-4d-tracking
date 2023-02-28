# DIPS 4d tracking

**Goal:** Preprocessing and training code for training DIPS on a generic Ntuple.

## Step 1: Setup

1. Clone this repo.
2. Get the .root file (location from Francesco): `https://cernbox.cern.ch/rootjs/public/qJykQrcBIjiCEzi/`
3. Sofware setup for preprocessing currently using swan notebooks with the LCG102b enviornment.

(When Ariel gets Camille and Madison accounts, I can revise this for the SLAC machines.)

## Step 2: Preprocessing

The notebook `Ntuple-Exploration.ipynb` is my work in progress for the updated preprocessing.

I'm still working on the aspect of defining the track's with respect to the PV (needed for the preprocessing cuts on the IPs).
I've added this to the to do list below.

**Plan:** After finalizing the functions, I'll put these funcions in a script to process more easily!!

## Step 3: Training

Inside of the `Deep-Sets` folder are the notebooks from the Berkeley workshop, after finishing the pre-processing notebook, I plan to edit these too to harmonize w/ the output of `Ntuple-Exploration.ipynb`.

## TO DO

**Before a first
- Check the IP calc w/r.t. the PV
- How do we get the other track parameters (and errors) w/r.t. the PV?

**Next steps
- Add the lifetime IP sign
- Implement a prompt muon / lepton veto
- Maybe to be consitent with the FTAG track selection with a $p_T$ dependent $\Delta R$ cut? 
- Use resampling for the FTAG training instead
- Brainstorm with Lorenzo to add the rest of the track quality inputs to the Ntuple
