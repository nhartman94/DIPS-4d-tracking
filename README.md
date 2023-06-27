# DIPS 4d tracking

**Goal:** Preprocessing and training code for training DIPS on a generic Ntuple.

## Step 1: Setup

1. Clone this repo.
2. Get the .root file (location from Francesco): `https://cernbox.cern.ch/rootjs/public/qJykQrcBIjiCEzi/`
3. Sofware setup for preprocessing currently using swan notebooks with the LCG102b enviornment.

## Step 2: Preprocessing

The notebook `Ntuple-Exploration.ipynb` is my work in progress for the updated preprocessing.

The track's IP still need to be defined with respect to the PV.
I've added this to the to do list below and made notes in the notebook.

Also, to run this as a swan notebook can only handle 16 GB of data, so the notebook is currently only processing 1500 events, but I think putting these function in a script with batching the reading, jet, track and track sort will let us process the whole root file quickly :) 

But I wanted to share the notebook first, b/c I think this is a little more intuitive.

## Step 3: Training

`Deep-Sets-tutorial.ipynb` is a pedagogical version for DIPS from a recent [tutorial]() I gave at Berkeley, and the solutions for the exercises are in `Deep-Sets-soln.ipynb`.

## To keep in mind (with the new ACTS Ntuple)

- Get the Impact Parameters w/r.t. the PV.
- Add the lifetime IP sign
- Implement a prompt muon / lepton veto
- Maybe to be consitent with the FTAG track selection with a $p_T$ dependent $\Delta R$ cut? 
- Use resampling for the FTAG training instead
- Add the rest of the track quality inputs to the Ntuple


**Logo credit:** Ty to the one and only [Katharine Leney](https://twitter.com/PhysicsCakes)!
