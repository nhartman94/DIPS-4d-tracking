'''
root_to_ml.py


Goal: Preprocessing code for going from a generic Ntuple to ML-ready
      inputs for track-based NN taggers


Starting point: This code starts from the RNNNIP / DIPS preprocessing
       code going from root files -> np arrays: https://gitlab.cern.ch/hartman/RNNIP
       But with some refactoring for vectorization in the input preprocessing


TO DO: (with the latest generic ACTS Ntuple)
       -> See the notes in Ntuple-Exploration nb and the README



Nicole Hartman
Summer 2023
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py
from tqdm import tqdm
import json 

import uproot
import awkward as ak
import xarray as xr

# The branches we'll load in for the jet branches
jet_vars  = ['pt','eta','phi','isHS','label']
jalias = {v:f'jet_{v}' for v in jet_vars}

# The branches (and aliases) for the trk branches
trk_vars  = ['d0','z0','theta','phi','qOverP']
trk_vars += [f'var_{v}' for v in trk_vars]

trk_vtrk_vars= ['t','z']  
trk_vars += [f't{n}' for n in [30,60,90,120,180]]

talias = {v: f'track_{v}' for v in trk_vars}

trk_vars += ['numPix','numSCT','numPix1L','numPix2L']
for v in trk_vars[-4:]:
    talias[v] = f'tracks_{v}'

# derived vars 
talias['sd0'] = 'd0/var_d0'
talias['abs_sd0'] = 'abs(d0)'
talias['sz0'] = 'z0/var_z0'
talias['pt']  = '0.001 * abs(1/qOverP) * sin(theta)'
talias['eta']  = '- log(tan(theta/2))'


def processBatch(t,start,stop,iVars,deriv_vars,maxNumTrks=40,sort_var="abs_sd0"):
    '''
    Goal: Given the tree and a start and a stop, process this
          bunch of the tracks.
    '''

    # Load in the jet array
    jdf = t.arrays(jet_vars,library='pd',aliases=jalias,
                   entry_start=start,entry_stop=stop)

    jmask = (jdf['pt'] > 20) & (np.abs(jdf['eta']) < 4) & jdf['isHS'].astype('bool')

    # load in the track level array
    tak = t.arrays(trk_vars+['sd0','sz0','pt','eta','abs_sd0'],aliases=talias,
                   entry_start=start,entry_stop=stop)

    nEvts = len(tak)
    print(f'In processBatch, start={start}, stop={stop}, len(tak)={nEvts}')

    jet_trk_idx = t.arrays('jet_tracks_idx',entry_start=start,entry_stop=stop)['jet_tracks_idx']

    """
    This has the structure I want: 
    nJets, nTrks, nFeatures

    Also... make sure that the jet mask is getting applied to
    - the tracks we select (in the tarr computation) 
    - and the same mask gets applied to jdf
    """
    jmask = (jdf['pt'] > 20) & (np.abs(jdf['eta']) < 4) & jdf['isHS'].astype('bool')
    njets_all = np.array([len(jdf.loc[i,'pt']) for i in range(start,start+nEvts)])
    jmask_hier = ak.unflatten(jmask,counts=njets_all)

    tarr = ak.Array([tak_i[jlinks] for tak_i,evt_lev_links,jmask_evt in zip(tak,jet_trk_idx,jmask_hier) 
                 for jlinks, jmask_i in zip(evt_lev_links,jmask_evt) if jmask_i])

    jdf = jdf[jmask]

    # 1. Mask
    '''
    TO DO: Add the z0 * sin(theta) < 5 mm cut
    (and pixel quality cuts? or are those already here?)
    '''
    tmask = (tarr['pt'] > 0.5) & (abs(tarr['d0']) < 3.5)
    
    # Sort
    idx = ak.argsort(tarr[sort_var][tmask],ascending=False)
    

    # Calculate the derived variables: dr and ptfrac
    deta = tarr['eta']-jdf['eta']
    dphi = np.arcsin(np.sin(tarr['phi']-jdf['phi']))
    
    dr = np.sqrt(deta**2 + dphi**2)

    ptfrac = tarr['pt']/jdf['pt']

    
    tnp = np.zeros((len(jdf),maxNumTrks,len(iVars)+len(deriv_vars)))  

    assert len(iVars) + 2 == tnp.shape[-1] # Sanity check the dimensions

    for i,v in enumerate(iVars):
        padded = ak.fill_none(ak.pad_none(tarr[tmask][idx][v], maxNumTrks, clip=True), 0)
        tnp[:,:,i] = np.asarray(padded)
        
    # And now also for the newly derived variables
    for k,varr in zip(['ptfrac','dr'],[ptfrac,dr]):
        
        i+=1
        
        padded = ak.fill_none(ak.pad_none(varr[tmask][idx], maxNumTrks, clip=True), 0)
        tnp[:,:,i] = np.asarray(padded)

    return tnp


def pTReweight(jdf):
    '''
    Do the pT reweighting for the sample, and add a new col 
    `sample_weight` to the dataframe

    Inputs:
    - jdf: pandas dataframe w/ the jet level variables
    '''
    # Set the bin edged
    start, step = 0, 10 # GeV
    stop = np.max(jdf.pt)
    pT_edges = np.arange(start, stop+step, step)

    # Get the bin that each entry corresponds to
    x_ind = np.digitize(jdf.pt, pT_edges) - 1

    # Make the histograms
    l_hist, _ = np.histogram(jdf.loc[jdf.label==0,'pt'].values, bins=pT_edges)
    c_hist, _ = np.histogram(jdf.loc[jdf.label==4,'pt'].values, bins=pT_edges)
    b_hist, _ = np.histogram(jdf.loc[jdf.label==5,'pt'].values, bins=pT_edges)

    # Normalize and add epsilon so that you never get a divide by 0 error
    epsilon = 1e-8
    l_hist = l_hist / np.sum(l_hist) + epsilon
    c_hist = c_hist / np.sum(c_hist) + epsilon
    b_hist = b_hist / np.sum(b_hist) + epsilon

    # Reweight the b-jets and c-jets to the l-jet pT (and maybe later eta) dist 
    jdf['sample_weight'] = np.ones(len(jdf))
    
    for pdg,ref_hist in zip([4,5],[c_hist,b_hist]):
        mi = (jdf.label==pdg)
        jdf.loc[mi,'sample_weight'] = l_hist[x_ind[mi]] / ref_hist[x_ind[mi]]

def scale(data, var_names, savevars, filename='data/trk_scales.json', mask_value=0,
          jetData=None):
    '''
    
    Scaling function for the track features
    
    Args:
    -----
        data: a numpy array of shape (nb_events, nb_particles, n_variables)
        var_names: list of keys to be used for the model
        savevars: bool -- True for training, False for testinh
                  it decides whether we want to fit on data to find mean and std
                  or if we want to use those stored in the json file
        filename: string: Where to save the output
        mask_value: the value to mask when taking the avg and stdev

    Returns:
    --------
        modifies data in place, writes out scaling dictionary

    Reference: Taken from Micky's dataprocessing.py file in
    https://github.com/mickypaganini/RNNIP
    '''
    
    scale = {}

    # data has shape (nJets,maxNumTrks,nFeatures), so to sort out the mask,
    # we need to find where the value is masked for a track over
    # all of it's features.
    mask = ~ np.all(data == mask_value, axis=-1) # shape (nJets,nTrks)

    if savevars:

        # track variables
        for v, name in enumerate(var_names):
            print('Scaling feature {} of {} ({}).'.format(v + 1, len(var_names), name))
            f = data[:, :, v]
            slc = f[mask]
            m, s = slc.mean(), slc.std()
            slc -= m
            slc /= s
            data[:, :, v][mask] = slc.astype('float32')
            scale[name] = {'mean' : float(m), 'sd' : float(s)}
           
        # Jet variables
        if jetData is not None:
            scaler = StandardScaler(copy=False) # scale the data in place
            scaler.fit_transform(jetData)
            scale['jet_mean'] = scaler.mean_
            scale['jet_scale'] = scaler.scale_
            scale['jet_var'] = scaler.var_
            scale['n_samples_seen'] = scaler.n_samples_seen_
            
        with open(filename, 'w') as varfile:
            json.dump(scale, varfile)

    else:
        with open(filename, 'r') as varfile:
            varinfo = json.load(varfile)

        # track variables
        for v, name in enumerate(var_names):
            print('Scaling feature {} of {} ({}).'.format(v + 1, len(var_names), name))
            f = data[:, :, v]
            slc = f[mask]
            m = varinfo[name]['mean']
            s = varinfo[name]['sd']
            slc -= m
            slc /= s
            data[:, :, v][mask] = slc.astype('float32')
                        
        if jetData is not None:
            scaler = StandardScaler(copy=False) # scale the data in place
            varinfo['jet_mean'] = scaler.mean_
            varinfo['jet_scale'] = scaler.scale_
            varinfo['jet_var'] = scaler.var_
            varinfo['n_samples_seen'] = scaler.n_samples_seen_
            scaler.transform(jetData)

def prepareForKeras(jet,trk_xr,outputFile):
    '''
    Prepare the ML inputs for keras and save the file
    
    MISSING from this Ntuple: nNextToInnHits,nInnHits,nsharedBLHits, nsplitBLHits,
                              nsharedPixHits,nsplitPixHits,nsharedSCTHits
    TO DO (later) -- add to noNormVars
    
    ALSO TO DO: d0, z0, sd0 and sz0 need to be w/r.t. PV (instead of beam spot)
    '''
    
    # Step 0: Process the string inputs for the vars in each norm sheme -> list
    noNormVars = ['sd0','sz0']
    logNormVars = ['ptfrac','dr']
    jointNormVars = ['numPix', 'numSCT','d0','z0']
    
    # Jet variables to pass to the network
    # Default an empty list, but pass pt and eta for GN2
    jetVars = []
    
    # Step 1: Select the relevant variables
    inpts = noNormVars + logNormVars + jointNormVars
    
    # Check that all of the requested inputs are actually vars in trk_xr
    trkInputs = list(trk_xr.coords['var'].values)
    for inpt in inpts:
        if inpt not in trkInputs:
            raise ValueError('In prepareForKeras(): requested var {} not in trk_xr'.format(inpt))
    
    X = trk_xr.loc[:,:,inpts].values
    X_jet = jdf[jetVars]
    ix = trk_xr.indexes['jet']
    print("X.shape = ", X.shape)
    
    # Keep track of which tracks in the jet are masked
    mask = ~ np.all(X == 0, axis=-1)
    print("mask",mask.shape)
    
    # Take the log of the desired variables
    for i, v in enumerate(logNormVars):
        j = i + len(noNormVars)
        X[:,:,j][mask] = np.log(np.where(X[:,:,j][mask]==0,1e-8,X[:,:,j][mask]))
    
    # Step 2: Train / test split
    pdg_to_class = {0:0, 4:1, 5:2, 15:3}
    y = jdf.label.replace(pdg_to_class).values
    
    random_seed = 25
    X_train, X_test, y_train, y_test, ix_train, ix_test, w_train, w_test, = \
        train_test_split(X, y, ix, jdf.sample_weight, test_size=0.333,
                         random_state=random_seed)
    
    # Step 3: Normalize the requested inputs
    
    # Get a string representing the variables getting scaled
    varTag = "_".join(noNormVars) if len(noNormVars) != 0 else ''
    varTag += '_logNorm_' + "_".join(logNormVars) if len(logNormVars) != 0 else ''
    varTag += '_norm_' + "_".join(jointNormVars) if len(jointNormVars) != 0 else ''
    
    # Scale the vars and save the files
    scalingfile = f"data/scale_{varTag}.json"
    print("scalingfile",scalingfile)
    
    if len(logNormVars)+len(jointNormVars) > 0:
    
        scale(X_train[:,:,len(noNormVars):], logNormVars+jointNormVars, savevars=True,  filename=scalingfile)
        scale(X_test[:,:, len(noNormVars):], logNormVars+jointNormVars, savevars=False, filename=scalingfile)
    
    myDict = {
        "X_train" : X_train,
        "y_train" : y_train,
        "ix_train" : ix_train,
        "weights_train" : w_train,
    
        "X_test" : X_test,
        "y_test" : y_test,
        "ix_test" : ix_test,
        "weights_test" : w_test,
    }
    
    # Step 4: Save as h5py files
    print("Saving datasets in {}".format(outputFile))
    f = h5py.File(outputFile, 'w')
    
    for key, val in myDict.items():
        f.create_dataset(key, data=val)
    
    f.close() 


if __name__ == '__main__':

    from argparse import ArgumentParser

    p = ArgumentParser()

    p.add_argument('--filename', type=str,
                   default="data/hadded.root",
                   dest="filename",
                   help="path to and name of the root file")

    p.add_argument('--tName', type=str,
                   default="EventTree",
                   dest="tName",
                   help="path to and name of the root file")

    p.add_argument('--output', type=str,
                   default="data/output.hdf5",
                   dest="output",
                   help="Name of the output .h5 file")

    args = p.parse_args()

    fName = args.filename
    tName = args.tName
    outputFile = args.output


    # Step 0: Open the file 
    f = uproot.open(fName)

    # Step 1: Read in the number of events
    t = f[tName]
    nEntries = t.num_entries


    # Step 2a: Load in the jets
    jdf = t.arrays(jet_vars,library='pd',aliases=jalias)
    jmask = (jdf['pt'] > 20) & (np.abs(jdf['eta']) < 4) & jdf['isHS'].astype('bool')
    jdf = jdf[jmask] 

    # Step 2b: Read in just a part of the tree and batch the track preprocessing over these chunks
    batch_size = 1500
    chunks = np.arange(0,nEntries+batch_size, batch_size)

    # Subset of the track vars (needed for training dips and / or GN1/2)
    tVars = ['d0','z0','var_d0', 'var_z0','qOverP','theta','phi','numPix','numSCT']

    iVars = tVars + ['sd0','sz0']
    deriv_vars= ['dr','ptfrac']

    maxNumTrks=40

    trk_xr = xr.DataArray(0,
                      coords=[('jet',np.arange(len(jdf))),
                              ('trk',np.arange(maxNumTrks)),
                              ('var',iVars+deriv_vars)])


    i=0
    for start, stop in tqdm(zip(chunks[:-1],chunks[1:])):

        t_np_i = processBatch(t,start,stop,maxNumTrks=maxNumTrks,iVars=iVars,deriv_vars=deriv_vars)   

        trk_xr[i:i+t_np_i.shape[0]] = t_np_i         
        i += t_np_i.shape[0]

    # Step 3: ML pre-processing
    prepareForKeras(jdf,trk_xr,outputFile)

