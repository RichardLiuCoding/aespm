# Tutorials and example notebooks

## aespm 101 -- Get started with aespm

This tutorial notebook talks about basics of aespm. It contains information about how to make a workflow notebook run on your instrument and how to write your own workflows from scratch.

## Example -- Combi_library_grid_explorer

Following topics are covered:

* Exploration of a large combinatorial library sample in the AC mode
* Exploration of a large combinatorial library sample in the PFM-DART mode
* Switching and initializing different scanning modes

## Example -- Feature based discovery

Following topics are covered:

* Connect to a remote supercomputer
* Use Canny filter or Variational Autoencoder (VAE) to extract characteristic features from an image
* Extract the coordinates for the features of interest
* Move the probe to only these coordinates to take spectral measurements

## Example -- Spectra based discovery

Following topics are covered:

* Connect to a remote supercomputer
* Set up Deekp Kernel Learning (DKL) workflow
* Segment image into small patches and take initial seeding measurements from seeding points
* Train the DKL model through the active learning process
* Visualize the training process and make predictions based on the DKL model


# Default actions implemented in ```spm_control()``` function

```Python
# You can use any element in the same line to achieve the same control:
[
 # Map-related controls
 ['ACTune', 'AC Tune', 'Tune'], # Start an AC tuning of probe
 ['AC_Mode', 'Mode=AC', 'AC'],  # Switch scanning mode to AC modes
 ['ChangeName', 'FileName', 'BaseName'],     # Change the base save name
 ['DownScan', 'ScanDown', 'Start', 'start'], # Start a downward scan
 ['DriveAmp', 'DriveAmplitude', 'DriveVoltage', 'DriveVolt', 'v_ac', 'V_ac'], # Change the drive voltage v_ac
 ['DriveFreq', 'DriveFrequency', 'Freq', 'Frequency'], # Change the drive frequency
 ['FBHeight', 'FeedbackHeight', 'PID2=Height', 'FB_Height', 'FB_height'],     # 
 ['GetTune', 'TuneData'], # Read the tune data
 ['IGain', 'IGains', 'IntegralGain', 'igain'], # Change the I Gain for the scan
 ['PFM_Mode', 'Mode=PFM', 'PFM'],              # Switch to PFM mode
 ['Points and Lines', 'Points', 'Pixels', 'points', 'pixels'], # Change the number of pixels
 ['ScanAngle', 'Scan Angle', 'Angle'], # Change the scan angle
 ['ScanRate', 'ScanRates', 'speed'],   # Change the scan rate
 ['ScanSize', 'Scan Size', 'scan size', 'scan_size'],  # Change the scan size
 ['Setpoint', 'SetPoint'], # Change the setpoint
 ['SetpointAmp', 'AC Setpoint', 'SetpointAC'], # Change the setpoint of amplitde
 ['SetpointDefl', 'PFMSetpoint', 'SetpointPFM', 'SetpointContact'], # Change the setpoint of deflection
 ['StartFB0', 'StartLoop0', 'StartLoopX'], # Start PID loop for x piezo
 ['StartFB1', 'StartLoop1', 'StartLoopY'], # Start PID loop for y piezo
 ['StartFB2', 'Engage', 'StartLoop2', 'StartLoopZ'], # Start PID loop for z piezo
 ['StopScan', 'Stop', 'stop'], # Stop the scan
 ['UpScan', 'ScanUp'],         # Start an upward scan
 ['XOffset', 'X Offset', 'Offset_X'], # Move the scan center along the x-axis
 ['YOffset', 'Y Offset', 'Offset_Y'], # Move the scan center along the y-axis
 ['ZeroPD', 'PDZero'], # Zero the deflection

 # Force curve related controls
 ['ClearForce', 'Clear', 'ClearMarker'], # Clear all existing force point
 ['GoThere', 'Gothere'], # Go to the current force point (if no existing force point, it will move the probe to the center of image)
 ['SingleF', 'SingleForce'], # Start a single force-distance curve
 ['ThatsIt', 'Thats'], #
 ['FDTrigger', 'FDTriggerPoint'], # Change the trigger point for the F-D

 # Nap panel related controls
 ['SurfaceVoltage', 'V_surface', 'v_surface', 'v_surf'], # Change the surface voltage
 ['TipVoltage', 'V_tip', 'v_tip'], # Change the tip voltage

 # Stage related controls
 ['Approach', 'StartApproach'], # Start approching to the surface
 ['DisableStage', 'StageDisable', 'DisableMotor', 'MotorDisable'],  # Disable the stage movement
 ['EnableStage', 'StageEnable', 'EnableMotor', 'MotorEnable'],      # Enable the stage movement

 # DART related controls
 ['CenterPhase', 'Center Phase', 'PhaseCenter'], # Center the DART phase
 ['Cycles', 'Cycle', 'NumCycles'], # Change the number of cycles for DART spec
 ['DARTFreq', 'DART Freq', 'CentralFreq'], # Change the DART center frequency
 ['DARTPhase1', 'DART Phase1'], # Change the DART phase 1
 ['DARTPhase2', 'DART Phase2'], # Change the DART phase 2
 ['DARTSweepWidth', 'SweepWidth'], # Change the DART sweep frequency range in tuning
 ['DARTWidth', 'DART Width', 'FreqWidth'], # Change the DART frequency separation 
 ['DualFreq', 'DARTMode', 'DARTDualFreq'], # Enable/Disable the dual frequency modes
 ['DARTTrigger', 'SSTrigger'], # Change the trigger point for DART
 ['DARTAmp', 'DART v_ac'], # Change the DART drive voltage 

 # I-V panel related controls
 ['IVAmpDART', 'IV_Amp_DART', 'iv_amp_DART'], # DART spec amplitude
 ['IVDoItDART', 'ivdoit_DART', 'IVDoit_DART'], # Take a single DART spec
 ['IVFreqDART', 'IV_Freq_DART', 'iv_freq_DART'], # Change the DART spec freq
 ['OneTuneDART', 'TuneDART', 'DARTTune', 'DART_Tune'], # Start one tune for DART mode
 ['IVAmp', 'IV_Amp', 'iv_amp'], # 
 ['IVDoIt', 'ivdoit', 'IVDoit'],
 ['IVFreq', 'IV_Freq', 'iv_freq'],

 # Video panel related controls
 ['Capture', 'TakePhoto'], # Take a snapshot in the video panel
] 
```