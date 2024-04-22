# aespm -- "A"utomated "E"xperiment on "SPM"

Python interface that enables local and remote control of Scanning Probe Microscope (SPM) with codes.

It offers a modular way to write autonomous workflows ranging from simple routine operations, to advanced automatic scientific discovery based on machine learning.

# Installation

```Python
pip install aespm
```

# Examples

## Create an aespm.Experiment() object

```Python
import aespm as ae
folder = 'where_you_save_your_data'
exp = ae.Experiment(folder=folder)
```

There is a list of default actions in the end of this page.

## Execute a single command
To start a downwards scan:
```Python
exp.execute('DownScan', wait=1.5)
```

## Define a custom action based on a list of sequential operations
To start a scan with given save name and scan rate:

```Python
def ac_scan(self, scanrate, fname):
    action_list = [
        ['ChangeName', fname, None], # Change file names
        ['ScanRate', scanrate, None],# Change scan rate
        ['DownScan', None, 1.5],     # Start a down scan
        ['check_files', None, None], # Pause the workflow until number of files change in the save_folder 
    ]
    self.execute_sequence(operation=action_list)
    # Load the latest modified file in the data saving folder
    img = ae.ibw_read(get_files(folder=self.folder)[0])
    return img

exp.add_func(ac_scan)
```

Then the custom action can be called in three ways:
```Python
# Preferred way: call it the same way as default actions
img = exp.execute('ac_scan', value=[1, 'NewImage'])

# Put it in the operation sequence list
action_list = [['ac_scan', [1, 'NewImage'], None], ['Stop', None, None]]
img = exe.execute_sequence(operation=action_list)

# Call as a method directly
img = exp.ac_scan(1, 'NewImage')
```

This is how a functional block is created.

## Keep track of experiment parameters with exp.param dict

Let's store some useful parameters in the **exp.param**

```Python
# You can directly add it using exp.update_param()
exp.update_param(key='f_dart', value=353.125e3) # DART will tune the probe around this freq to make sure it can relibly track the resonance

# You can also update multiple parameters together:
p = {
    'v_dart': 1, # unit is V
    'f_width': 10e3, # Hz 
    'ScanSize': 5e-6, # um
    'ScanPixels': 256, # pixels
}
for key in p:
    exp.update_param(key=key, value=p[key])
```

This is particularly useful when you need to track many different data formats (list, np.ndarray, tensors) when integrating with machine learning algorithms.

For more detailed examples, please take a look at tutorials and example notebooks in **aespm/notebooks** folder.

## Combinatorial library exploration
[Link](https://drive.google.com/file/d/1kcdGX46scTYePuiLnQiLbcp94MKt3_AG/view?usp=sharing) to the video of combinatorial exploration on a grid.

## Deek Kernel Learning on Jupiter
[Link](https://drive.google.com/file/d/1fOdsmjxh1PEiKI6Drm49-SOtuo5KTxVx/view?usp=share_link) to the video of Deep Kernel active learning controlled from a supercomputer.

# Three levels of integration

## spm_contrl()

The key function in the interface layer is ```spm_control()```, which unifies the interaction with SPM controller.

```spm_control()``` is a wrapper of lower-level ```write_spm()``` and ```read_spm()```. It translates hyper-language actions that are familiar to users to instrument-specific codes/commands that controller can understand.

## Experiment object

This is the fundamental object for AE workflows.

* **exp.execute()** calls spm_control() directly
* **exp.execute_sequence()** runs a list of actions in sequence
* **exp.add_func()** is the key to build functional blocks
* **exp.param** is a dict to keep track of experimental parameters and ML intermediate data
* It handles file I/O, SSH file transfer and connection automatically


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
