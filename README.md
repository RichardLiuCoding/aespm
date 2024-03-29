# aespm
Python interface that enables local and remote control of AR SPM with codes

# spm_contrl()

The key function in this library is ```spm_control()```, which unifies the interaction with AR controller.

Here are a few examples:

## Simple actions

```Python
# Start a scan
# Wait is waiting time in the unit of second.
# It's required to wait until the operation is done.
spm_control('DownScan', wait=1.5)

# Change the scan rate to 1 Hz
spm_control('ScanRate', wait=0.5)
```

## Domain writing and spectroscopy
```Python
# global parameters
save_folder = # where you save your ibw files
xsize, ysize = 5e-6, 5e-6 # size of the scan
pos0 = [xsize/2, ysize/2] # center of the scan
pos_measure = [1e-6, 1e-6] # measurement location, in the unit of m

# Write a domain at + 10 V, wait for the 5 um scan to finish, and then start a hysteresis loop measurement at the (1 um, 1 um) location
spm_control('SurfaceVoltage', value=10)
spm_control('UpScan', wait=2/1+3)

# We wait until the file number in the save folder has changed before next action
check_file_number(path=save_folder)

# Tune the probe and start the spectroscopy
spm_control('ClearForce')
spm_control('GoThere', wait=1)

spm_control('SurfaceVoltage', value=0)
tune_probe(num=1, out=False, center=350e3, width=50e3)
time.sleep(1)

# Get the x y setpoint for the current location 
v0 = read_igor(key=['PIDSLoop.0.Setpoint', 'PIDSLoop.1.Setpoint'])
r = pos_measure - pos0
move_tip(r=r, v0=v0)
time.sleep(1)
spm_control('IVDoItDART')
check_file_number(path=save_folder)
```

# More complicated workflows

## Combinatorial library exploration
[Link](https://drive.google.com/file/d/1kcdGX46scTYePuiLnQiLbcp94MKt3_AG/view?usp=sharing) to the video of combinatorial exploration on a grid.

Here are the core codes for moving the stage this workflow:
```Python
# Enable the stage move --> 5 sec, 8 seconds for safety
spm_control('EnableStage', wait=8)
move_stage(distance=[step_x, 0])
time.sleep(2* 2)
spm_control('DisableStage', wait=1)
spm_control('StartApproach', wait=45)
```

## Deek Kernel Learning on Jupiter
[Link](https://drive.google.com/file/d/1fOdsmjxh1PEiKI6Drm49-SOtuo5KTxVx/view?usp=share_link) to the video of combinatorial exploration on a grid.

# All implemented controls in ```spm_control()``` function

```Python
# You can use any element in the same line to achieve the same control:
[
 # Map-related controls
 ['ACTune', 'AC Tune', 'Tune'],
 ['AC_Mode', 'Mode=AC', 'AC'],
 ['ChangeName', 'FileName', 'BaseName'],
 ['DownScan', 'ScanDown', 'Start', 'start'],
 ['DriveAmp', 'DriveAmplitude', 'DriveVoltage', 'DriveVolt', 'v_ac', 'V_ac'],
 ['DriveFreq', 'DriveFrequency', 'Freq', 'Frequency'],
 ['FBHeight', 'FeedbackHeight', 'PID2=Height', 'FB_Height', 'FB_height'],
 ['GetTune', 'TuneData'],
 ['IGain', 'IGains', 'IntegralGain', 'igain'],
 ['PFM_Mode', 'Mode=PFM', 'PFM'],
 ['Points and Lines', 'Points', 'Pixels', 'points', 'pixels'],
 ['ScanAngle', 'Scan Angle', 'Angle'],
 ['ScanRate', 'ScanRates', 'speed'],
 ['ScanSize', 'Scan Size', 'scan size', 'scan_size'],
 ['Setpoint', 'SetPoint'],
 ['SetpointAmp', 'AC Setpoint', 'SetpointAC'],
 ['SetpointDefl', 'PFMSetpoint', 'SetpointPFM', 'SetpointContact'],
 ['StartFB0', 'StartLoop0', 'StartLoopX'],
 ['StartFB1', 'StartLoop1', 'StartLoopY'],
 ['StartFB2', 'Engage', 'StartLoop2', 'StartLoopZ'],
 ['StopScan', 'Stop', 'stop'],
 ['UpScan', 'ScanUp'],
 ['XOffset', 'X Offset', 'Offset_X'],
 ['YOffset', 'Y Offset', 'Offset_Y'],

 # Force curve related controls
 ['ClearForce', 'Clear', 'ClearMarker'],
 ['GoThere', 'Gothere'],
 ['SingleF', 'SingleForce'],
 ['ThatsIt', 'Thats'],

 # Nap panel related controls
 ['SurfaceVoltage', 'V_surface', 'v_surface', 'v_surf'],
 ['TipVoltage', 'V_tip', 'v_tip'],

 # Stage related controls
 ['Approach', 'StartApproach'],
 ['DisableStage', 'StageDisable', 'DisableMotor', 'MotorDisable'],
 ['EnableStage', 'StageEnable', 'EnableMotor', 'MotorEnable'],

 # DART related controls
 ['CenterPhase', 'Center Phase', 'PhaseCenter'],
 ['Cycles', 'Cycle', 'NumCycles'],
 ['DARTFreq', 'DART Freq', 'CentralFreq'],
 ['DARTPhase1', 'DART Phase1'],
 ['DARTPhase2', 'DART Phase2'],
 ['DARTSweepWidth', 'SweepWidth'],
 ['DARTWidth', 'DART Width', 'FreqWidth'],
 ['DualFreq', 'DARTMode', 'DARTDualFreq'],

 # I-V panel related controls
 ['IVAmpDART', 'IV_Amp_DART', 'iv_amp_DART'],
 ['IVDoItDART', 'ivdoit_DART', 'IVDoit_DART'],
 ['IVFreqDART', 'IV_Freq_DART', 'iv_freq_DART'],
 ['OneTuneDART', 'TuneDART', 'DARTTune', 'DART_Tune'],
 ['IVAmp', 'IV_Amp', 'iv_amp'],
 ['IVDoIt', 'ivdoit', 'IVDoit'],
 ['IVFreq', 'IV_Freq', 'iv_freq'],

 # Video panel related controls
 ['Capture', 'TakePhoto']]
```
