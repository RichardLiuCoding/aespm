
from igor2 import binarywave as bw

import time
import os
from subprocess import Popen
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import shutil
import subprocess

from utils import *

def write_igor(commands, remote=False, connection=None, wait=0.35):
    '''
    Control Jupiter by writing and executing Igor command in the ToIgor.arcmd file.

    Input:
        commands   - String: AR command strings to be executed by Igor.exe
        wait       - Float: sleep time after the writing process is done.
        connection - SSH connection: return by utils.return_connection()

    Return:
        N/A

    Usage:
        # Enable the x PISLoop
        write_igor('ARExecuteControl("StartPIDSLoop0","PIDSLoopPanel",0,"")')

        # Enable the x PISLoop by remote control
        connection = return_connection()
        write_igor('ARExecuteControl("StartPIDSLoop0","PIDSLoopPanel",0,"")', remote=True, connection=connection)
    '''
    if connection==None:
        file = open(r"C:\\Users\\Asylum User\\Documents\\AEtesting\\ToIgor.arcmd","w",encoding = 'utf-8')
        file.writelines(commands)
        file.close()

        p = Popen("C:\\Users\\Asylum User\\Documents\\AEtesting\\SendToIgor.bat")
        p.wait()
        time.sleep(wait)
    else:
        write_to_remote_file(connection,
                             file_path = "C:\\Users\\Asylum User\\Documents\\AEtesting\\ToIgor.arcmd", data = commands)
        main_exe_on_server()
        time.sleep(wait)


def read_igor(key, commands=None, remote=False, connection=None):
    '''
    Read Jupiter parameters by exporting them into readout.txt file.

    Input:
        key       - list: keys in AR to read the parameter
        commands  - String: user-defined command to read parameters
        connection - SSH connection: return by utils.return_connection()

    Returns:
        An array of parameter values specified by key.

    Usage:
        # Read out x and y LVDT sensitivity
        keys = ['XLVDTSens', 'YLVDTSens']
        xsens, ysens = read_igor(key=keys)
    '''
    if connection == None:
        if commands == None:
            N = len(key)
            start = 'Make/N = ({})/O ReadOut // change N to number of parameters to read out\n'.format(N)
            command = ''
            for i in range(N):
                if key[i] == 'PIDSLoop.0.Setpoint' or key[i] == 'PIDSLoop.1.Setpoint':
                    command += 'ReadOut[{}] = td_ReadValue("{}")\n'.format(i, key[i])
                else:
                    command += 'ReadOut[{}] = GV("{}")\n'.format(i, key[i])
                # command += 'ReadOut[{}] = td_ReadValue("{}")\n'.format(i, key[i])
            end = r'Save/O/G/J ReadOut as "C:\\Users\\Asylum User\\Documents\\AEtesting\\readout.txt"'
            file = open("C:\\Users\\Asylum User\\Documents\\AEtesting\\ToIgor.arcmd","w",encoding = 'utf-8')
            file.writelines(start+command+end)
            file.close()
            p = Popen("C:\\Users\\Asylum User\\Documents\\AEtesting\\SendToIgor.bat")
            p.wait()
            return np.loadtxt(r"C:\\Users\\Asylum User\\Documents\\AEtesting\\readout.txt")

        else:
            file = open("C:\\Users\\Asylum User\\Documents\\AEtesting\\ToIgor.arcmd","w",encoding = 'utf-8')
            file.writelines(commands)
            file.close()

            p = Popen("C:\\Users\\Asylum User\\Documents\\AEtesting\\SendToIgor.bat")
            p.wait()
            return np.loadtxt(r"C:\\Users\\Asylum User\\Documents\\AEtesting\\readout.txt")

    else:
        if commands == None:
            N = len(key)
            start = 'Make/N = ({})/O ReadOut // change N to number of parameters to read out\n'.format(N)
            command = ''
            for i in range(N):
                command += 'ReadOut[{}] = td_ReadValue("{}")\n'.format(i, key[i])
            end = r'Save/O/G/J ReadOut as "C:\\Users\\Asylum User\\Documents\\AEtesting\\readout.txt"'
            commands = start+command+end
            write_to_remote_file(connection,
                         file_path = "C:\\Users\\Asylum User\\Documents\\AEtesting\\ToIgor.arcmd", data = commands)# doubt as richard
            main_exe_on_server()

            s = read_remote_file(connection, r"C:\\Users\\Asylum User\\Documents\\AEtesting\\readout.txt")

            return [float(k) for k in s.decode('utf-8').split('\r')[:-1]]

        else:
            write_to_remote_file(connection,
                             file_path = "C:\\Users\\Asylum User\\Documents\\AEtesting\\ToIgor.arcmd", data = commands)# doubt as richard

            main_exe_on_server()
            s = read_remote_file(connection, r"C:\\Users\\Asylum User\\Documents\\AEtesting\\readout.txt")

            return [float(k) for k in s.decode('utf-8').split('\r')[:-1]]
        
def spm_control(action, value=None, wait=0.35, connection=None):
    '''
    Control SPM with action. An action-command dict is used to convert the action/input to the actual AR commands.

    Input:
        action      - String: Name of the button or input to be executed.
        value       - Int/Float/String: New value for the parameter to change.
        wait        - Float: sleep time after the action is finished.
        connection  - SSH connection: return by return_connection()
    Return:
        N/A

    Examples:
        # Start a downward scan
        spm_control(action='ScanDown')
        # Change scan rate to 1 Hz
        spm_control(action='ScanRate', value=1)
    '''
    # key: action, value[0]: input/button name, value[1]: window/panel name, 
    # value[-1]: 
    #       0-Button, 1-numeric, 2-string, 4-pure commands, 
    #       5-use value[-2] for numeric input, 6-use value[-2] for string input
    action_dict = {}
    key_list = [
        ['ScanRate', 'ScanRates', 'speed'], # 0 
        ['DownScan', 'ScanDown', 'Start', 'start'],
        ['UpScan', 'ScanUp'],
        ['DriveAmp', 'DriveAmplitude', 'DriveVoltage', 'DriveVolt', 'v_ac', 'V_ac'],
        ['IGain', 'IGains', 'IntegralGain', 'igain'], # 4
        ['DriveFreq', 'DriveFrequency', 'Freq', 'Frequency'],
        ['Setpoint', 'SetPoint'],
        ['StopScan', 'Stop', 'stop'],
        ['ClearForce', 'Clear', 'ClearMarker'],
        ['ThatsIt', 'Thats'], # 9
        ['GoThere', 'Gothere'],
        ['TipVoltage', 'V_tip', 'v_tip'],
        ['SurfaceVoltage', 'V_surface', 'v_surface', 'v_surf'],
        ['SingleF', 'SingleForce'],
        ['EnableStage', 'StageEnable', 'EnableMotor', 'MotorEnable'], # 14
        ['Approach', 'StartApproach'], 
        ['DisableStage', 'StageDisable', 'DisableMotor', 'MotorDisable'],
        ['ChangeName', 'FileName', 'BaseName'],
        ['GetTune', 'TuneData'],
        ['IVAmp', 'IV_Amp', 'iv_amp'], # 19
        ['IVFreq', 'IV_Freq', 'iv_freq'],
        ['IVDoIt', 'ivdoit', 'IVDoit'],
        ['FBHeight', 'FeedbackHeight', 'PID2=Height', 'FB_Height', 'FB_height'],
        ['StartFB2', 'Engage', 'StartLoop2', 'StartLoopZ'],
        ['IVAmpDART', 'IV_Amp_DART', 'iv_amp_DART'], # 24
        ['IVFreqDART', 'IV_Freq_DART', 'iv_freq_DART'],
        ['IVDoItDART', 'ivdoit_DART', 'IVDoit_DART'],
        ["OneTuneDART", "TuneDART", "DARTTune", "DART_Tune"],
        ['StartFB0', 'StartLoop0', 'StartLoopX'],
        ['StartFB1', 'StartLoop1', 'StartLoopY'], # 29
        ['PFM_Mode', 'Mode=PFM', 'PFM'],
        ['AC_Mode', 'Mode=AC', 'AC'],
        ['ScanSize', 'Scan Size', 'scan size', 'scan_size'],
        ['Points and Lines', 'Points', 'Pixels', 'points', 'pixels'],
        ['DARTFreq', 'DART Freq', 'CentralFreq'], #34
        ['DARTPhase1', 'DART Phase1'],
        ['DARTPhase2', 'DART Phase2'],
        ['DARTWidth', 'DART Width', 'FreqWidth'],
        ['CenterPhase', 'Center Phase', 'PhaseCenter'],
        ['Cycles', 'Cycle', 'NumCycles'], # 39
        ['DualFreq', 'DARTMode', 'DARTDualFreq'],
        ['DARTSweepWidth', 'SweepWidth', ],
        ['Capture', 'TakePhoto'],
        ['ACTune','AC Tune', 'Tune'],
        ['XOffset', 'X Offset', 'Offset_X'], # 44
        ['YOffset', 'Y Offset', 'Offset_Y'],
        ['ScanAngle', 'Scan Angle', 'Angle'],
        ['SetpointAmp', 'AC Setpoint', 'SetpointAC'],
        ['SetpointDefl', 'PFMSetpoint', 'SetpointPFM', 'SetpointContact']
        
    ]

    value_list = [
        ['ScanRateSetVar_0', 'MasterPanel', 1], # 0
        ['DownScan_0', 'MasterPanel', 0],
        ['UpScan_0', 'MasterPanel', 0],
        ['DriveAmplitudeSetVar_0', 'MasterPanel', 1],
        ['IntegralGainSetVar_0', 'MasterPanel', 1], # 4
        ['DriveFrequencySetVar_0', 'MasterPanel', 1],
        ['td_WriteValue("Cypher.PIDSLoop.2.SetPoint", {})'.format(value), 4], #['td_WriteValue("Cypher.PIDSLoop.2.SetPoint", {})'.format(value), 4],
        ['StopScan_0', 'MasterPanel', 0],
        ['ClearForce_1', 'MasterPanel', 0],
        ['ForceSpotNumberSetVar_1', 'MasterPanel', 0], # 9 
        ['GoForce_1', 'MasterPanel', 0],
        ['TipVoltageSetVar_0', 'NapPanel', 1],
        ['SurfaceVoltageSetVar_0', 'NapPanel', 1],
        ['SingleForce_1', 'MasterPanel', 0],
        ["EnableStageCB_0","MasterMotorPanel", 1, 5], # 14
        ["MotorEngageButton_0","MasterMotorPanel",0], 
        ["EnableStageCB_0","MasterMotorPanel", 0, 5],
        ['ChangeName("{}")'.format(value), 4],
        ['GetTune()', 4],
        ["ARDoIVAmpSetVar_1","ARDoIVPanel", 1], # 19
        ["ARDoIVFrequencySetVar_1","ARDoIVPanel", 1],
        ["ARDoIVDoItButton_1","ARDoIVPanel", 0],
        ["DefaultPIDSLoop2","PIDSLoopPanel", "Height", 6],
        ["StartPIDSLoop2","PIDSLoopPanel", 0],
        ["ARDoIVAmpSetVar_1","DARTSpectroscopy", 1], # 24
        ["ARDoIVFrequencySetVar_1","DARTSpectroscopy", 1],
        ["SingleForce_2","DARTSpectroscopy", 0],
        ["DoTuneDFRT_3","DART", 0],
        ["StartPIDSLoop0","PIDSLoopPanel", 0],
        ["StartPIDSLoop1","PIDSLoopPanel", 0], # 29
        ["ImagingModePopup_0","MasterPanel","PFM Mode", 6],
        ["ImagingModePopup_0","MasterPanel","AC Mode", 6],
        ['ScanSizeSetVar_0', 'MasterPanel', 1],
        ['PointsLinesSetVar_0', 'MasterPanel', 1],
        ["DFRTFrequencyCenterSetvar","DART", 1], # 34
        ["PhaseOffsetSetVar_3","DART", 1],
        ["PhaseOffset1SetVar_3","DART", 1],
        ["DFRTFrequencyWidthSetvar","DART", 1],
        ["DoTuneCent_3", "DART", 1], 
        ["ARDoIVCyclesSetVar_1","DARTSpectroscopy", 1], # 39
        ["DualACModeBox_3", "DART", 1], 
        ["SweepWidthSetVar_3", "DART", 1], 
        ["ARVCaptureButton_0","ARVideoPanel", 0],
        ["DoTuneOnceButton","TuneGraph", 0],
        ['PV("XOffset", {})'.format(value), 4], # 44
        ['PV("YOffset", {})'.format(value), 4],
        ['PV("ScanAngle", {})'.format(value), 4],
        ['PV("AmplitudeSetpointVolts", {})'.format(value), 4],
        ['PV("DeflectionSetpointVolts", {})'.format(value), 4],
        
    ]
    # Construct the action dict
    for i, key in enumerate(key_list):
        for j in range(len(key)):
            action_dict[key[j]] = value_list[i]

    cmd = action_dict[action]

    if cmd[-1] == 0: # Button
        commands = 'ARExecuteControl("{}","{}",0,"")'.format(cmd[0], cmd[1])
    elif cmd[-1] == 1: # Numeric input
        commands = 'ARExecuteControl("{}","{}",{},"")'.format(cmd[0], cmd[1], value)
    elif cmd[-1] == 2: # String input
        commands = 'ARExecuteControl("{}","{}",0,"{}")'.format(cmd[0], cmd[1], value)
    elif cmd[-1] == 4: # Pure commands
        commands = cmd[0]
    elif cmd[-1] == 5: # Default numeric input
        commands = 'ARExecuteControl("{}","{}",{},"")'.format(cmd[0], cmd[1], cmd[2])
    elif cmd[-1] == 6: # Default string input
        commands = 'ARExecuteControl("{}","{}",0,"{}")'.format(cmd[0], cmd[1], cmd[2])
    else:
        raise ValueError("Function not implemented yet.")

    write_igor(commands=commands, connection=connection, wait=wait)


# Move the tip 
def move_tip(r, v0=None, s=None, remote=False, connection=None):
    '''
    Move the tip position based on given displacement.

    Input:
        r   - list: displacement vectors of [x_delta, y_delta]
        v0  - list: voltage applied [V0x, V0y] (returned from read_igor)
        s   - list: sensitivity of LVDT [sx, sy] for x and y
        connection  - SSH connection: return by return_connection()

    Returns:
        N/A

    Usage:
        keys = ['PIDSLoop.0.Setpoint', 'PIDSLoop.1.Setpoint']
        Vx0, Vy0 = read_igor(key=keys)
        r = [1e-6, 1e-6]
        v0 = [Vx0, Vy0]
        s = [xsens, ysens]
        move_tip(r=r, v0=v0, s=s)
    '''
    x_delta, y_delta = r
    if v0 is None:
        v0x, v0y = read_igor(key=['PIDSLoop.0.Setpoint', 'PIDSLoop.1.Setpoint'], connection=connection)
    else:
        v0x, v0y = v0
    if s is None:
        sx, sy = read_igor(key=['XLVDTSens', 'YLVDTSens'], connection=connection)
    else:
        sx, sy = s
    vx = x_delta / sx
    vy = y_delta / sy

    command1 = 'td_WriteValue("PIDSLoop.0.Setpoint",{})\n'.format(vx+v0x)
    command2 = 'td_WriteValue("PIDSLoop.1.Setpoint",{})\n'.format(vy+v0y)

    write_igor(commands=command1+command2, connection=connection)


def move_stage(distance, remote=False, connection=None):
    '''
    Move the stage position based on given displacement.

    Input:
        distance   - list: displacement vectors of [x_delta, y_delta]
        remote     - Boolean: True for remote control, default is False (local run)
        connection - SSH connection: return by utils.return_connection()

    Returns:
        N/A

    Usage:
        move_stage(distance=[x_delta, y_delta])
    '''
    # Determine the moving direction of the stage
    x, y = distance
    x_direction = 'MoveStageLeftButton_0' if x > 0 else 'MoveStageRightButton_0'
    y_direction = 'MoveStageDownButton_0' if y > 0 else 'MoveStageUpButton_0'

    # Change the motor step size for x direction moving
    if x != 0:
        write_igor(commands='PV("StageMoveStepSize", {})'.format(abs(x)), connection=connection)
        write_igor(commands='MoveStage("{}")'.format(x_direction), connection=connection)
        # write_igor(commands='ARExecuteControl("{}","MasterMotorPanel#StepAndVac#StepPanel",0,"")'.format(x_direction), connection=connection)
    
    if y != 0:
        # Change the motor step size for y direction moving
        write_igor(commands='PV("StageMoveStepSize", {})'.format(abs(y)), connection=connection)
        write_igor(commands='MoveStage("{}")'.format(y_direction), connection=connection)
        # write_igor(commands='ARExecuteControl("{}","MasterMotorPanel#StepAndVac#StepPanel",0,"")'.format(y_direction), connection=connection)


def tune_probe(num=3, path=r"C:\Users\Asylum User\Documents\AEtesting\Tune.ibw", center=None, width=50e3, out=False, connection=None):
    for i in range(num):
        spm_control('OneTuneDART', wait=1, connection=connection)
        spm_control('GetTune', wait=0.5, connection=connection)
        if connection is not None:
            download_file(connection=connection, file_path=path, local_file_name='tune.ibw')
            w = ibw_read('tune.ibw')
        else:
            w = ibw_read(path)
        freq = w[0][w[1].argmax()]
        if center is not None:
            if abs(freq-center) > width:
                freq = center
        spm_control('DARTFreq', value=freq, connection=connection)
    spm_control('GetTune', wait=0.5, connection=connection)
    if connection is not None:
        download_file(connection=connection, file_path=path, local_file_name='tune.ibw')
        w = ibw_read('tune.ibw')
    else:
        w = ibw_read(path)
    # Get the freq corresponds to the max intensity
    freq = w[0][w[1].argmax()]
    # Set this freq to be the driven freq
    spm_control('DARTFreq', value=freq, connection=connection)
    spm_control('CenterPhase', connection=connection)
    if out == True:
        return w

def get_files(path, retry=10, sleep_time=6e-3, client=None):
    '''
    Get the file names and sort them in the new to old modification time from a given folder.
    '''
    command = 'dir "{}" /b /o:-d'.format(path)
    retries = 0
    while retries < retry:
        try:
            # Run the command and capture its output
            if client == None:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
                file_names = result.stdout.split('\n')
                return [os.path.join(path, file) for file in file_names]
            else:
                stdin, stdout, stderr = client.exec_command(command)
                # Read the output of the command
                file_names = stdout.read().decode('utf-8').splitlines()
                return [os.path.join(path, file) for file in file_names]
            
        except FileNotFoundError:
            print("File not found. Retrying {} times...".format(retries), end='\r')
            retries += 1
            time.sleep(sleep_time)
        except PermissionError:
            print("Permission denied. Retrying {} times...".format(retries), end='\r')
            retries += 1
            time.sleep(sleep_time)
    return 0

def check_file_number(path, wait=1e-1, retry=1e4, client=None):
    '''
    Check if the spectrum/topo measurement is done by monitoring the number of files in the save folder.
    It waits until 
    '''
    num = len(get_files(path, client=client))
    num_new = num
    retries = 0
    # Check the file number in the folder until 16 min is passed
    while num_new == num and retries < retry:
        num_new = len(get_files(path, client=client))
        retries += 1
        time.sleep(wait)
    return True


def ibw_read(fname, retry=10, wait=0.1, copy=True, connection=None):
    '''
    Make a copy of realtime saved image and read the wave data from it.
    Three data channels will be read with the order of ZSenor/Height -> Amplitude -> Phase -> Height/Deflection.
    Each channel contains the trace and retrace scan lines.
    '''
    retries = 0
    if connection==None:
        if copy==True:
            while retries < retry:
                try:
                    shutil.copy(fname, "C:\\Users\\Asylum User\\Documents\\AEtesting\\copy.ibw")
                    data = bw.load("C:\\Users\\Asylum User\\Documents\\AEtesting\\copy.ibw")
                    wave = data['wave']['wData']
                    return wave.T
                except FileNotFoundError:
                    print("File not found. Retrying {} times...".format(retries), end='\r')
                    retries += 1
                    time.sleep(wait)
                except PermissionError:
                    print("Permission denied. Retrying {} times...".format(retries), end='\r')
                    retries += 1
                    time.sleep(wait)
        else:
            return bw.load(fname)['wave']['wData'].T
    else:
        download_file(connection=connection, file_path=fname, local_file_name='temp.ibw')
        time.sleep(wait)
        return bw.load('temp.ibw')['wave']['wData'].T
    return 0

# Function to update and display the plot in real-time
def update_plot_all(folder):
    try:
        while True:
            # Read data from the file
            fname = get_files(folder)
            w = ibw_read(fname[0])

            # Clear the previous plot
            clear_output(wait=True)
            fig,ax = plt.subplots(1,4,figsize=[14,3])
            # Update the plot
            ax[0].plot(w[0], 'r')
            ax[0].plot(w[1], 'b')
            ax[0].set_title("Z Sensor/Height")
            ax[0].set_xlabel("Pixels")
            ax[0].set_ylabel("Z Sensor/Height")

            ax[1].plot(w[2], 'r')
            ax[1].plot(w[3], 'b')
            ax[1].set_title("Amplitude")
            ax[1].set_xlabel("Pixels")
            ax[1].set_ylabel("Amplitude")

            ax[2].plot(w[4], 'r')
            ax[2].plot(w[5], 'b')
            ax[2].set_title("Phase")
            ax[2].set_xlabel("Pixels")
            ax[2].set_ylabel("Phase")

            ax[3].plot(w[6], 'r')
            ax[3].plot(w[7], 'b')
            ax[3].set_title("Height/Deflection")
            ax[3].set_xlabel("Pixels")
            ax[3].set_ylabel("Height/Deflection")

            plt.show()

    except KeyboardInterrupt:
        print("Real-time plotting stopped.")
