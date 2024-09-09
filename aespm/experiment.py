
from igor2 import binarywave as bw

import time
import os
from subprocess import Popen
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import shutil
import subprocess

import types
import pickle

import aespm
from aespm.utils import shared

import platform

if platform.system() == 'Windows':
    _buffer_path = os.path.join(os.path.expanduser('~'), 'Documents', 'buffer')
    _command_buffer = os.path.join(_buffer_path, 'ToIgor.arcmd')
    _read_out_buffer = os.path.join(_buffer_path, 'readout.txt')
    _bash_buffer = os.path.join(_buffer_path, 'SendToIgor.bat')
    _path_txt = os.path.join(_buffer_path, 'path.txt')
    
    with open(_path_txt, 'r') as fopen:
        _exe_path = fopen.readline()
    
    # _exe_path = r"C:\AsylumResearch\v19\RealTime\Igor Pro Folder\Igor.exe"
else:
    _buffer_path = r"C:\Users\Asylum User\Documents\buffer"
    _command_buffer = r"C:\Users\Asylum User\Documents\buffer\ToIgor.arcmd"
    _read_out_buffer = r"C:\Users\Asylum User\Documents\buffer\readout.txt"
    _bash_buffer = r"C:\Users\Asylum User\Documents\buffer\SendToIgor.bat"
    _exe_path = r"C:\AsylumResearch\v19\RealTime\Igor Pro Folder\Igor.exe"

class Experiment(object):
    '''
    Experiment data structure for SPM workflows.

    Attributes:
        obj.param      - Dict: A collection of parameters can be updated with obj.update_param
        obj.folder     - String: Data saving folder on the local computer. Img/Spec data will be retrieved as 
                            the latest modified file in this folder.
        obj.temp_folder- String: Data exchange folder on the local computer. Commands and parameter reading buffer files
                            will be saved in this folder.
        obj.action_list- List: Names of custom functions. It will override default action list if doubled
        obj.log        - List: logged operations/parameter changes/reading
        obj.connection - SSH connection: return by utils.return_connection()
        obj.client     - SSH Client that can directly run commands on the local computer: return by utils.return_connection()
        
    Methods:
        obj.execute          - Execute a single operation, a wrapper of spm_control()
        obj.execute_sequence - Execute a list of operations in sequence
        obj.update_param     - update parameters in obj.param
        obj.add_func         - Add a custom function as the new method of obj
        obj.save_exp         - Save the whole experiment obj as .pkl file
        obj.log_operation    - log all the operations in obj.log
        obj.print_log        - print the logged operations nicely

    Usage:
        # Connection: [host, username, password]
        exp = Experiment(folder=your_folder, temp_folder=you_temp_folder, connection=connectio)
    '''
    def __init__(self, folder, temp_folder=r"C:\Users\Asylum User\Documents\AEtesting\data_exchange", connection=None):
        super(Experiment, self).__init__()

        # log of all the operations
        self.log = [] 
        # Initialize parameter dict 
        self.param = {} 
        # Initialize action list of custom functions 
        self.action_list = {}
        # This is the folder where your measurements are saved
        self.folder = folder

        # This is the folder where your temp data/command files are saved
        self.temp_folder = temp_folder
        
        # Remote control
        if connection is not None:
            host, username, password = connection
            local_info = aespm.utils.get_local_directory(host=host).split('$')
            global shared
            #shared = SharedInfo(host, local_info)--> future implementation
            shared.set_values((_buffer_path, _command_buffer, _read_out_buffer, _bash_buffer, _exe_path))
            shared.set_host(host)
            self.connection, self.client = aespm.utils.return_connection(host, username, password)

        else:
            self.connection, self.client = None, None

    def execute(self, action, value=None, wait=None, log=True, **kwargs):
        '''
        A wrapper of spm_control() function. 

        Input:
            action      - String: SPM instructions in hyper-language.
            value       - Int/Float/String: New value for the parameter to change.
            wait        - Float: sleep time after the action is finished.
            log         - Boolean: If true, this action will be logged in obj.log
            kwargs      - Keyword arguments for custom functions

        Return:
            N/A

        Examples:
            # Start a downward scan
            obj.execute(action='ScanDown')
            # Change scan rate to 1 Hz
            obj.execute(action='ScanRate', value=1)
        '''
        
        wait = 0.35 if wait is None else wait
        if wait <= 0.35:
            wait = 0.35

        # custom functions
        if action in self.action_list:
            if value == None:
                return getattr(self, action)()
            else:
                if type(value) != list and type(value) != np.ndarray:
                    value = [value]
                return getattr(self, action)(*value, **kwargs)
            time.sleep(wait)

        # default action list 
        else:
            if type(value) != list and type(value) != np.ndarray:
                value = [value]
            try:
                spm_control(action=action, value=value[0], wait=wait, connection=self.connection)
            except KeyError:
                print("Invalid action: not found in default list or custom list.")

        if log is True:
            self.log_operation(operation=[action, value, kwargs], dtype=0)

    def execute_sequence(self, operation, log=True, **kwargs):
        '''
        Execute multiple operations sequentially. 

        Input:
            operation - List: [[Action1, Value1, Wait_Time1], ..., [ActionN, ValueN, Wait_TimeN]]
                               if there are keyword args, using [[Action1, Value1, Wait_Time1, Dict1], ...]
            log       - Boolean: If true, all operations will be logged in obj.log
            kwargs    - Keyword arguments for custom functions

        Output:
            N/A

        Usage:
            obj.execute_sequence(operation=[['Setpoint', 0.1, None], ['DownScan', None, 1.5]])
        '''
        for item in operation:
            if item[0] in self.action_list:
                if len(item) > 3:
                    self.execute(action=item[0], value=item[1], wait=item[2], log=log, **item[-1])
                else:
                    self.execute(action=item[0], value=item[1], wait=item[2], log=log)
            else:
                self.execute(action=item[0], value=item[1], wait=item[2], log=log)

    def update_param(self, key, value, log=True):
        '''
        Update the value stored in obj.param

        Input:
            key     - List: keys to be modifies in obj.param
            value   - List: values to be entered in obj.param['key']
            log     - Boolean: If true, all operations will be logged in obj.log

        Output:
            N/A

        Usage:
            obj.update_param(key=['DriveAmplitude', 'Setpoint'], value=[0.1, 0.2])
        '''
        if type(key) is not list:
            key = [key]
        
        if len(key) == 1:
            self.param[key[0]] = value
            
        else:
            for i, ix in enumerate(key):
                self.param[ix] = value[i]
                if log is True:
                    self.log_operation(operation=[ix], dtype=1)
                    # self.log_operation(operation=[ix, value[i]], dtype=1)

    def log_operation(self, operation, dtype=0, **kwargs):
        '''
        Formats and logs an operation in hyper-language syntax in obj.log.
        Input:
            operation - List: [Action, Value, Wait_Time], the same as inputs used in spm_control()
            dtype      - Int: 0 - Operations/actions, 1 - Parameter update, 2 - Read data 
        Output:
            N/A

        Usage:
            obj.log(operation=['DownScan', None, 1.5]) # Start a downscan
            obj.log(operation=['Setpoint', 0.1, None]) # Change setpoint to 0.1 V
            obj.log(operation=[['DriveAmplitude', 'Setpoint'], [0.1, 0.2], type=1)
        '''
        to_log = {}
        to_log[operation[0]] = operation
        to_log['type'] = dtype
        self.log.append(to_log)

    def add_func(self, NewFunc, log=True):
        '''
        Add a custom function as the method to Experiment object.

        Input:
            NewFunc - Function: Custom function defined by user. This function has aceess 
                        to all the attributes and methods of obj
        Output:
            N/A 
        Usage:
            def measure(self, operation, key, value):
                self.update_param(key=key, value=value)
                self.execute(operation)
            obj.add_func(measure)
        '''
        # method_name = NewFunc.__name__
        # setattr(self, method_name, NewFunc)
        # getattr(self, method_name).__doc__ = NewFunc.__doc__

        # # add new function name to the custom action list
        # self.action_list.append(method_name)

        method_name = NewFunc.__name__
        
        # add new function name to the custom action list
        self.action_list[method_name] = method_name
        
        # Bind the function as a method of the instance
        bound_method = types.MethodType(NewFunc, self)
        
        # # Optionally, update the docstring (if necessary)
        # getattr(self, method_name).__doc__ = NewFunc.__doc__

        # Set the method to the instance
        setattr(self, method_name, bound_method)
        

        if log is True:
            self.log_operation(operation=[method_name], dtype=3)

    def save_exp(self, save_name):
        '''
        Save the whole experiment object as .pkl file.
        Input: 
            save_name - String: save name for the .pkl file
        Output:
            N/A
        Usage:
            obj.save_exp('my exp')
        '''

        with open('{}.pkl'.format(save_name), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def print_log(self):
        '''
        Nicely plot the logged operations.
        '''
        for i, log_item in enumerate(self.log):
            if log_item['dtype'] == 0: # Normal operations
                print("Operation - {}: Action={}, Value={}, wait={}".format(i, 
                    log_item['operation'][0], log_item['operation'][1], log_item['operation'][2]))
            elif log_item['dtype'] == 1: # Normal operations
                print("Update - {}: Param={}, Value={}".format(i, 
                    log_item['operation'][0], log_item['operation'][1]))
            elif log_item['dtype'] == 2: # Read data
                print("Reading - {}: Data={}, Value={}".format(i, 
                    log_item['operation'][0], log_item['operation'][1]))
            elif log_item['dtype'] == 3: # Add custom function
                print("Add new function - {}: Name={}".format(i, 
                    log_item['operation'][0]))

            else:
                print("Unsupported operations in the log.")

def write_spm(commands, connection=None, wait=0.35):
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
        write_spm('ARExecuteControl("StartPIDSLoop0","PIDSLoopPanel",0,"")')

        # Enable the x PISLoop by remote control
        connection = return_connection()
        write_spm('ARExecuteControl("StartPIDSLoop0","PIDSLoopPanel",0,"")', connection=connection)
    '''
    if connection==None:
        file = open(_command_buffer,"w",encoding = 'utf-8')
        file.writelines(commands)
        file.close()

        p = Popen(_bash_buffer)
        p.wait()
        time.sleep(wait)
    else:
        aespm.utils.write_to_remote_file(connection,
                             file_path = _command_buffer, data = commands)
        aespm.utils.main_exe_on_server(host=shared._host)
        time.sleep(wait)


def read_spm(key, commands=None, connection=None):
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
        xsens, ysens = read_spm(key=keys)
    '''
    if type(key) != list and type(key) != np.ndarray:
        key = [key]
    if connection == None:
        if commands == None:
            N = len(key)
            start = 'Make/N = ({})/O ReadOut // change N to number of parameters to read out\n'.format(N)
            command = ''
            for i in range(N):
                if key[i].startswith('PIDSLoop'): # == 'PIDSLoop.0.Setpoint' or key[i] == 'PIDSLoop.1.Setpoint':
                    command += 'ReadOut[{}] = td_ReadValue("{}")\n'.format(i, key[i])
                else:
                    command += 'ReadOut[{}] = GV("{}")\n'.format(i, key[i])
            end = r'Save/O/G/J ReadOut as "{}"'.format(_read_out_buffer.replace('\\', '\\\\'))
            file = open(_command_buffer,"w",encoding = 'utf-8')
            file.writelines(start+command+end)
            file.close()
            p = Popen(_bash_buffer)
            p.wait()
            return np.loadtxt(_read_out_buffer)

        else:
            file = open(_command_buffer,"w",encoding = 'utf-8')
            file.writelines(commands)
            file.close()

            p = Popen(_bash_buffer)
            p.wait()
            return np.loadtxt(_read_out_buffer)

    else:
        if commands == None:
            N = len(key)
            start = 'Make/N = ({})/O ReadOut // change N to number of parameters to read out\n'.format(N)
            command = ''
            for i in range(N):
                if key[i].startswith('PIDSLoop'): # == 'PIDSLoop.0.Setpoint' or key[i] == 'PIDSLoop.1.Setpoint':
                    command += 'ReadOut[{}] = td_ReadValue("{}")\n'.format(i, key[i])
                else:
                    command += 'ReadOut[{}] = GV("{}")\n'.format(i, key[i])
            end = r'Save/O/G/J ReadOut as "{}"'.format(_read_out_buffer.replace('\\', '\\\\'))
            commands = start+command+end
            aespm.utils.write_to_remote_file(connection,
                         file_path = _command_buffer, data = commands)# doubt as richard
            aespm.utils.main_exe_on_server(host=shared._host)

            s = aespm.utils.read_remote_file(connection, _read_out_buffer)

            return [float(k) for k in s.decode('utf-8').split('\r')[:-1]]

        else:
            aespm.utils.write_to_remote_file(connection,
                             file_path = _command_buffer, data = commands)# doubt as richard

            aespm.utils.main_exe_on_server(host=shared._host)
            s = aespm.utils.read_remote_file(connection, _read_out_buffer)

            return [float(k) for k in s.decode('utf-8').split('\r')[:-1]]
        
def spm_control(action, value=None, wait=0.35, connection=None):
    '''
    Control SPM with action. An action-command dict is used to convert the action/input to the actual AR commands.

    Input:
        action      - String: SPM instructions in hyper-language.
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
        ['IGain', 'IGains', 'IntegralGain', 'igain', 'I'], # 4
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
        ['SetpointDefl', 'PFMSetpoint', 'SetpointPFM', 'SetpointContact'],
        ['BlueDriveAmp', 'BlueDriveAmplitude'], # 49
        ['FDTrigger', 'FDTriggerPoint'],
        ['ZeroPD', 'PDZero'],
        ['DARTTrigger', 'SSTrigger'],
        ['DARTAmp', 'DART v_ac'],
        ['SampleHeight', 'Sample Height'], # 54
        ['DARTIGain', 'DART i_gain', 'DART_I_Gain', 'DART I Gain'],
        ['Arg4', 'ARG 4', 'arg4', 'ARG4'],
        ['SamplingFreq', 'Sampling Freq'],
        # KPFM related
        ['Trigger Point', 'TriggerPoint', 'ElectricalTuneTrigger'],
        ['NapTipVoltage', 'TipVoltageNap'], # 59
        ['Potential I Gain', 'PotentialIGain', 'ElectricalIGain', 'Electrical I Gain'],
        ['Potential P Gain', 'PotentialPGain', 'ElectricalPGain', 'Electrical P Gain'],
        ['ElectricalTune', 'ElectricalTuneOnce'],
        ['ElectricalTuneCenter', 'ElectricalTuneCenterPhase'],
        ['SingleForce', 'ElectricalSingleForce', 'NapSingleForce'], # 64
        ['GetForce'],
        
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
        ['bdDriveAmplitudeSetVar_0', 'MasterPanel', 1], # 49
        ['TriggerPointSetVar_1', 'MasterPanel', 1],
        ["ZeroLDPDButton_1","MasterMotorPanel",0], 
        ["TriggerPointSetVar_2","DARTSpectroscopy", 1],
        ["DriveAmplitudeSetVar_3", "DART", 1], 
        ['PV("SampleHeight", {})'.format(value), 4], # 54
        ["DARTIGainSetVar_0", "DART", 1], 
        ["ARDoIVArg3SetVar_1","DARTSpectroscopy", 1],
        ["NumPtsPerSecSetVar_2","DARTSpectroscopy", 1],
        # KPFM related
        ["TriggerPointSetVar_0","ElectricPanel", 1],
        ["NapTipVoltageSetVar_0","ElectricPanel", 1], # 59
        ["PotentialIGainSetVar_0","ElectricPanel", 1],
        ["PotentialPGainSetVar_0","ElectricPanel", 1],
        ["DoTuneElectric_0","ElectricPanel", 0],
        ["DoTuneCent_0","ElectricPanel", 0],
        ["SingleForce_0","ElectricPanel", 0], # 64
        ['GetForce()', 4],
        
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

    write_spm(commands=commands, connection=connection, wait=wait, )


# Move the tip 
def move_tip(r, v0=None, s=None, connection=None):
    '''
    Move the tip position based on given displacement.

    Input:
        r   - list: displacement vectors of [x_delta, y_delta]
        v0  - list: voltage applied [V0x, V0y] at the old position (returned from read_spm)
        s   - list: sensitivity of LVDT [sx, sy] for x and y
        connection  - SSH connection: return by return_connection()

    Returns:
        N/A

    Usage:
        keys = ['PIDSLoop.0.Setpoint', 'PIDSLoop.1.Setpoint']
        Vx0, Vy0 = read_spm(key=keys)
        r = [1e-6, 1e-6]
        v0 = [Vx0, Vy0]
        s = [xsens, ysens]
        move_tip(r=r, v0=v0, s=s)
    '''
    x_delta, y_delta = r
    if v0 is None:
        v0x, v0y = read_spm(key=['PIDSLoop.0.Setpoint', 'PIDSLoop.1.Setpoint'], connection=connection)
    else:
        v0x, v0y = v0
    if s is None:
        sx, sy = read_spm(key=['XLVDTSens', 'YLVDTSens'], connection=connection)
    else:
        sx, sy = s
    vx = x_delta / sx
    vy = y_delta / sy

    command1 = 'td_WriteValue("PIDSLoop.0.Setpoint",{})\n'.format(vx+v0x)
    command2 = 'td_WriteValue("PIDSLoop.1.Setpoint",{})\n'.format(vy+v0y)

    write_spm(commands=command1+command2, connection=connection, )


def move_stage(distance, connection=None):
    '''
    Move the stage position based on given displacement.

    Input:
        distance   - list: displacement vectors of [x_delta, y_delta]
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
        write_spm(commands='PV("StageMoveStepSize", {})'.format(abs(x)), connection=connection)
        write_spm(commands='MoveStage("{}")'.format(x_direction), connection=connection)
    
    if y != 0:
        # Change the motor step size for y direction moving
        write_spm(commands='PV("StageMoveStepSize", {})'.format(abs(y)), connection=connection)
        write_spm(commands='MoveStage("{}")'.format(y_direction), connection=connection)


def tune_probe(num=1, path=os.path.join(_buffer_path, 'Tune.ibw'), center=None, width=50e3, out=False, readonly=False, connection=None):
    '''
    Tune the probe in the DART mode and optionally read the tune data.

    Input:
        num     - Int: how many times tune will be repeated
        path    - String: path where the tune result is saved
        center  - float: center of the tuning frequency range
        width   - float: frequency range of runing
        out     - Boolean: flag to indicate if tune data will be output
        readonly- Boolean: flag to indicate if tune will be read only or after retune
    Output:
        tune    - an AR wave containing frequency and amplitude
    Example:
        w = ae.tune_probe(num=2, out=True)
    '''
    if readonly is False:
        for i in range(num):
            spm_control('OneTuneDART', wait=1, connection=connection)
            spm_control('GetTune', wait=0.5, connection=connection)
            if connection is not None:
                aespm.utils.download_file(connection=connection, file_path=path, local_file_name='tune.ibw')
                w = ibw_read('tune.ibw').data
            else:
                w = ibw_read(path).data
            freq = w[0][w[1].argmax()]
            if center is not None:
                if abs(freq-center) > width:
                    freq = center
            spm_control('DARTFreq', value=freq, connection=connection)
        spm_control('GetTune', wait=0.5, connection=connection)
        if connection is not None:
            aespm.utils.download_file(connection=connection, file_path=path, local_file_name='tune.ibw')
            w = ibw_read('tune.ibw').data
        else:
            w = ibw_read(path).data
        # Get the freq corresponds to the max intensity
        freq = w[0][w[1].argmax()]
        # Set this freq to be the driven freq
        spm_control('DARTFreq', value=freq, connection=connection)
        spm_control('CenterPhase', connection=connection)
    else:
        spm_control('GetTune', wait=0.5, connection=connection)
        if connection is not None:
            aespm.utils.download_file(connection=connection, file_path=path, local_file_name='tune.ibw')
            w = ibw_read('tune.ibw').data
        else:
            w = ibw_read(path).data
            
    if out == True:
        return w

def get_files(path, retry=10, sleep_time=6e-3, client=None):
    '''
    Get the file names and sort them in the new to old modification time from a given folder.

    Input:
        path    - String: path to the folder where the SPM data is saved locally
        retry   - Int: number of retries if any of the data file is occupied by SPM controller
        sleep_time - float: wait time between the retries
        client  - SSH client returned by utils.return_connection()
    Output:
        fname   - List: a list of filenames in the order of new to old modification time
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
    It waits until the number of files has changed to return a True value.
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

def ibw_read(fname, retry=10, wait=0.1, lines=False, connection=None):
    '''
    Make a copy of realtime saved image and read the wave data from it.
    Four data channels will be read with the order of ZSenor/Height -> Amplitude -> Phase -> Height/Deflection.
    Each channel contains both the trace and retrace scan lines.
    '''
    retries = 0
    if connection==None:
        if lines==True:
            while retries < retry:
                try:
                    shutil.copy(fname, os.path.join(_buffer_path, 'copy.ibw'))
                    data = bw.load(os.path.join(_buffer_path, 'copy.ibw'))
                    wave = data['wave']['wData']
                    return wave.T
                    # return aespm.tools.load_ibw(fname)
                except FileNotFoundError:
                    print("File not found. Retrying {} times...".format(retries), end='\r')
                    retries += 1
                    time.sleep(wait)
                except PermissionError:
                    print("Permission denied. Retrying {} times...".format(retries), end='\r')
                    retries += 1
                    time.sleep(wait)
        else:
            return aespm.tools.load_ibw(fname)
            # return bw.load(fname)['wave']['wData'].T
    else:
        aespm.utils.download_file(connection=connection, file_path=fname, local_file_name='temp.ibw')
        time.sleep(wait)
        if lines == True:
            return bw.load('temp.ibw')['wave']['wData'].T
        else:
            return aespm.tools.load_ibw('temp.ibw')
        # return bw.load('temp.ibw')['wave']['wData'].T
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
