#Ifdef ARrtGlobals
#pragma rtGlobals=1        // Use modern global access method.
#else
#pragma rtGlobals=3        // Use strict wave reference mode
#endif 
#include ":AsylumResearch:Code3D:Initialization"
#include ":AsylumResearch:Code3D:MotorControl"
#include ":AsylumResearch:Code3D:UserPanels:DART", optional
#include ":AsylumResearch:Code3D:PZTHyst"

// Function to get tuning data
Function GetTune()
    String FileName = "C:\\Users\\Asylum User\\Documents\\buffer\\Tune.ibw"
    String DataFolder = GetDF("Main")
    // Take out Z Sensor trace and retrace
    Wave Freq = root:packages:MFP3D:Tune:Frequency
    Wave Amp = root:packages:MFP3D:Tune:Amp
    Variable nop = DimSize(Freq,0)  //we know they have to be the same number of points
    // Take out Amplitude trace and retrace
    Make/N=(nop*1,2)/O $DataFolder+"TuneScope"/Wave=TuneScope
    TuneScope[0,nop-1][0] = Freq[P]
    TuneScope[0,nop-1][1] = Amp[P]
    Save/C/O/P=SaveImage TuneScope as FileName
End //GetTune

// Function to read the meter panels
Function GetMeter()
    String FileName = "C:\\Users\\Asylum User\\Documents\\buffer\\Meter.ibw"
    String DataFolder = GetDF("Main")
    // Take out Z Sensor trace and retrace
    Wave Meter = root:packages:MFP3D:Meter:ReadMeterRead
    Variable nop = DimSize(Meter,0)  //we know they have to be the same number of points
    // Take out Amplitude trace and retrace
    Make/N=(nop)/O $DataFolder+"MeterRead"/Wave=MeterRead
    MeterRead[0,nop-1] = Meter[P]
    Save/C/O/P=SaveImage MeterRead as FileName
End //GetTune

// FUnction to change base filename
Function ChangeName(NewName)
    String NewName
//    Struct WMSetvariableAction ButtonStruct
//    ButtonStruct.EventCode = 2
//    ButtonStruct.Win = "MasterPanel"
//    ButtonStruct.CtrlName = "BaseNameSetVar_0"
//    ButtonStruct.SVal = NewName
//    ButtonStruct.DVal = 0
//    BaseNameSetVarFunc(ButtonStruct)
      BaseNameSetVarFunc("BaseNameSetVar_0",Nan,NewName,"BaseName")
End //

// Function to move stage
//Function MoveStage(Direction)
//    String Direction
//    Struct WMButtonAction ButtonStruct
//
//    ButtonStruct.EventCode = 2
//    //  ButtonStruct.Win = "MasterPanel"
//    ButtonStruct.CtrlName = Direction
//    StageButtonFunc(ButtonStruct)
//End //
