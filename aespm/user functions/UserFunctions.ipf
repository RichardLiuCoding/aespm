#Ifdef ARrtGlobals
#pragma rtGlobals=1        // Use modern global access method.
#else
#pragma rtGlobals=3        // Use strict wave reference mode
#endif 
#include ":AsylumResearch:Code3D:Initialization"
#include ":AsylumResearch:Code3D:MotorControl"
#include ":AsylumResearch:Code3D:UserPanels:DART", optional
#include ":AsylumResearch:Code3D:PZTHyst"

Function GetTune()
    String FileName = "C:\\Users\\Asylum User\\Documents\\buffer\\Tune.ibw"
    String DataFolder = GetDF("Main")
    // Take out Freq, Amp, and Phase of the tune curve
    Wave Freq = root:packages:MFP3D:Tune:Frequency
    Wave Amp = root:packages:MFP3D:Tune:Amp
	Wave Phase = root:packages:MFP3D:Tune:Phase
    Variable nop = DimSize(Freq,0)  //we know they have to be the same number of points
    // Create a TuneScope wave to store Freq, Amp, and Phase of the tune
    Make/N=(nop*1,3)/O $DataFolder+"TuneScope"/Wave=TuneScope
    TuneScope[0,nop-1][0] = Freq[P]
    TuneScope[0,nop-1][1] = Amp[P]
	TuneScope[0,nop-1][2] = Phase[P]
    Save/C/O/P=SaveImage TuneScope as FileName
End //GetTune

Function GetForce()
    String FileName = "C:\\Users\\Asylum User\\Documents\\buffer\\Force.ibw"
    String DataFolder = GetDF("Main")
    // Take out ZSensor, Amplitude, and Phase of the force curve
    Wave ZSensor = root:packages:MFP3D:Force:ZSensor
    Wave Amplitude = root:packages:MFP3D:Force:Amplitude
	Wave Phase = root:packages:MFP3D:Force:Phase
    Variable nop = DimSize(ZSensor,0)  //we know they have to be the same number of points
    // Create a ForceScope wave to store ZSensor, Amplitude, and Phase of the force curve
    Make/N=(nop*1,3)/O $DataFolder+"ForceScope"/Wave=ForceScope
    ForceScope[0,nop-1][0] = ZSensor[P]
    ForceScope[0,nop-1][1] = Amplitude[P]
	ForceScope[0,nop-1][2] = Phase[P]
    Save/C/O/P=SaveImage ForceScope as FileName
End //GetForce

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



Function MoveStage(Direction)
    String Direction
    Struct WMButtonAction ButtonStruct
    
    ButtonStruct.EventCode = 2
    //  ButtonStruct.Win = "MasterPanel"
    ButtonStruct.CtrlName = Direction
    StageButtonFunc(ButtonStruct)
End //

Function OverwriteZSensorScope()
    Variable/G Index
    Variable BufferSize = 20
    Index = Mod(Index+1, BufferSize)
    String FileName = "C:\\Users\\Asylum User\\Documents\\AEtesting\\data_exchange\\Scope" + Num2str(Index) + ".ibw"
    String DataFolder = GetDF("Main")
	// AC or SKPM mode
	
		// Take out Z Sensor trace and retrace
		Wave ZSensorScopeTrace = $DataFolder+"ZSensorScope0"
		Wave ZSensorScopeRetrace = $DataFolder+"ZSensorScope1"
		//  Variable nop = DimSize(ZSensorScopeTrace,0)  //we know they have to be the same number of points
		// Take out Amplitude trace and retrace
		Wave AmplitudeScopeTrace = $DataFolder+"AmplitudeScope0"
		Wave AmplitudeScopeRetrace = $DataFolder+"AmplitudeScope1"
		Variable nop = DimSize(AmplitudeScopeTrace,0)  //we know they have to be the same number of points
		// Take out Phase trace and retrace
		Wave PhaseScopeTrace = $DataFolder+"PhaseScope0"
		Wave PhaseScopeRetrace = $DataFolder+"PhaseScope1"
		Wave HeightScopeTrace = $DataFolder+"HeightScope0"
		Wave HeightScopeRetrace = $DataFolder+"HeightScope1"
		// Make an image array to store all the lines
		Make/N=(nop*1,8)/O $DataFolder+"ZSensorFullScope"/Wave=ZSensorScope
		ZSensorScope[0,nop-1][0] = ZSensorScopeTrace[P]
		ZSensorScope[0,nop-1][1] = ZSensorScopeRetrace[P]
		ZSensorScope[0,nop-1][2] = AmplitudeScopeTrace[P]
		ZSensorScope[0,nop-1][3] = AmplitudeScopeRetrace[P]
		ZSensorScope[0,nop-1][4] = PhaseScopeTrace[P]
		ZSensorScope[0,nop-1][5] = PhaseScopeRetrace[P]
		ZSensorScope[0,nop-1][6] = HeightScopeTrace[P]
		ZSensorScope[0,nop-1][7] = HeightScopeRetrace[P]
		// print GV("ImagingMode")

	Save/C/O/P=SaveImage ZSensorScope as FileName
	
End //OverwriteZSensorScope


Function SetDriveAmpAndSetpoint(NewDriveAmp,NewSetpoint)
            	Variable NewDriveAmp, NewSetpoint
                
            	String Callback = ""
            	Callback += "SetDriveAmpAndSetpoint_CB"+"("
            	Callback += num2str(NewDriveAmp)+","
            	Callback += num2str(NewSetpoint)+")"
                
            	String SetpointAddress = "$HeightLoop.Setpoint"
            	if (StringMatch(ir_ReadAlias(SetpointAddress),"Cypher.*"))
                            	SetpointAddress = "$HeightLoop.SetpointOffset"                                    	//backpack ramp can't ramp a setpoint
                            	NewSetpoint = NewSetpoint-td_ReadValue("$HeightLoop.Setpoint")
            	endif
            	String DriveAmpAddress = "$Lockin.0.Amp"
                
            	Variable RunTime = .25                            	//seconds
            	Variable Bank = 1                        	//Standard scanning uses bank 0, Nap Scanning uses bank 2, so bank 1 is only used when the user is changing scan size / or scan angle.
            	String ErrorStr = ""
                
            	String Event = "12"
            	ErrorStr += Ramp2Channels(SetpointAddress,NewSetpoint,DriveAmpAddress,NewDriveAmp,RunTime,Bank,Event,Callback)
            	ErrorStr += ir_WriteString("Event."+Event,"Once")
 
            	ARReportError(ErrorStr)
                
            	return 0
End //SetDriveAmpAndSetpoint
 
Function SetDriveAmpAndSetpoint_CB(NewDriveAmp,NewSetpoint)
            	Variable NewDriveAmp, NewSetpoint
                
            	//the controller is in the correct state, this is to get the UI updated, which is easier to just send the commands to the controller again
            	//but that is not really great, it would be better to bypass sending the values back to the controller
            	//but then again, we don't have any limit safety on the parameters....
                
            	Variable FixCypherSetpoint = True
            	if (FixCypherSetpoint)
                            	//if we had to ramp the setpoint offset, because it is on the backpack, then set them both to their correct values at the same time
                            	String Address = "$HeightLoop.Setpoint"
                            	Address = ir_ReadAlias(Address)
                            	if (StringMatch(Address,"Cypher.*"))
                                            	Wave/T Group = NewFreeWave(0x0,0)
                                            	String ErrorStr = ir_ReadGroup("$HeightLoop",Group)
                                            	Group[%Setpoint] = num2str(Str2num(Group[%Setpoint])+Str2num(Group[%SetpointOffset]))
                                            	Group[%SetpointOffset] = "0"
                                            	ErrorStr += ir_WriteGroup("$HeightLoop",Group)
                                            	ARReportError(ErrorStr)
                            	endif
            	endif
                
            	CallSetVarSimple(FmapSetVarFunc,"AmplitudeSetpointVolts",Value=NewSetpoint)
            	CallSetVarSimple(FmapSetVarFunc,"DriveAmplitude",Value=NewDriveAmp)
                
End  //SetDriveAmpAndSetpoint_CB
 
Function/S Ramp2Channels(AddressA,DestA,AddressB,DestB,RampTime,Bank,Event,Callback)
            	String AddressA
            	Variable DestA
            	String AddressB
            	Variable DestB
            	Variable RampTime
            	Variable Bank
            	String Event
            	String Callback
                
            	Wave ARCDestWave = NewFreeWave(0x4,0)
            	Wave CypherDestWave = NewFreeWave(0x4,0)
            	String ARCList = ""
            	String CypherList = ""
            	Variable HasARC = False
                
            	AddressA = ir_ReadAlias(AddressA)
            	if (StringMatch(AddressA,"Cypher.*"))
                            	CypherList += AddressA+";"
                            	Add2Wave(CypherDestWave,DestA)
            	else
                            	ARCList += AddressA+";"
                            	Add2Wave(ARCDestWave,DestA)
                            	HasARC = True
            	endif
 
            	AddressB = ir_ReadAlias(AddressB)
            	if (StringMatch(AddressB,"Cypher.*"))
                            	CypherList += AddressB+";"
                            	Add2Wave(CypherDestWave,DestB)
            	else
                            	ARCList += AddressB+";"
                            	Add2Wave(ARCDestWave,DestB)
                            	HasARC = True
            	endif
 
            	String ErrorStr = ""
            	ErrorStr += RampARCItems(ARCList,ARCDestWave,RampTime,Bank,Event,Callback=Callback)
            	if (HasARC)
                            	Callback = ""
            	endif
 
            	ErrorStr += RampBackpackItems(CypherList,CypherDestWave,RampTime,Event,Callback=Callback)
            	return ErrorStr
End //Ramp2Channels
 
Function/S RampARCItems(AddressList,DestPositions,RampTime,Bank,Event,[Callback])
            	String AddressList
            	Wave DestPositions
            	Variable RampTime, Bank
            	String Event
            	String Callback
 
            	Variable NumOfChannels = ItemsInList(AddressList,";")
            	if (NumOfChannels == 0)
                            	return ""
            	endif
                
            	if (ParamIsDefault(Callback))
                            	Callback = ""
            	endif
                
            	Variable SampleRate = 4e3
            	Variable Deci = ARGetDeci(SampleRate)
            	SampleRate = cMasterSampleRate/Deci
            	Variable Points = Round(RampTime*SampleRate)
 
            	Wave OutWave = InitOrDefaultWave("Root:OutWave",Points)
            	LinSpace2(td_ReadValue(StringFromList(0,AddressList,";")),DestPositions[0],Points,OutWave)
            	String ErrorStr = ""
 
            	if (NumOfChannels == 2)
                            	Wave OutWave2 = InitOrDefaultWave("Root:OutWave2",Points)
                            	LinSpace2(td_ReadValue(StringFromList(1,AddressList,";")),DestPositions[1],Points,OutWave2)
                            	ErrorStr += ir_SetOutWavePair(Bank,Event,StringFromList(0,AddressList,";"),OutWave,StringFromList(1,AddressList,";"),OutWave2,callback,-Deci)
            	elseif (NumOfChannels == 1)
                            	ErrorStr += ir_SetOutWave(Bank,Event,StringFromList(0,AddressList,";"),OutWave,Callback,-Deci)
            	endif
            	return ErrorStr
End //RampARCItems
 
Function/S RampBackpackItems(AddressList,DestPositions,RampTime,Event,[Callback])
            	String AddressList
            	Wave DestPositions
            	Variable RampTime
            	String Event
            	String Callback
            	String Address2
            	Variable Dest2
                
            	Variable NumOfChannels = ItemsInList(AddressList,";")
            	if (NumOfChannels == 0)
                            	return ""
            	endif
 
            	if (ParamIsDefault(Callback))
                            	Callback = ""
            	endif
                
            	String ErrorStr = ""
            	if (NumOfChannels == 2)
                            	ErrorStr += BottledRampPain(StringFromList(0,AddressList,";"),Nan,DestPositions[0],RampTime,Event,Callback=callback,ChannelB=StringFromList(1,AddressList,";"),StopB=DestPositions[1])
            	else
                            	ErrorStr += BottledRampPain(StringFromList(0,AddressList,";"),Nan,DestPositions[0],RampTime,Event,Callback=callback)
            	endif
            	Return ErrorStr
End //RampBackpackItems

