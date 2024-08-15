import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
#For Python 3.8+, to avoid SyntaxWarning: "is" with a literal. Did you mean "=="?

__all__ = ['experiment']

# If there is no buffer files, create them in Documents/buffers
import os 
import shutil
import platform

doc_path = os.path.expanduser('~')
buffer_path = os.path.join(doc_path, 'Documents', 'buffer')
    
# Initialize the buffer files only on Windows operating system

if platform.system() == 'Windows':
    # Load the path from file if there is a path.txt in the buffer folder
    path_txt = os.path.join(buffer_path, 'path.txt')
    
    command_buffer = os.path.join(buffer_path, 'ToIgor.arcmd')
    read_out_buffer = os.path.join(buffer_path, 'readout.txt')
    bash_buffer = os.path.join(buffer_path, 'SendToIgor.bat')
    
    try:
        # Create the buffer folder
        if not os.path.exists(buffer_path):
            os.makedirs(buffer_path)

        # Create the commands buffer file
        if not os.path.exists(command_buffer):
            with open(command_buffer, 'w') as fopen:
                fopen.write(' ')
                
        if os.path.exists(path_txt):
            # Load the exe path from the file
            with open(path_txt, 'r') as fopen:
                exe_path = fopen.readline()
            
            # Write the exe_path into the bash buffer
            with open(bash_buffer, 'w') as fopen:
                fopen.write('"{}" "{}"'.format(exe_path, command_buffer))
        else:
            # Fist-time installation: no path_txt file available
            # Detect the AR version and installation path automatically
            if os.path.exists("C:\\AsylumResearch\\v19"):
                exe_path = "C:\\AsylumResearch\\v19\\RealTime\\Igor Pro Folder\\Igor.exe"
            elif os.path.exists("C:\\AsylumResearch\\v18"):
                exe_path = "C:\\AsylumResearch\\v18\\RealTime\\Igor Pro Folder\\Igor.exe"
            elif os.path.exists("C:\\AsylumResearch\\v17"):
                exe_path = "C:\\AsylumResearch\\v17\\RealTime\\Igor Pro Folder\\Igor.exe"
            elif os.path.exists("C:\\AsylumResearch\\v16"):
                exe_path = "C:\\AsylumResearch\\v16\\RealTime\\Igor Pro Folder\\Igor.exe"
            elif os.path.exists("C:\\AsylumResearch\\v15"):
                exe_path = "C:\\AsylumResearch\\v15\\RealTime\\Igor Pro Folder\\Igor.exe"
            elif os.path.exists("C:\\AsylumResearch\\v14"):
                exe_path = "C:\\AsylumResearch\\v14\\RealTime\\Igor Pro Folder\\Igor.exe"
            elif os.path.exists("C:\\AsylumResearch\\v13"):
                exe_path = "C:\\AsylumResearch\\v13\\RealTime\\Igor Pro Folder\\Igor.exe"
            else:
                print("No AR detected. Please modify path.txt manually to include the AR path.")
        
            with open(path_txt, 'w') as fopen:
                fopen.write(exe_path)
                
            with open(bash_buffer, 'w') as fopen: 
                fopen.write('"{}" "{}"'.format(exe_path, command_buffer))

            # Copy the user functions into the default include folder of AR
            # shutil.copy(os.path.join(aespm.__path__, 'user functions', 'UserFunctions.ipf'), os.path.join(doc_path, 'AsylumResearch', 'UserIncludes'))

    except PermissionError:
        print('No writing permission to ~/Documents. Please create buffer folder and files manually.')
        
from aespm.experiment import *

from aespm import tools
from aespm import utils
# 	try:
# 		# Create the buffer folder
# 		if not os.path.exists(buffer_path):
# 			os.makedirs(buffer_path)

# 		# Create the commands buffer file
# 		if not os.path.exists(os.path.join(buffer_path, 'ToIgor.arcmd')):
# 			with open(os.path.join(buffer_path, 'ToIgor.arcmd'), 'w') as fopen:
# 				fopen.write(' ')
# 		# Create bash file to execute commands in hte buffer file
# 		if not os.path.exists(os.path.join(buffer_path, 'SendToIgor.bat')):
# 			with open(os.path.join(buffer_path, 'SendToIgor.bat'), 'w') as fopen:
# 				if os.path.exists("C:\\Asylum\\Igor Pro Folder AR15\\"):
# 					fopen.write('"C:\\Asylum\\Igor Pro Folder AR15\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				# elif os.path.exists("C:\\AsylumResearch\\v18"):
# 				# 	fopen.write('"C:\\AsylumResearch\\v18\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				# elif os.path.exists("C:\\AsylumResearch\\v17"):
# 				# 	fopen.write('"C:\\AsylumResearch\\v17\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				# elif os.path.exists("C:\\AsylumResearch\\v16"):
# 				# 	fopen.write('"C:\\AsylumResearch\\v16\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				# elif os.path.exists("C:\\AsylumResearch\\v15"):
# 				# 	fopen.write('"C:\\AsylumResearch\\v15\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				# elif os.path.exists("C:\\AsylumResearch\\v14"):
# 				# 	fopen.write('"C:\\AsylumResearch\\v14\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				# elif os.path.exists("C:\\AsylumResearch\\v13"):
# 				# 	fopen.write('"C:\\AsylumResearch\\v13\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
# 				else:
# 					print("No supported AR versions!")

# 		# Copy the user functions into the default include folder of AR
# 		# shutil.copy(os.path.join(aespm.__path__, 'user functions', 'UserFunctions.ipf'), os.path.join(doc_path, 'AsylumResearch', 'UserIncludes'))

# 	except PermissionError:
# 		print('No writing permission to ~/Documents. Please create buffer folder and files manually.')

