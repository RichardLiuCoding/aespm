import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
#For Python 3.8+, to avoid SyntaxWarning: "is" with a literal. Did you mean "=="?

__all__ = ['aespm']

from aespm.aespm import *

from aespm import tools
from aespm import utils

# If there is no buffer files, create them in Documents/buffers
import os 
import shutil
import platform

doc_path = os.path.expanduser('~')
buffer_path = os.path.join(doc_path, 'Documents', 'buffer')

# Initialize the buffer files only on Windows operating system

if platform.system() == 'Windows':
	try:
		# Create the buffer folder
		if not os.path.exists(buffer_path):
			os.makedirs(buffer_path)

		# Create the commands buffer file
		if not os.path.exists(os.path.join(buffer_path, 'ToIgor.arcmd')):
			with open(os.path.join(buffer_path, 'ToIgor.arcmd'), 'w') as fopen:
				fopen.write(' ')
		# Create bash file to execute commands in hte buffer file
		if not os.path.exists(os.path.join(buffer_path, 'SendToIgor.bat')):
			with open(os.path.join(buffer_path, 'SendToIgor.bat'), 'w') as fopen:
				if os.path.exists("C:\\AsylumResearch\\v19"):
					fopen.write('"C:\\AsylumResearch\\v19\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
				elif os.path.exists("C:\\AsylumResearch\\v18"):
					fopen.write('"C:\\AsylumResearch\\v18\\RealTime\\Igor Pro Folder\\Igor.exe" "{}"'.format(os.path.join(buffer_path, 'ToIgor.arcmd')))
				else:
					print("No supported AR versions!")

		# Copy the user functions into the default include folder of AR
		# shutil.copy(os.path.join(aespm.__path__, 'user functions', 'UserFunctions.ipf'), os.path.join(doc_path, 'AsylumResearch', 'UserIncludes'))

	except PermissionError:
		print('No writing permission to ~/Documents. Please create buffer folder and files manually.')