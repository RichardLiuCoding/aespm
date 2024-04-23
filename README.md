# aespm -- "A"utomated "E"xperiment on "SPM"

Python interface that enables local and remote control of Scanning Probe Microscope (SPM) with codes.

It offers a modular way to write autonomous workflows ranging from simple routine operations, to advanced automatic scientific discovery based on machine learning.

# Installation

```Python
pip install aespm
```

For AR SPM users, download the "UserFunctions.ipf" from [**aespm/user functions**](https://github.com/RichardLiuCoding/aespm/blob/main/aespm/user%20functions/UserFunctions.ipf) folder and put it in "Documents/AsylumResearch/UserIncludes" folder so that it will automatically loaded at AR startup. You only need to do it one time after installing aespm.

ps, if you software is already running, you need to click "Programming -> Rescan User Includes" to make this change effective.


# Examples

For more detailed tutorials and example workflows, please take a look at [**aespm/notebooks**](https://github.com/RichardLiuCoding/aespm/tree/main/aespm/notebooks) folder.

## Create an aespm.Experiment() object

```Python
import aespm as ae
folder = 'where_you_save_your_data'
exp = ae.Experiment(folder=folder)
```

There is a list of default actions in the end of this page.

## Work from a remote cluster

Start a new anaconda terminal and run the following codes on your **local** computer:

```Python
import aespm as ae
host = "your_ip_address"
ae.utils.connect(host=host)
```

On the **remote** server, you only need the following information to build the connection:

```Python
host = 'IP_address_local_desktop'
username = 'your_login_name'
password = 'your_login_credential'

exp = ae.Experiment(folder=folder, connection=[host, username, password])
```

Then all your local notebooks should run automatically with this exp object!

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
    img = ae.ibw_read(ae.get_files(folder=self.folder)[0])
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

## Functional blocks and workflows

Functional blocks are defined based on a list of sequential actions that achieve one major task:
* Start an AC scan and load and plot the acquired image when it's done
* Move the probe to the location (x, y) and take a force-distance curve and extract final force when it's done.

These functional blocks can be created with **Experiment.execute_sequence()** and **Experiment.add_func()** methods. 
Once a functional block is appened to Experiment object, it can be called the same way as default single actions. 
Therefore, the full workflow can be built upon functional blocks, which makes it organized, concise, and easy to debug.

An example of AC exploration on a grid workflow build on functional blocks:
```Python
for i in range(num_points):
    print("Working on Location: {}/{}".format(i+1, num_points), end='\r')
   
    # Skip the first point
    if i:
        # Move the stage to the next grid point
        exe.execute('stage', value=[displacement])

    # AC scan
    exe.execute('start_ac_scan', value=['BSFO_AC{:03}_'.format(i+1)])
```


