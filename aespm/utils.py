import paramiko
import Pyro5.api
import subprocess
import os
import platform

if platform.system() == 'Windows':
    buffer_path = os.path.join(os.path.expanduser('~'), 'Documents', 'buffer')
    command_buffer = os.path.join(buffer_path, 'ToIgor.arcmd')
    read_out_buffer = os.path.join(buffer_path, 'readout.txt')
    bash_buffer = os.path.join(buffer_path, 'SendToIgor.bat')
    
    if os.path.exists("C:\\AsylumResearch\\v19"):
        exe_path = "C:\\AsylumResearch\\v19\\RealTime\\Igor Pro Folder\\Igor.exe"
    elif os.path.exists("C:\\AsylumResearch\\v18"):
        exe_path = "C:\\AsylumResearch\\v18\\RealTime\\Igor Pro Folder\\Igor.exe"
    else:
        exe_path = ""

import aespm
from aespm import experiment

class SharedInfo:
    '''
    A class to pass host and local computer directory info across the package.
    '''
    def __init__(self):
        self._buffer_path = ''
        self._command_buffer = ''
        self._read_out_buffer = ''
        self._buffer_path = ''
        self._exe_path = ''
        self._host = ''

    def set_host(self, value):
        self._host = value
    
    def set_value(self, values):
        self._buffer_path, self._command_buffer, self._read_out_buffer, self._bash_buffer, self._exe_path = values

shared = SharedInfo()

@Pyro5.api.expose
class CommandExecutor:
    def execute_exe_with_args(self, exe_path, arg):
        try:
            p = subprocess.Popen([exe_path, arg], shell=True)
            p.wait()
        except Exception as e:
            return f"Execution Failed: {str(e)}"
    def get_local_dir(self):
        return "$".join([buffer_path, command_buffer, read_out_buffer, bash_buffer, exe_path])

def connect(host, port=9091):
    '''
    Start a python server on your local computer. This server can accept
    commands from a remote cluster.

    Args:
        host    - String: IP address of the local computer.
        port    - Int (Optional): the port for the server.
        
    Output: None
    Example:
        >>> import aespm as ae
        >>> host = 'your_ip_address'
        >>> ae.utils.connect(host=host)
    '''
    daemon = Pyro5.server.Daemon(host=host, port=port)  # Bind to the specific IP
    uri = daemon.register(CommandExecutor, "command.executor")
    # print("Server is ready. Object uri =", uri)
    daemon.requestLoop()  # Start the request loop

def get_local_directory(host=shared._host, port=9091):
    """Retrieve the buffer path on the local computer.

    Args:
        uri (string): address of server
        args (string): arguments to pass to the executable
    
    Returns: "$".join([buffer_path, command_buffer, read_out_buffer, bash_buffer, exe_path])
    """
    
    uri = "PYRO:command.executor@{}:{}".format(host, port)
    command_executor = Pyro5.api.Proxy(uri)
    response = command_executor.get_local_dir()
    return response

def execute_exe_on_server(exe_path, args, host=shared._host, port=9091):
    """Execute an executable on the remote server.

    Args:
        exe_path (string): path to the executable
        uri (string): address of server
        args (string): arguments to pass to the executable
    
    Returns: None
    """
    uri = "PYRO:command.executor@{}:{}".format(host, port)
    #   print("Connecting to server at", uri)
    command_executor = Pyro5.api.Proxy(uri)
    response = command_executor.execute_exe_with_args(exe_path, args)
    #   print("Server response:", response)

def main_exe_on_server(host=shared._host):
    """Execute the main executable on the remote server.
    """
    exe_path = shared._exe_path
    # args = aespm._command_buffer
    execute_exe_on_server(exe_path, args=shared._command_buffer, host=host)

def return_connection(host: str, username: str, password: str):
    """
    Return a connection object to the remote server.
    Args:
        host (str): IP address of the remote server.
        username (str): Username for the remote server.
        password (str): Password for the remote server.
    Returns:
        connection (object): Connection object to the remote server.
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)
    connection = client.open_sftp()
    return connection, client

def read_remote_file(connection: object, file_path: str):
    """
    Read content from the remote file.
    Args:
        connection (object): Connection object to the remote server.
        file_path (str): Path to the remote file.
    Returns:
        content (str): Content of the remote file.
    """
    with connection.open(file_path, 'r') as remote_file:
        content = remote_file.read()
    return content

def write_to_remote_file(connection: object, file_path: str, data: str):
    """
    Write data to the remote file.
    Args:
        connection (object): Connection object to the remote server.
        file_path (str): Path to the remote file.
        data (str): Data to be written to the remote file.
    Returns:
        None
    """
    with connection.open(file_path, 'w') as remote_file:
        remote_file.write(data)

def download_file(connection: object, file_path: str, local_file_name: str = "isaac_copy.ibw"):
    """
    Download a file from the remote server to the local machine.
    Args:
        connection (object): Connection object to the remote server.
        file_path (str): Path to the remote file.
        local_file_name (str): Name of the local file.
    Returns:
        None
    """
    connection.get(file_path, local_file_name)

