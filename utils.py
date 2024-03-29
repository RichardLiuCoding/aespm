# @title
# utils.py

import paramiko
import Pyro5.api

def execute_exe_on_server(exe_path, args):
    uri = "PYRO:command.executor@10.128.163.6:9091" # Replace with your server's URI
    #   print("Connecting to server at", uri)
    command_executor = Pyro5.api.Proxy(uri)
    response = command_executor.execute_exe_with_args(exe_path, args)
    #   print("Server response:", response)

def main_exe_on_server():
    exe_path = r"C:\AsylumResearch\v19\RealTime\Igor Pro Folder\Igor.exe"
    args = r"C:\Users\Asylum User\Documents\AEtesting\ToIgor.arcmd"
    execute_exe_on_server(exe_path, args)

# host = '10.128.163.6'
# username = 'Asylum User'
# password = 'jupiter'
# connection, client = return_connection(host, username, password)

def return_connection(host: str='10.128.163.6', username: str='Asylum User', password: str='jupiter'):
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

