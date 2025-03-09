import torch
from pathlib import Path
from io import BufferedReader
from typing import Optional, Tuple

def _skip_whitespace(reader : BufferedReader):
    while chr(reader.peek()[0]).isspace():
        reader.read(1)
        
def _read_keyword(reader : BufferedReader):
    keyword = []
    while not chr(reader.peek()[0]).isspace():
        keyword.append(reader.read(1))
        
    breakpoint()
        
def load_ply(
    filename : str | Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Load a PLY file
        
        Parameters
        ----------
        filename : str | Path
            The path to the PLY file to load.
            
        Returns
        -------
    """
    with open(filename, "rb") as ply_file:
        # Read 'ply'
        magic_str = ply_file.read(3)
        print(magic_str)
        assert magic_str == b'ply', "Did not read 'ply' magic string in the file header"
        
        _skip_whitespace(ply_file)
            
        elements = []
        
        keyword = _read_keyword(ply_file)