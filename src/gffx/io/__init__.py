import torch
import struct
from enum import Enum
from pathlib import Path
from io import BufferedReader
from typing import Optional, Tuple

def _skip_whitespace(reader : BufferedReader):
    while chr(reader.peek()[0]).isspace():
        reader.read(1)
        
def _read_word(reader : BufferedReader):
    _skip_whitespace(reader)
    
    word = []
    while not chr(reader.peek()[0]).isspace():
        word.append(reader.read(1))
    word = b''.join(word)
    
    return word
        
class PLYState(Enum):
    KEYWORD = 0
    FORMAT = 1
    COMMENT = 2
    ELEMENT = 3
    PROPERTY = 4
    LIST = 5
    DATA = 6
    
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
        
        ply_meta = {
            'format': None,
            'version': None,
            'elements': {}
        }
        element_names = []
        state = PLYState.KEYWORD
        while state:
            if state == PLYState.KEYWORD:
                keyword = _read_word(ply_file)
                print(keyword)
                
                if keyword == b'format':
                    state = PLYState.FORMAT
                elif keyword == b'comment':
                    state = PLYState.COMMENT
                elif keyword == b'element':
                    state = PLYState.ELEMENT
                elif keyword == b'property':
                    state = PLYState.PROPERTY
                elif keyword == b'end_header':
                    breakpoint()
                    state = PLYState.DATA
            
            elif state == PLYState.FORMAT:
                ply_meta['format'] = _read_word(ply_file)
                ply_meta['version'] = _read_word(ply_file)
                
                state = PLYState.KEYWORD

            elif state == PLYState.COMMENT:
                while ply_file.peek()[0] != ord('\n'):
                    ply_file.read(1)
                state = PLYState.KEYWORD
                
            elif state == PLYState.ELEMENT:
                element_name = _read_word(ply_file)
                element_count = int(_read_word(ply_file))
                
                ply_meta['elements'][element_name] = {}
                ply_meta['elements'][element_name]['count'] = element_count
                ply_meta['elements'][element_name]['properties'] = []
                element_names.append(element_name)
                state = PLYState.KEYWORD
                
            elif state == PLYState.PROPERTY:
                property_type = _read_word(ply_file)

                if property_type == b'list':
                    list_count_type = _read_word(ply_file)
                    list_property_type = _read_word(ply_file)
                    property_name = _read_word(ply_file)
                    ply_meta['elements'][element_names[-1]]['properties'].append({
                        'type'          : property_type,
                        'count_type'    : list_count_type,
                        'property_type' : list_property_type,
                        'name'          : property_name
                    })
                else:
                    property_name = _read_word(ply_file)
                    
                    ply_meta['elements'][element_names[-1]]['properties'].append({
                        'type' : property_type,
                        'name' : property_name
                    })
                state = PLYState.KEYWORD
                
            elif state == PLYState.DATA:
                for element_name in element_names:
                    element_count = ply_meta['elements'][element_name]['count']
                    element_properties = ply_meta['elements'][element_name]['properties']
                    
                    elements = []
                    for i in range(element_count):
                        elements.append([])
                        for property in element_properties:
                            if property['type'] == b'list':
                                count = _read_word(ply_file)
                                breakpoint()
                            else:
                                if property['type'] == b'float':
                                    datum = ply_file.read(4)
                                    datum = struct.unpack(f'{">" if ply_meta["format"] == b"binary_big_endian" else "<"}f', datum[::-1])[0]
                                    elements[-1].append(datum)
                        breakpoint()
                state = None