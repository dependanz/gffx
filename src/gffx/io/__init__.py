import torch
import struct
import numpy as np
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
    filename : str | Path,
    device : Optional[torch.device] = None,
    debug  : bool = False
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
                if debug:
                    print("Keyword: ", keyword)
                
                if keyword == b'format':
                    state = PLYState.FORMAT
                elif keyword == b'comment':
                    state = PLYState.COMMENT
                elif keyword == b'element':
                    state = PLYState.ELEMENT
                elif keyword == b'property':
                    state = PLYState.PROPERTY
                elif keyword == b'end_header':
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
                element_name = _read_word(ply_file).decode('utf-8')
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
                _skip_whitespace(ply_file)
                
                for element_name in element_names:
                    element_count = ply_meta['elements'][element_name]['count']
                    element_properties = ply_meta['elements'][element_name]['properties']
                    
                    elements = []
                    for i in range(element_count):
                        elements.append([])
                        for property in element_properties:
                            if property['type'] == b'list':
                                # List Count
                                if property['count_type'] == b'uchar':
                                    count = ply_file.read(1)
                                    count = struct.unpack(f"<B", count)[0]
                                else:
                                    raise ValueError(f"Unsupported list count type: {property['count_type']}")
                                
                                # List Data
                                list_data = []
                                for j in range(count):
                                    if property['property_type'] == b'int':
                                        datum = ply_file.read(4)
                                        if ply_meta["format"] == b'binary_little_endian':
                                            datum = struct.unpack(f"<i", datum)[0]
                                        elif ply_meta["format"] == b'binary_big_endian':
                                            datum = struct.unpack(f">i", datum)[0]
                                        else:
                                            raise ValueError(f"Unsupported format: {ply_meta['format']}")
                                        list_data.append(datum)
                                    else:
                                        raise ValueError(f"Unsupported list property type: {property['property_type']}")
                                elements[-1].append(list_data)
                            else:
                                if property['type'] == b'float':
                                    datum = ply_file.read(4)
                                    if ply_meta["format"] == b'binary_little_endian':
                                        datum = struct.unpack(f"<f", datum)[0]
                                    elif ply_meta["format"] == b'binary_big_endian':
                                        datum = struct.unpack(f">f", datum)[0]
                                    else:
                                        raise ValueError(f"Unsupported format: {ply_meta['format']}")
                                    elements[-1].append(datum)
                                else:
                                    raise ValueError(f"Unsupported property type: {property['type']}")
                                
                    ply_meta['elements'][element_name]['data'] = elements
                state = None
                  
    # [DEBUG] Print metadata
    if debug:
        print("PLY Metadata:")
        print("Format: ", ply_meta['format'])
        print("Version: ", ply_meta['version'])
        print("Elements:")
        for element_name in ply_meta['elements']:
            print(f"  {element_name}:")
            print(f"    Count: {ply_meta['elements'][element_name]['count']}")
            print(f"    Properties:")
            for property in ply_meta['elements'][element_name]['properties']:
                print(f"      {property['name']}: {property['type']}")
                  
    # [TODO] Generalize, but for now just assume vertices and faces
    vertices = np.array(ply_meta['elements']['vertex']['data'])
    faces    = np.array(ply_meta['elements']['face']['data'])
    
    # Flatten 1 dim components
    i = 0 
    while i < len(vertices.shape):
        if vertices.shape[i] == 1:
            vertices = np.squeeze(vertices, axis=i)
            i = 0
        else:
            i += 1
    i = 0
    while i < len(faces.shape):
        if faces.shape[i] == 1:
            faces = np.squeeze(faces, axis=i)
            i = 0
        else:
            i += 1
            
    # Convert to torch tensors
    vertices = torch.from_numpy(vertices).float().to(device)
    faces    = torch.from_numpy(faces).long().to(device)
    
    return vertices, faces