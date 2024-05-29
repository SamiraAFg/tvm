from enum import Enum

class memcmd(Enum):
    LOAD_A=0x01000000 #first element goes to the LSB
    LOAD_B=0x01000004
    STORE_RES=0x01000009

class gemcmd(Enum):
    GEMM=0x02
